#!/usr/bin/env python3
"""测试关键问题：删除后重新插入的bug"""

import sys
sys.path.insert(0, 'clients/dxmt-python-sdk')

from dxmt.client import DXMTClient
import time

def test_delete_reinsert_bug():
    """测试问题1：删除后重新插入相同ID"""
    print("\n" + "="*80)
    print("测试问题1: 删除后重新插入Bug (向量索引删除缺失)")
    print("="*80)
    
    client = DXMTClient(host='localhost', port=9999)
    
    # 创建测试表
    table_name = 'test_delete_bug'
    try:
        client.delete_table(table_name)
    except:
        pass
    
    print("\n1. 创建表...")
    client.create_table(
        name=table_name,
        dimension=3,
        distance_metric='cosine'
    )
    print("   ✓ 表创建成功")
    
    # 插入记录
    record_id = 'test_record_001'
    print(f"\n2. 插入记录 ID={record_id}...")
    client.insert_records(
        table_name,
        [{'id': record_id, 'vector': [1.0, 2.0, 3.0], 'payload': {'version': 1}}]
    )
    print("   ✓ 记录插入成功")
    
    # 验证记录存在
    print("\n3. 验证记录存在...")
    records = client.get_records(table_name, [record_id])
    print(f"   ✓ 找到记录: {records[0].get('payload')}")
    
    # 删除记录
    print(f"\n4. 删除记录 ID={record_id}...")
    result = client.delete_records(table_name, [record_id])
    print(f"   ✓ 删除结果: {result}")
    
    # 验证记录已删除
    print("\n5. 验证记录已删除...")
    records = client.get_records(table_name, [record_id])
    if len(records) == 0:
        print("   ✓ 记录已从HashMap删除")
    else:
        print(f"   ✗ 警告：记录仍存在: {records}")
    
    # 等待一段时间
    print("\n6. 等待1秒...")
    time.sleep(1)
    
    # 尝试重新插入相同ID
    print(f"\n7. 尝试重新插入相同ID={record_id}...")
    try:
        client.insert_records(
            table_name,
            [{'id': record_id, 'vector': [4.0, 5.0, 6.0], 'payload': {'version': 2}}]
        )
        print("   ✓ 重新插入成功！")
        print("   → 问题已修复或未触发")
        
        # 验证新数据
        records = client.get_records(table_name, [record_id])
        print(f"   ✓ 新记录payload: {records[0].get('payload')}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if 'already exists' in error_msg.lower():
            print("   ✗ 重新插入失败：ID已存在")
            print(f"   → 错误信息: {error_msg[:200]}")
            print("\n结论：")
            print("   🔴 Bug确认：向量索引未删除记录")
            print("   📍 位置：src/storage/intelligent_indexed_storage.rs:1588")
            print("   📝 原因：delete_records()未调用index.remove()")
            return False
        else:
            print(f"   ✗ 意外错误: {e}")
            return False
    finally:
        # 清理
        try:
            client.delete_table(table_name)
        except:
            pass


def test_langgraph_basic():
    """测试LangGraph集成基本功能"""
    print("\n" + "="*80)
    print("测试LangGraph集成基本功能")
    print("="*80)
    
    from dxmt.langgraph_store import DXMTLangGraphStore
    
    client = DXMTClient(host='localhost', port=9999)
    store = DXMTLangGraphStore(client)
    
    # 清理旧数据
    print("\n清理旧测试数据...")
    tables = client.list_tables()
    for table in tables:
        if table.get('name', '').startswith('langgraph_'):
            try:
                client.delete_table(table['name'])
            except:
                pass
    
    try:
        # 测试1: 基本put/get
        print("\n1. 测试基本put/get...")
        store.put(("test", "user1"), "profile", {"name": "Alice", "age": 30})
        item = store.get(("test", "user1"), "profile")
        assert item is not None
        assert item.value["name"] == "Alice"
        print("   ✓ Put/Get 正常工作")
        
        # 测试2: 删除
        print("\n2. 测试删除...")
        store.delete(("test", "user1"), "profile")
        item = store.get(("test", "user1"), "profile")
        assert item is None
        print("   ✓ Delete 正常工作")
        
        # 测试3: 命名空间列表
        print("\n3. 测试命名空间...")
        store.put(("test", "user2"), "profile", {"name": "Bob"})
        store.put(("prod", "user1"), "profile", {"name": "Charlie"})
        namespaces = store.list_namespaces()
        print(f"   ✓ 找到 {len(namespaces)} 个命名空间")
        
        print("\n✅ LangGraph基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ LangGraph测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("DXMT关键问题验证测试")
    print("="*80)
    
    results = {}
    
    # 测试问题1
    results['delete_bug'] = test_delete_reinsert_bug()
    
    # 测试LangGraph集成
    results['langgraph'] = test_langgraph_basic()
    
    # 总结
    print("\n" + "="*80)
    print("测试结果总结")
    print("="*80)
    print(f"问题1 (删除Bug):      {'✅ 已修复' if results['delete_bug'] else '❌ 确认存在'}")
    print(f"LangGraph集成:        {'✅ 正常' if results['langgraph'] else '❌ 有问题'}")
    
    if not results['delete_bug']:
        print("\n⚠️  建议立即修复删除Bug：")
        print("   1. 修改 IntelligentTableData.index 为 Arc<RwLock<Box<dyn VectorIndex>>>")
        print("   2. 在 delete_records() 中添加 index.remove(id)")
    
    return all(results.values())


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""测试纯元数据搜索功能"""

import sys
sys.path.insert(0, 'dxmt-python-sdk')

from dxmt.client import DXMTClient
import json

def test_list_records_api():
    """测试新的list_records API"""
    print("\n" + "="*80)
    print("测试问题2: 纯元数据搜索API")
    print("="*80)
    
    client = DXMTClient(host='localhost', port=9999)
    
    # 创建测试表
    table_name = 'test_metadata_search'
    try:
        client.delete_table(table_name)
    except:
        pass
    
    print("\n1. 创建表并插入测试数据...")
    client.create_table(
        name=table_name,
        dimension=3,
        distance_metric='cosine'
    )
    
    # 插入多条记录
    records = [
        {'id': 'user1', 'vector': [1.0, 0.0, 0.0], 'payload': {'role': 'admin', 'age': 30}},
        {'id': 'user2', 'vector': [0.0, 1.0, 0.0], 'payload': {'role': 'user', 'age': 25}},
        {'id': 'user3', 'vector': [0.0, 0.0, 1.0], 'payload': {'role': 'admin', 'age': 35}},
        {'id': 'user4', 'vector': [1.0, 1.0, 0.0], 'payload': {'role': 'user', 'age': 28}},
    ]
    client.insert_records(table_name, records)
    print(f"   ✓ 插入 {len(records)} 条记录")
    
    # 测试新的list_records API
    print("\n2. 测试list_records API (POST /tables/{name}/records/list)...")
    
    try:
        # 构造请求
        payload = {
            "filter": {
                "must": [
                    {"key": "role", "operation": {"type": "Equal", "value": "admin"}}
                ]
            },
            "limit": 100,
            "offset": 0
        }
        
        # 发送请求
        import httpx
        http_client = httpx.Client(base_url="http://localhost:9999/api/v1")
        response = http_client.post(
            f"/tables/{table_name}/records/list",
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            records = result.get('data', {}).get('records', [])
            print(f"   ✓ API调用成功，找到 {len(records)} 条记录")
            print(f"   ✓ 记录详情:")
            for r in records:
                print(f"      - ID: {r['id']}, Payload: {r.get('payload_json', r.get('payload'))}")
            
            # 验证结果
            if len(records) == 2:  # 应该找到2个admin
                print("\n   ✅ 纯元数据搜索功能正常工作！")
                return True
            else:
                print(f"\n   ⚠️  期望找到2条记录，实际找到{len(records)}条")
                return False
                
        elif response.status_code == 404:
            print(f"   ✗ API端点不存在 (404)")
            print(f"   → 说明：list_records API尚未部署")
            print(f"   → 状态：代码已实现，需要重新编译服务器")
            return False
        else:
            print(f"   ✗ API调用失败: {response.status_code}")
            print(f"   → 响应: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理
        try:
            client.delete_table(table_name)
        except:
            pass


def main():
    result = test_list_records_api()
    
    print("\n" + "="*80)
    print("问题2测试结果")
    print("="*80)
    
    if result:
        print("✅ 纯元数据搜索API工作正常")
    else:
        print("❌ API未部署或有问题")
        print("\n下一步：")
        print("1. 修复编译错误 (src/main.rs 中的 vectordb::Error)")
        print("2. 重新编译服务器")
        print("3. 部署新版本")
        print("4. 重新测试")
    
    return result


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

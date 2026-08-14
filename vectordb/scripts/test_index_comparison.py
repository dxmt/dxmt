#!/usr/bin/env python3
"""
对比不同索引类型的插入性能
验证 HNSW 是否是瓶颈
"""
import requests
import random
import time

BASE_URL = "http://localhost:9999/api/v1"

def test_index_performance(batch_sizes, dimension=128):
    """测试不同批次大小在不同索引下的性能"""

    print("\n" + "="*80)
    print("索引类型性能对比")
    print("="*80)

    results = {}

    for batch_size in batch_sizes:
        print(f"\n测试批次大小: {batch_size}")
        print("-" * 80)

        # 测试每种配置
        configs = [
            ("default", {}),  # 使用默认索引选择（HNSW）
        ]

        batch_results = {}

        for config_name, config_params in configs:
            # 创建测试表
            table_name = f"idx_{config_name}_{batch_size}_{int(time.time())}"

            response = requests.post(
                f"{BASE_URL}/tables",
                json={
                    "name": table_name,
                    "dimension": dimension,
                    "distance_metric": "cosine",
                    **config_params
                }
            )

            if response.status_code != 200:
                print(f"  {config_name}: 创建表失败")
                continue

            # 生成记录
            records = [
                {
                    "id": f"vec_{i}",
                    "vector": [random.random() for _ in range(dimension)],
                    "payload": {"index": i}
                }
                for i in range(batch_size)
            ]

            # 插入记录
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/tables/{table_name}/records",
                json={"records": records}
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                throughput = batch_size / elapsed
                batch_results[config_name] = {
                    "time": elapsed,
                    "throughput": throughput
                }
                print(f"  {config_name:15s}: {elapsed:7.3f}s | {throughput:8.0f} VPS")
            else:
                print(f"  {config_name}: 插入失败 - {response.status_code}")

        results[batch_size] = batch_results

    # 总结
    print("\n" + "="*80)
    print("性能总结")
    print("="*80)
    print("\n{:>10s} {:>20s}".format("批次大小", "默认(HNSW) VPS"))
    print("-" * 80)

    for batch_size in batch_sizes:
        if batch_size in results and "default" in results[batch_size]:
            default_throughput = results[batch_size]["default"]["throughput"]
            print(f"{batch_size:>10d} {default_throughput:>20.0f}")

    print("\n建议:")
    print("  • 如果看到性能随批次大小显著下降，说明索引构建是瓶颈")
    print("  • 解决方案: 实现延迟索引构建或优化 HNSW 批量插入")

if __name__ == "__main__":
    # 测试较小的批次范围以快速验证
    batch_sizes = [50, 100, 500, 1000, 2000]
    test_index_performance(batch_sizes)

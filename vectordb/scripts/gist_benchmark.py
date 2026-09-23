#!/usr/bin/env python3
"""
GIST-960 数据集基准测试（通过 HTTP API，存储由服务端配置决定）

测试项目：
1. 插入性能（不同批次大小）
2. 搜索性能（不同数据规模）
3. 召回率测试
4. 延迟分布（P50/P95/P99）
5. 并发性能

GIST 数据集规格：
- 维度: 960
- 距离度量: 欧氏距离
- 训练集: 1M 向量
- 测试集: 1K 查询向量
"""

import requests
import time
import numpy as np
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json

BASE_URL = "http://localhost:9999/api/v1"
DIM = 960  # GIST 维度
DISTANCE_METRIC = "euclidean"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """打印标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}{Colors.END}\n")

def print_success(text):
    """打印成功消息"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    """打印警告消息"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    """打印错误消息"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def generate_gist_like_vector(seed=None):
    """生成类似 GIST 的向量（960维，模拟真实分布）"""
    if seed is not None:
        np.random.seed(seed)
    # GIST 向量通常是归一化的特征向量
    vec = np.random.randn(DIM).astype(np.float32)
    # 归一化
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    return vec.tolist()

def test_insert_performance():
    """测试 1: 插入性能（不同批次大小）"""
    print_header("测试 1: 插入性能测试（GIST-960）")

    table_name = f"gist_insert_{int(time.time())}"

    # 创建表
    print("创建测试表...")
    response = requests.post(f"{BASE_URL}/tables", json={
        "name": table_name,
        "dimension": DIM,
        "distance_metric": DISTANCE_METRIC,
        "index_type": "hnsw"
    })

    if response.status_code != 200:
        print_error(f"创建表失败: {response.text}")
        return

    print_success(f"表 '{table_name}' 创建成功")

    # 测试不同批次大小
    batch_sizes = [10, 50, 100, 500, 1000, 2000, 5000]
    results = []

    print(f"\n{'批次大小':<12} {'耗时(s)':<12} {'吞吐量(vec/s)':<18} {'平均延迟(ms)':<15}")
    print("-" * 80)

    for batch_size in batch_sizes:
        # 生成测试数据
        records = [
            {
                "id": f"vec_{i}",
                "vector": generate_gist_like_vector(seed=i),
                "payload": {"batch": batch_size, "index": i}
            }
            for i in range(batch_size)
        ]

        # 插入测试
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/tables/{table_name}/records",
            json={"records": records}
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            throughput = batch_size / elapsed
            avg_latency = (elapsed * 1000) / batch_size

            print(f"{batch_size:<12} {elapsed:<12.3f} {throughput:<18.0f} {avg_latency:<15.3f}")

            results.append({
                "batch_size": batch_size,
                "elapsed": elapsed,
                "throughput": throughput,
                "avg_latency_ms": avg_latency
            })
        else:
            print_error(f"{batch_size:<12} 插入失败: {response.status_code}")

    # 删除测试表
    requests.delete(f"{BASE_URL}/tables/{table_name}")

    return results

def test_search_performance():
    """测试 2: 搜索性能（不同数据规模）"""
    print_header("测试 2: 搜索性能测试（不同数据规模）")

    # 测试不同数据规模
    data_sizes = [1000, 5000, 10000, 50000, 100000]
    search_results = []

    for data_size in data_sizes:
        table_name = f"gist_search_{data_size}_{int(time.time())}"

        # 创建表
        print(f"\n准备测试数据集（{data_size:,} 向量）...")
        response = requests.post(f"{BASE_URL}/tables", json={
            "name": table_name,
            "dimension": DIM,
            "distance_metric": DISTANCE_METRIC,
            "index_type": "hnsw"
        })

        if response.status_code != 200:
            print_error(f"创建表失败")
            continue

        # 批量插入数据
        batch_size = 5000
        num_batches = (data_size + batch_size - 1) // batch_size
        insert_start = time.time()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, data_size)
            batch_records = [
                {
                    "id": f"vec_{i}",
                    "vector": generate_gist_like_vector(seed=i),
                    "payload": {"index": i}
                }
                for i in range(start_idx, end_idx)
            ]

            response = requests.post(
                f"{BASE_URL}/tables/{table_name}/records",
                json={"records": batch_records}
            )

            if response.status_code != 200:
                print_error(f"批次 {batch_idx+1}/{num_batches} 插入失败")
                break

        insert_elapsed = time.time() - insert_start
        print_success(f"插入完成: {data_size:,} 向量，耗时 {insert_elapsed:.2f}s")

        # 生成查询向量
        num_queries = 100
        query_vectors = [generate_gist_like_vector(seed=1000000+i) for i in range(num_queries)]

        # 搜索测试
        print(f"执行 {num_queries} 次搜索查询...")
        latencies = []

        for query_vec in query_vectors:
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/tables/{table_name}/search",
                json={
                    "vector": query_vec,
                    "k": 10,
                    "with_payload": False
                }
            )
            latency = (time.time() - start) * 1000  # ms

            if response.status_code == 200:
                latencies.append(latency)

        if latencies:
            avg_latency = statistics.mean(latencies)
            p50 = statistics.median(latencies)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            qps = 1000 / avg_latency

            print(f"\n搜索性能统计（数据规模: {data_size:,}）:")
            print(f"  平均延迟: {avg_latency:8.2f} ms")
            print(f"  P50 延迟: {p50:8.2f} ms")
            print(f"  P95 延迟: {p95:8.2f} ms")
            print(f"  P99 延迟: {p99:8.2f} ms")
            print(f"  QPS:      {qps:8.0f} queries/sec")

            search_results.append({
                "data_size": data_size,
                "num_queries": len(latencies),
                "avg_latency_ms": avg_latency,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "qps": qps
            })

        # 删除测试表
        requests.delete(f"{BASE_URL}/tables/{table_name}")

    return search_results

def test_recall():
    """测试 3: 召回率测试（Recall@10）"""
    print_header("测试 3: 召回率测试（Recall@10）")

    table_name = f"gist_recall_{int(time.time())}"
    data_size = 10000  # 使用较小的数据集进行精确测试

    # 创建表
    print(f"创建测试表（{data_size:,} 向量）...")
    response = requests.post(f"{BASE_URL}/tables", json={
        "name": table_name,
        "dimension": DIM,
        "distance_metric": DISTANCE_METRIC,
        "index_type": "hnsw"
    })

    if response.status_code != 200:
        print_error("创建表失败")
        return None

    # 插入数据并保存向量（用于计算真实最近邻）
    vectors = []
    batch_size = 5000
    num_batches = (data_size + batch_size - 1) // batch_size

    print("插入数据...")
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, data_size)
        batch_records = []
        batch_vectors = []

        for i in range(start_idx, end_idx):
            vec = generate_gist_like_vector(seed=i)
            vectors.append(vec)
            batch_records.append({
                "id": f"vec_{i}",
                "vector": vec,
                "payload": {"index": i}
            })

        response = requests.post(
            f"{BASE_URL}/tables/{table_name}/records",
            json={"records": batch_records}
        )

    print_success(f"数据插入完成")

    # 生成查询向量并测试召回率
    num_queries = 50
    k = 10
    recalls = []

    print(f"\n执行召回率测试（{num_queries} 个查询，k={k}）...")

    vectors_np = np.array(vectors)

    for q_idx in range(num_queries):
        query_vec = generate_gist_like_vector(seed=2000000 + q_idx)
        query_np = np.array(query_vec)

        # 计算真实的 k 最近邻（暴力搜索）
        distances = np.linalg.norm(vectors_np - query_np, axis=1)
        true_neighbors = set(np.argsort(distances)[:k].tolist())

        # 使用 HNSW 搜索
        response = requests.post(
            f"{BASE_URL}/tables/{table_name}/search",
            json={
                "vector": query_vec,
                "k": k,
                "with_payload": False
            }
        )

        if response.status_code == 200:
            result_data = response.json()
            hits = result_data.get('data', {}).get('hits', [])
            found_neighbors = set([int(hit['id'].split('_')[1]) for hit in hits[:k]])

            # 计算召回率
            recall = len(true_neighbors & found_neighbors) / k
            recalls.append(recall)

    if recalls:
        avg_recall = statistics.mean(recalls)
        min_recall = min(recalls)
        max_recall = max(recalls)

        print(f"\n召回率统计:")
        print(f"  平均召回率: {avg_recall*100:6.2f}%")
        print(f"  最小召回率: {min_recall*100:6.2f}%")
        print(f"  最大召回率: {max_recall*100:6.2f}%")

        # 删除测试表
        requests.delete(f"{BASE_URL}/tables/{table_name}")

        return {
            "avg_recall": avg_recall,
            "min_recall": min_recall,
            "max_recall": max_recall
        }

    return None

def test_concurrent_search():
    """测试 4: 并发搜索性能"""
    print_header("测试 4: 并发搜索性能测试")

    table_name = f"gist_concurrent_{int(time.time())}"
    data_size = 50000

    # 创建表
    print(f"准备测试数据（{data_size:,} 向量）...")
    response = requests.post(f"{BASE_URL}/tables", json={
        "name": table_name,
        "dimension": DIM,
        "distance_metric": DISTANCE_METRIC,
        "index_type": "hnsw"
    })

    # 插入数据
    batch_size = 5000
    num_batches = (data_size + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, data_size)
        batch_records = [
            {
                "id": f"vec_{i}",
                "vector": generate_gist_like_vector(seed=i),
                "payload": {"index": i}
            }
            for i in range(start_idx, end_idx)
        ]

        requests.post(
            f"{BASE_URL}/tables/{table_name}/records",
            json={"records": batch_records}
        )

    print_success("数据准备完成")

    # 生成查询
    num_queries = 500
    queries = [generate_gist_like_vector(seed=3000000+i) for i in range(num_queries)]

    def execute_search(query_vec):
        """执行单次搜索"""
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/tables/{table_name}/search",
            json={
                "vector": query_vec,
                "k": 10,
                "with_payload": False
            }
        )
        latency = (time.time() - start) * 1000
        return latency if response.status_code == 200 else None

    # 测试不同并发数
    concurrent_results = []
    thread_counts = [1, 2, 4, 8, 16]

    print(f"\n{'线程数':<10} {'总QPS':<15} {'平均延迟(ms)':<18} {'P95延迟(ms)':<15}")
    print("-" * 80)

    for num_threads in thread_counts:
        start = time.time()
        latencies = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(execute_search, q) for q in queries]

            for future in as_completed(futures):
                latency = future.result()
                if latency is not None:
                    latencies.append(latency)

        elapsed = time.time() - start
        qps = len(latencies) / elapsed
        avg_latency = statistics.mean(latencies)
        p95 = np.percentile(latencies, 95)

        print(f"{num_threads:<10} {qps:<15.0f} {avg_latency:<18.2f} {p95:<15.2f}")

        concurrent_results.append({
            "threads": num_threads,
            "qps": qps,
            "avg_latency_ms": avg_latency,
            "p95_ms": p95
        })

    # 删除测试表
    requests.delete(f"{BASE_URL}/tables/{table_name}")

    return concurrent_results

def generate_report(insert_results, search_results, recall_result, concurrent_results):
    """生成测试报告"""
    print_header("GIST-960 性能测试报告")

    report = {
        "test_time": datetime.now().isoformat(),
        "config": {
            "dimension": DIM,
            "distance_metric": DISTANCE_METRIC,
            # The LMDB backend was removed; the server decides where rows live (DXMT_RECORD_STORE).
            "storage_type": "server-configured",
            "index_type": "HNSW"
        },
        "results": {
            "insert": insert_results,
            "search": search_results,
            "recall": recall_result,
            "concurrent": concurrent_results
        }
    }

    # 保存 JSON 报告
    report_file = f"gist_benchmark_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print_success(f"详细报告已保存到: {report_file}")

    # 打印摘要
    print(f"\n{Colors.BOLD}性能摘要:{Colors.END}")
    print("-" * 80)

    if insert_results:
        max_throughput = max(r['throughput'] for r in insert_results)
        print(f"插入性能:")
        print(f"  最大吞吐量: {max_throughput:,.0f} vectors/sec")

    if search_results:
        best_search = min(search_results, key=lambda x: x['avg_latency_ms'])
        print(f"\n搜索性能 (数据规模: {best_search['data_size']:,}):")
        print(f"  平均延迟: {best_search['avg_latency_ms']:.2f} ms")
        print(f"  P99 延迟: {best_search['p99_ms']:.2f} ms")
        print(f"  QPS: {best_search['qps']:.0f}")

    if recall_result:
        print(f"\n召回率:")
        print(f"  平均 Recall@10: {recall_result['avg_recall']*100:.2f}%")

    if concurrent_results:
        max_qps_result = max(concurrent_results, key=lambda x: x['qps'])
        print(f"\n并发性能 ({max_qps_result['threads']} 线程):")
        print(f"  最大 QPS: {max_qps_result['qps']:.0f}")
        print(f"  平均延迟: {max_qps_result['avg_latency_ms']:.2f} ms")

    print("-" * 80)

    return report_file

def main():
    """主测试流程"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 80)
    print("  GIST-960 基准测试")
    print("=" * 80)
    print(f"{Colors.END}")

    print(f"\n配置信息:")
    print(f"  向量维度: {DIM}")
    print(f"  距离度量: {DISTANCE_METRIC}")
    print("  存储类型: 由服务端配置决定（DXMT_RECORD_STORE / DXMT_VECTOR_STORE）")
    print(f"  索引类型: HNSW")

    # 运行测试
    insert_results = test_insert_performance()
    search_results = test_search_performance()
    recall_result = test_recall()
    concurrent_results = test_concurrent_search()

    # 生成报告
    report_file = generate_report(insert_results, search_results, recall_result, concurrent_results)

    print(f"\n{Colors.GREEN}{Colors.BOLD}✓ 所有测试完成！{Colors.END}")
    print(f"详细报告: {report_file}\n")

if __name__ == "__main__":
    main()

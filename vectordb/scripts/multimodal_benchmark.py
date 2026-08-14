#!/usr/bin/env python3
"""
DXMT VectorDB - 多模态功能性能基准测试

此脚本测试多模态功能的性能指标:
1. 编码延迟 (图像/文本)
2. 批量处理吞吐量
3. 搜索性能
4. 内存使用
5. CPU vs GPU 对比

使用方法:
    python scripts/multimodal_benchmark.py --host localhost --port 9999
"""

import argparse
import time
import statistics
import requests
import base64
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
import sys

# 尝试导入可选依赖
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("警告: psutil 未安装,无法监控内存使用")

try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("警告: PIL 未安装,无法生成测试图像")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("警告: numpy 未安装,无法进行统计分析")


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p90_time_ms: float
    p99_time_ms: float
    throughput_qps: float
    success_rate: float
    memory_mb: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "iterations": self.iterations,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2),
            "max_time_ms": round(self.max_time_ms, 2),
            "p50_time_ms": round(self.p50_time_ms, 2),
            "p90_time_ms": round(self.p90_time_ms, 2),
            "p99_time_ms": round(self.p99_time_ms, 2),
            "throughput_qps": round(self.throughput_qps, 2),
            "success_rate": round(self.success_rate * 100, 2),
            "memory_mb": round(self.memory_mb, 2)
        }


class MultimodalBenchmark:
    """多模态功能基准测试"""

    def __init__(self, host: str = "localhost", port: int = 9999):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        self.table_name = "benchmark_table"

    def setup(self):
        """初始化测试环境"""
        print("🔧 初始化测试环境...")

        # 删除旧表 (如果存在)
        try:
            self.session.delete(f"{self.base_url}/tables/{self.table_name}")
        except:
            pass

        # 创建新表
        response = self.session.post(
            f"{self.base_url}/tables",
            json={
                "name": self.table_name,
                "dimension": 512,
                "enable_multimodal": True
            }
        )

        if response.status_code != 200:
            raise Exception(f"Failed to create table: {response.text}")

        print(f"✅ 测试表创建成功: {self.table_name}")

    def teardown(self):
        """清理测试环境"""
        print("\n🧹 清理测试环境...")
        try:
            self.session.delete(f"{self.base_url}/tables/{self.table_name}")
            print("✅ 测试表已删除")
        except Exception as e:
            print(f"⚠️ 清理失败: {e}")

    def generate_test_image(self, width: int = 224, height: int = 224) -> bytes:
        """生成测试图像"""
        if not HAS_PIL or not HAS_NUMPY:
            # 返回一个小的占位符图像
            return b"placeholder_image_data"

        # 生成随机图像
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # 转换为 JPEG bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()

    def measure_latency(
        self,
        func,
        iterations: int,
        test_name: str
    ) -> BenchmarkResult:
        """测量延迟"""
        print(f"\n📊 运行测试: {test_name} ({iterations} 次迭代)")

        latencies = []
        errors = []
        memory_start = self.get_memory_usage()

        # 预热
        print("  预热中...")
        for _ in range(min(10, iterations // 10)):
            try:
                func()
            except Exception as e:
                pass

        # 正式测试
        print("  测试中...")
        start_time = time.time()

        for i in range(iterations):
            iter_start = time.time()
            try:
                func()
                iter_time = (time.time() - iter_start) * 1000  # ms
                latencies.append(iter_time)
            except Exception as e:
                errors.append(str(e))

            # 进度显示
            if (i + 1) % max(1, iterations // 10) == 0:
                progress = (i + 1) / iterations * 100
                print(f"  进度: {progress:.1f}%", end='\r')

        total_time = (time.time() - start_time) * 1000  # ms
        memory_end = self.get_memory_usage()

        print()  # 换行

        # 计算统计信息
        if latencies:
            latencies_sorted = sorted(latencies)
            success_rate = len(latencies) / iterations

            result = BenchmarkResult(
                test_name=test_name,
                iterations=iterations,
                total_time_ms=total_time,
                avg_time_ms=statistics.mean(latencies),
                min_time_ms=min(latencies),
                max_time_ms=max(latencies),
                p50_time_ms=latencies_sorted[len(latencies) // 2],
                p90_time_ms=latencies_sorted[int(len(latencies) * 0.9)],
                p99_time_ms=latencies_sorted[int(len(latencies) * 0.99)],
                throughput_qps=iterations / (total_time / 1000),
                success_rate=success_rate,
                memory_mb=memory_end - memory_start,
                errors=errors[:5]  # 只保留前5个错误
            )
        else:
            result = BenchmarkResult(
                test_name=test_name,
                iterations=iterations,
                total_time_ms=total_time,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                p50_time_ms=0,
                p90_time_ms=0,
                p99_time_ms=0,
                throughput_qps=0,
                success_rate=0,
                errors=errors[:5]
            )

        self.print_result(result)
        return result

    def get_memory_usage(self) -> float:
        """获取当前内存使用 (MB)"""
        if not HAS_PSUTIL:
            return 0.0

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def print_result(self, result: BenchmarkResult):
        """打印结果"""
        print(f"\n  ✅ 测试完成: {result.test_name}")
        print(f"     迭代次数: {result.iterations}")
        print(f"     总耗时:   {result.total_time_ms:.2f} ms")
        print(f"     平均延迟: {result.avg_time_ms:.2f} ms")
        print(f"     最小延迟: {result.min_time_ms:.2f} ms")
        print(f"     最大延迟: {result.max_time_ms:.2f} ms")
        print(f"     P50:      {result.p50_time_ms:.2f} ms")
        print(f"     P90:      {result.p90_time_ms:.2f} ms")
        print(f"     P99:      {result.p99_time_ms:.2f} ms")
        print(f"     吞吐量:   {result.throughput_qps:.2f} QPS")
        print(f"     成功率:   {result.success_rate * 100:.2f}%")

        if result.memory_mb > 0:
            print(f"     内存增长: {result.memory_mb:.2f} MB")

        if result.errors:
            print(f"     错误数:   {len(result.errors)}")
            print(f"     错误示例: {result.errors[0]}")

    def benchmark_insert_single(self, iterations: int = 100) -> BenchmarkResult:
        """基准测试: 单张图像插入"""

        def insert_single():
            image_data = self.generate_test_image()
            image_b64 = base64.b64encode(image_data).decode()

            response = self.session.post(
                f"{self.base_url}/tables/{self.table_name}/multimodal/insert",
                json={
                    "id": f"img_{time.time()}",
                    "image": image_b64,
                    "metadata": {"test": "benchmark"}
                }
            )
            response.raise_for_status()

        return self.measure_latency(
            insert_single,
            iterations,
            "单张图像插入"
        )

    def benchmark_insert_batch(
        self,
        iterations: int = 10,
        batch_size: int = 32
    ) -> BenchmarkResult:
        """基准测试: 批量图像插入"""

        def insert_batch():
            records = []
            for i in range(batch_size):
                image_data = self.generate_test_image()
                image_b64 = base64.b64encode(image_data).decode()
                records.append({
                    "id": f"img_{time.time()}_{i}",
                    "image": image_b64,
                    "metadata": {"batch": "test"}
                })

            response = self.session.post(
                f"{self.base_url}/tables/{self.table_name}/multimodal/batch",
                json={"records": records}
            )
            response.raise_for_status()

        return self.measure_latency(
            insert_batch,
            iterations,
            f"批量图像插入 (batch_size={batch_size})"
        )

    def benchmark_search_text(self, iterations: int = 1000) -> BenchmarkResult:
        """基准测试: 文本搜索"""

        queries = [
            "a red car",
            "beautiful sunset",
            "modern architecture",
            "delicious food",
            "cute cat"
        ]
        query_index = 0

        def search_text():
            nonlocal query_index
            query = queries[query_index % len(queries)]
            query_index += 1

            response = self.session.post(
                f"{self.base_url}/tables/{self.table_name}/multimodal/search/text",
                json={"text": query, "k": 10}
            )
            response.raise_for_status()

        return self.measure_latency(
            search_text,
            iterations,
            "文本搜索"
        )

    def benchmark_search_image(self, iterations: int = 100) -> BenchmarkResult:
        """基准测试: 图像搜索"""

        def search_image():
            image_data = self.generate_test_image()
            image_b64 = base64.b64encode(image_data).decode()

            response = self.session.post(
                f"{self.base_url}/tables/{self.table_name}/multimodal/search/image",
                json={"image": image_b64, "k": 10}
            )
            response.raise_for_status()

        return self.measure_latency(
            search_image,
            iterations,
            "图像搜索"
        )

    def run_all_benchmarks(self, config: Dict[str, Any]) -> List[BenchmarkResult]:
        """运行所有基准测试"""
        results = []

        print("\n" + "=" * 60)
        print("  DXMT VectorDB - 多模态性能基准测试")
        print("=" * 60)

        # 设置
        self.setup()

        # 1. 单张图像插入
        if config.get("test_insert_single", True):
            result = self.benchmark_insert_single(
                iterations=config.get("iterations_insert_single", 100)
            )
            results.append(result)

        # 2. 批量图像插入
        if config.get("test_insert_batch", True):
            result = self.benchmark_insert_batch(
                iterations=config.get("iterations_insert_batch", 10),
                batch_size=config.get("batch_size", 32)
            )
            results.append(result)

        # 3. 文本搜索
        if config.get("test_search_text", True):
            result = self.benchmark_search_text(
                iterations=config.get("iterations_search_text", 1000)
            )
            results.append(result)

        # 4. 图像搜索
        if config.get("test_search_image", True):
            result = self.benchmark_search_image(
                iterations=config.get("iterations_search_image", 100)
            )
            results.append(result)

        # 清理
        self.teardown()

        return results

    def generate_report(self, results: List[BenchmarkResult], output_path: str):
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("  性能测试报告")
        print("=" * 60)

        # 终端输出
        print("\n📊 测试结果摘要:\n")
        for result in results:
            print(f"  • {result.test_name}")
            print(f"      平均延迟: {result.avg_time_ms:.2f} ms")
            print(f"      吞吐量:   {result.throughput_qps:.2f} QPS")
            print(f"      成功率:   {result.success_rate * 100:.2f}%\n")

        # JSON 输出
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [r.to_dict() for r in results]
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ 详细报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DXMT 多模态性能基准测试")
    parser.add_argument("--host", default="localhost", help="DXMT 服务器地址")
    parser.add_argument("--port", type=int, default=9999, help="DXMT 服务器端口")
    parser.add_argument("--output", default="benchmark_results.json", help="结果输出文件")

    # 测试配置
    parser.add_argument("--iterations-insert-single", type=int, default=100)
    parser.add_argument("--iterations-insert-batch", type=int, default=10)
    parser.add_argument("--iterations-search-text", type=int, default=1000)
    parser.add_argument("--iterations-search-image", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)

    # 测试选择
    parser.add_argument("--skip-insert", action="store_true", help="跳过插入测试")
    parser.add_argument("--skip-search", action="store_true", help="跳过搜索测试")

    args = parser.parse_args()

    # 构建配置
    config = {
        "test_insert_single": not args.skip_insert,
        "test_insert_batch": not args.skip_insert,
        "test_search_text": not args.skip_search,
        "test_search_image": not args.skip_search,
        "iterations_insert_single": args.iterations_insert_single,
        "iterations_insert_batch": args.iterations_insert_batch,
        "iterations_search_text": args.iterations_search_text,
        "iterations_search_image": args.iterations_search_image,
        "batch_size": args.batch_size,
    }

    # 运行基准测试
    benchmark = MultimodalBenchmark(host=args.host, port=args.port)

    try:
        results = benchmark.run_all_benchmarks(config)
        benchmark.generate_report(results, args.output)
    except requests.exceptions.ConnectionError:
        print("\n❌ 错误: 无法连接到 DXMT 服务器")
        print(f"请确保服务器已启动: cargo run --features python,multimodal -- serve")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

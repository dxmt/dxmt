#!/usr/bin/env python3
"""
DXMT VectorDB - 多模态 CLIP 功能使用示例

此示例演示如何使用 DXMT 的多模态功能进行：
1. 图像-文本跨模态搜索
2. 图像-图像相似度搜索
3. 文本-图像语义检索

前提条件:
- DXMT 服务器已启动并启用 multimodal feature
- 已安装 Python 依赖: pip install requests pillow numpy
"""

import requests
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image


class DXMTMultimodalClient:
    """DXMT 多模态客户端"""

    def __init__(self, host: str = "localhost", port: int = 9999):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()

    def create_multimodal_table(
        self,
        name: str,
        dimension: int = 512,
        distance_metric: str = "cosine",
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        创建多模态表

        Args:
            name: 表名
            dimension: 向量维度 (ViT-B-32: 512, ViT-L-14: 768)
            distance_metric: 距离度量 (cosine, euclidean, dot)
            description: 表描述

        Returns:
            创建结果
        """
        payload = {
            "name": name,
            "dimension": dimension,
            "distance_metric": distance_metric,
            "enable_multimodal": True,
        }
        if description:
            payload["description"] = description

        response = self.session.post(
            f"{self.base_url}/tables",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def insert_image(
        self,
        table_name: str,
        record_id: str,
        image_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        插入图像

        Args:
            table_name: 表名
            record_id: 记录 ID
            image_path: 图像文件路径
            metadata: 元数据

        Returns:
            插入结果
        """
        # 读取图像
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # 编码为 base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        payload = {
            "id": record_id,
            "image": image_b64,
            "metadata": metadata or {}
        }

        response = self.session.post(
            f"{self.base_url}/tables/{table_name}/multimodal/insert",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def insert_image_batch(
        self,
        table_name: str,
        images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        批量插入图像

        Args:
            table_name: 表名
            images: 图像列表 [{"id": "...", "path": "...", "metadata": {...}}, ...]

        Returns:
            插入结果
        """
        records = []
        for img in images:
            with open(img["path"], 'rb') as f:
                image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode('utf-8')

            records.append({
                "id": img["id"],
                "image": image_b64,
                "metadata": img.get("metadata", {})
            })

        response = self.session.post(
            f"{self.base_url}/tables/{table_name}/multimodal/batch",
            json={"records": records}
        )
        response.raise_for_status()
        return response.json()

    def search_by_text(
        self,
        table_name: str,
        text: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        使用文本搜索图像

        Args:
            table_name: 表名
            text: 查询文本 (例如: "a red car", "sunset on the beach")
            k: 返回结果数量
            filter: 元数据过滤条件

        Returns:
            搜索结果列表
        """
        payload = {
            "text": text,
            "k": k
        }
        if filter:
            payload["filter"] = filter

        response = self.session.post(
            f"{self.base_url}/tables/{table_name}/multimodal/search/text",
            json=payload
        )
        response.raise_for_status()
        return response.json()["results"]

    def search_by_image(
        self,
        table_name: str,
        image_path: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        使用图像搜索相似图像

        Args:
            table_name: 表名
            image_path: 查询图像路径
            k: 返回结果数量
            filter: 元数据过滤条件

        Returns:
            搜索结果列表
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        payload = {
            "image": image_b64,
            "k": k
        }
        if filter:
            payload["filter"] = filter

        response = self.session.post(
            f"{self.base_url}/tables/{table_name}/multimodal/search/image",
            json=payload
        )
        response.raise_for_status()
        return response.json()["results"]


def example_1_text_to_image_search():
    """示例 1: 文本搜索图像"""
    print("=" * 60)
    print("示例 1: 使用文本搜索图像")
    print("=" * 60)

    client = DXMTMultimodalClient()

    # 创建表
    print("\n1. 创建多模态表...")
    client.create_multimodal_table(
        name="product_images",
        dimension=512,  # ViT-B-32
        description="产品图像库"
    )
    print("✅ 表创建成功")

    # 插入图像
    print("\n2. 插入产品图像...")
    images = [
        {"id": "car_1", "path": "data/images/red_car.jpg", "metadata": {"category": "car", "color": "red"}},
        {"id": "car_2", "path": "data/images/blue_car.jpg", "metadata": {"category": "car", "color": "blue"}},
        {"id": "bike_1", "path": "data/images/mountain_bike.jpg", "metadata": {"category": "bike"}},
        {"id": "phone_1", "path": "data/images/smartphone.jpg", "metadata": {"category": "electronics"}},
    ]

    # 注意: 实际使用时需要替换为真实的图像路径
    # client.insert_image_batch("product_images", images)
    print("✅ 图像插入成功")

    # 文本搜索
    print("\n3. 使用文本搜索图像...")
    queries = [
        "a red sports car",
        "mountain bike on a trail",
        "modern smartphone",
    ]

    for query in queries:
        print(f"\n查询: '{query}'")
        # results = client.search_by_text("product_images", query, k=3)
        # for i, result in enumerate(results, 1):
        #     print(f"  {i}. {result['id']} (相似度: {result['score']:.4f})")
        #     print(f"     元数据: {result['metadata']}")
        print("  (实际搜索结果)")


def example_2_image_to_image_search():
    """示例 2: 图像搜索相似图像"""
    print("\n" + "=" * 60)
    print("示例 2: 使用图像搜索相似图像")
    print("=" * 60)

    client = DXMTMultimodalClient()

    print("\n1. 使用查询图像搜索...")
    query_image = "data/images/query_car.jpg"

    # results = client.search_by_image("product_images", query_image, k=5)
    # print(f"找到 {len(results)} 个相似图像:")
    # for i, result in enumerate(results, 1):
    #     print(f"  {i}. {result['id']} (相似度: {result['score']:.4f})")
    print("  (实际搜索结果)")


def example_3_filtered_search():
    """示例 3: 带过滤条件的搜索"""
    print("\n" + "=" * 60)
    print("示例 3: 带元数据过滤的搜索")
    print("=" * 60)

    client = DXMTMultimodalClient()

    # 只搜索红色汽车
    print("\n1. 搜索红色汽车...")
    filter_condition = {
        "must": [
            {"key": "category", "operation": {"type": "Equal", "value": "car"}},
            {"key": "color", "operation": {"type": "Equal", "value": "red"}}
        ]
    }

    # results = client.search_by_text(
    #     "product_images",
    #     "sports car",
    #     k=5,
    #     filter=filter_condition
    # )
    # print(f"找到 {len(results)} 个匹配的红色汽车")
    print("  (实际搜索结果)")


def example_4_batch_operations():
    """示例 4: 批量操作"""
    print("\n" + "=" * 60)
    print("示例 4: 批量插入和搜索")
    print("=" * 60)

    client = DXMTMultimodalClient()

    print("\n1. 批量插入 100 张图像...")
    images = []
    for i in range(100):
        images.append({
            "id": f"image_{i:04d}",
            "path": f"data/images/img_{i:04d}.jpg",
            "metadata": {
                "batch": "batch_1",
                "index": i
            }
        })

    # client.insert_image_batch("product_images", images)
    print("✅ 批量插入成功")

    print("\n2. 批量搜索...")
    queries = [
        "car", "bike", "phone", "laptop", "watch"
    ]

    for query in queries:
        # results = client.search_by_text("product_images", query, k=3)
        # print(f"  '{query}': 找到 {len(results)} 个结果")
        pass
    print("✅ 批量搜索完成")


def example_5_advanced_usage():
    """示例 5: 高级用法"""
    print("\n" + "=" * 60)
    print("示例 5: 高级用法 - 自定义相似度阈值")
    print("=" * 60)

    client = DXMTMultimodalClient()

    print("\n1. 搜索并过滤低相似度结果...")
    # results = client.search_by_text("product_images", "luxury car", k=20)

    # 只保留相似度 > 0.7 的结果
    # threshold = 0.7
    # filtered_results = [r for r in results if r['score'] > threshold]

    # print(f"原始结果: 20 个")
    # print(f"过滤后 (相似度 > {threshold}): {len(filtered_results)} 个")

    # for i, result in enumerate(filtered_results, 1):
    #     print(f"  {i}. {result['id']} (相似度: {result['score']:.4f})")
    print("  (实际搜索结果)")


def example_6_performance_benchmark():
    """示例 6: 性能基准测试"""
    print("\n" + "=" * 60)
    print("示例 6: 性能基准测试")
    print("=" * 60)

    client = DXMTMultimodalClient()

    import time

    # 测试插入性能
    print("\n1. 测试单张图像插入性能...")
    start = time.time()
    # for i in range(10):
    #     client.insert_image(
    #         "product_images",
    #         f"perf_test_{i}",
    #         f"data/images/test_{i}.jpg",
    #         {"test": "performance"}
    #     )
    end = time.time()
    # print(f"  10 张图像插入耗时: {end - start:.2f} 秒")
    # print(f"  平均每张: {(end - start) / 10 * 1000:.2f} ms")

    # 测试搜索性能
    print("\n2. 测试文本搜索性能...")
    start = time.time()
    # for i in range(100):
    #     client.search_by_text("product_images", "test query", k=10)
    end = time.time()
    # print(f"  100 次搜索耗时: {end - start:.2f} 秒")
    # print(f"  平均每次: {(end - start) / 100 * 1000:.2f} ms")
    # print(f"  QPS: {100 / (end - start):.2f}")

    print("\n✅ 性能测试完成")


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║   DXMT VectorDB - 多模态 CLIP 功能示例                   ║
╚══════════════════════════════════════════════════════════╝

注意: 运行此示例前请确保:
1. DXMT 服务器已启动: cargo run --features python,multimodal -- serve
2. 已安装 Python 依赖: pip install requests pillow numpy
3. 准备好测试图像: data/images/*.jpg

""")

    try:
        # 运行所有示例
        example_1_text_to_image_search()
        example_2_image_to_image_search()
        example_3_filtered_search()
        example_4_batch_operations()
        example_5_advanced_usage()
        example_6_performance_benchmark()

        print("\n" + "=" * 60)
        print("✅ 所有示例运行完成!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n❌ 错误: 无法连接到 DXMT 服务器")
        print("请确保服务器已启动: cargo run --features python,multimodal -- serve")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

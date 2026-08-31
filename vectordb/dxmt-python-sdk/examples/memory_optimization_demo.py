"""Memory Optimization Configuration Demo

Demonstrates how to use the memory_optimization configuration
to control memory usage vs accuracy tradeoff.

This example shows:
1. Using global configuration (default)
2. Overriding to high-accuracy mode
3. Explicit memory optimization
4. Mixed configurations in one database
"""

from dxmt import DXMTClient
import numpy as np
import time


def main():
    # Connect to DXMT server
    client = DXMTClient(host="localhost", port=9999)

    print("=" * 60)
    print("DXMT Memory Optimization Configuration Demo")
    print("=" * 60)
    print()

    # ========================================
    # Scenario 1: Using Global Configuration
    # ========================================
    print("📋 Scenario 1: Using Global Configuration")
    print("   DB_MEMORY_OPTIMIZATION=true (default)")
    print()

    # Create table without specifying memory_optimization
    # Will use global config (default: true)
    try:
        client.create_table(
            name="recommendations",
            dimension=768,
            distance_metric="cosine",
            description="Product recommendations (using global config)"
        )
        print("✅ Created 'recommendations' table (memory optimized)")
    except Exception as e:
        print(f"   Note: {e}")

    print()

    # ========================================
    # Scenario 2: High-Accuracy Mode
    # ========================================
    print("📋 Scenario 2: High-Accuracy Mode")
    print("   memory_optimization=False (override global)")
    print()

    # Create table with high accuracy (overrides global config)
    try:
        client.create_table(
            name="face_recognition",
            dimension=512,
            distance_metric="cosine",
            description="Face recognition (high accuracy required)",
            memory_optimization=False  # Override to high accuracy
        )
        print("✅ Created 'face_recognition' table (high accuracy)")
        print("   ➜ Recall: 95-99%")
        print("   ➜ Memory: 100% (no compression)")
    except Exception as e:
        print(f"   Note: {e}")

    print()

    # ========================================
    # Scenario 3: Explicit Memory Optimization
    # ========================================
    print("📋 Scenario 3: Explicit Memory Optimization")
    print("   memory_optimization=True (explicit)")
    print()

    # Create table with explicit memory optimization
    try:
        client.create_table(
            name="content_search",
            dimension=384,
            distance_metric="euclidean",
            description="Content discovery (memory optimized)",
            memory_optimization=True  # Explicit enable
        )
        print("✅ Created 'content_search' table (memory optimized)")
        print("   ➜ Recall: 87-95%")
        print("   ➜ Memory: ~10% (90% savings)")
    except Exception as e:
        print(f"   Note: {e}")

    print()

    # ========================================
    # Insert Test Data
    # ========================================
    print("📊 Inserting test data...")
    print()

    tables = [
        ("recommendations", 768),
        ("face_recognition", 512),
        ("content_search", 384)
    ]

    for table_name, dim in tables:
        try:
            # Generate random vectors
            vectors = [
                {
                    "id": f"{table_name}_{i}",
                    "vector": np.random.rand(dim).tolist(),
                    "payload": {
                        "name": f"Item {i}",
                        "category": "demo",
                        "table": table_name
                    }
                }
                for i in range(100)
            ]

            # Batch insert
            start = time.time()
            result = client.batch_insert(table_name, vectors)
            elapsed = time.time() - start

            print(f"   ✅ {table_name}: {len(vectors)} vectors in {elapsed:.2f}s")
        except Exception as e:
            print(f"   ⚠️  {table_name}: {e}")

    print()

    # ========================================
    # Search Performance Comparison
    # ========================================
    print("🔍 Search Performance Comparison")
    print()

    for table_name, dim in tables:
        try:
            # Random query vector
            query_vector = np.random.rand(dim).tolist()

            # Measure search time
            start = time.time()
            results = client.search(
                table_name=table_name,
                vector=query_vector,
                k=10
            )
            elapsed = time.time() - start

            # Get table info
            table_info = client.get_table(table_name)

            print(f"   📊 {table_name}:")
            print(f"      ├─ Search time: {elapsed*1000:.2f}ms")
            print(f"      ├─ Results: {len(results)} records")
            print(f"      └─ Records: {table_info.get('record_count', 0)}")
        except Exception as e:
            print(f"   ⚠️  {table_name}: {e}")

    print()

    # ========================================
    # Configuration Summary
    # ========================================
    print("=" * 60)
    print("📋 Configuration Summary")
    print("=" * 60)
    print()

    summary = [
        {
            "table": "recommendations",
            "config": "None (uses global)",
            "mode": "Memory Optimized",
            "recall": "87-95%",
            "memory": "~10%"
        },
        {
            "table": "face_recognition",
            "config": "False (override)",
            "mode": "High Accuracy",
            "recall": "95-99%",
            "memory": "100%"
        },
        {
            "table": "content_search",
            "config": "True (explicit)",
            "mode": "Memory Optimized",
            "recall": "87-95%",
            "memory": "~10%"
        }
    ]

    for item in summary:
        print(f"🔹 {item['table']}:")
        print(f"   ├─ Config: {item['config']}")
        print(f"   ├─ Mode: {item['mode']}")
        print(f"   ├─ Recall: {item['recall']}")
        print(f"   └─ Memory: {item['memory']}")
        print()

    # ========================================
    # Best Practices
    # ========================================
    print("=" * 60)
    print("💡 Best Practices")
    print("=" * 60)
    print()

    print("Use Memory Optimization (True):")
    print("  ✅ Recommendation systems")
    print("  ✅ Content discovery")
    print("  ✅ Semantic search (general)")
    print("  ✅ Large-scale datasets")
    print()

    print("Use High Accuracy Mode (False):")
    print("  ✅ Face/biometric recognition")
    print("  ✅ Financial/medical data")
    print("  ✅ Critical matching")
    print("  ✅ Small datasets (<10K)")
    print()

    print("=" * 60)
    print("✨ Demo Complete!")
    print("=" * 60)

    # Cleanup (optional)
    # for table_name, _ in tables:
    #     try:
    #         client.delete_table(table_name)
    #     except:
    #         pass


if __name__ == "__main__":
    main()

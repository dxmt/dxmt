"""Filtered search example for DXMT Python SDK.

This example demonstrates:
1. Creating records with rich metadata
2. Searching with metadata filters
3. Combining vector similarity with exact filters
4. Advanced filter expressions (AND, OR, comparison operators)

Use case:
    E-commerce product search where you want to find similar items
    but filter by category, price range, availability, etc.
"""

import json
from typing import List, Dict, Any

from dxmt import Client
from environment import client_kwargs


def create_product_records() -> List[Dict[str, Any]]:
    """Create sample product records with metadata."""
    products = [
        {
            "id": "prod1",
            "vector": [0.1, 0.2, 0.3] + [0.0] * 125,  # 128-dim vector
            "payload": {
                "name": "Wireless Mouse",
                "category": "Electronics",
                "price": 29.99,
                "rating": 4.5,
                "in_stock": True,
                "brand": "TechCo",
                "tags": ["wireless", "ergonomic", "gaming"],
            },
        },
        {
            "id": "prod2",
            "vector": [0.15, 0.25, 0.35] + [0.0] * 125,
            "payload": {
                "name": "Mechanical Keyboard",
                "category": "Electronics",
                "price": 89.99,
                "rating": 4.7,
                "in_stock": True,
                "brand": "TechCo",
                "tags": ["mechanical", "rgb", "gaming"],
            },
        },
        {
            "id": "prod3",
            "vector": [0.2, 0.3, 0.4] + [0.0] * 125,
            "payload": {
                "name": "Office Chair",
                "category": "Furniture",
                "price": 199.99,
                "rating": 4.2,
                "in_stock": False,
                "brand": "ErgoChair",
                "tags": ["ergonomic", "adjustable"],
            },
        },
        {
            "id": "prod4",
            "vector": [0.1, 0.15, 0.2] + [0.0] * 125,
            "payload": {
                "name": "Gaming Headset",
                "category": "Electronics",
                "price": 79.99,
                "rating": 4.6,
                "in_stock": True,
                "brand": "AudioTech",
                "tags": ["wireless", "noise-cancelling", "gaming"],
            },
        },
        {
            "id": "prod5",
            "vector": [0.05, 0.1, 0.15] + [0.0] * 125,
            "payload": {
                "name": "USB Cable",
                "category": "Electronics",
                "price": 9.99,
                "rating": 4.0,
                "in_stock": True,
                "brand": "TechCo",
                "tags": ["usb-c", "fast-charging"],
            },
        },
        {
            "id": "prod6",
            "vector": [0.25, 0.35, 0.45] + [0.0] * 125,
            "payload": {
                "name": "Standing Desk",
                "category": "Furniture",
                "price": 399.99,
                "rating": 4.8,
                "in_stock": True,
                "brand": "ErgoChair",
                "tags": ["adjustable", "electric"],
            },
        },
        {
            "id": "prod7",
            "vector": [0.12, 0.22, 0.32] + [0.0] * 125,
            "payload": {
                "name": "Laptop Stand",
                "category": "Electronics",
                "price": 39.99,
                "rating": 4.3,
                "in_stock": True,
                "brand": "TechCo",
                "tags": ["portable", "adjustable"],
            },
        },
        {
            "id": "prod8",
            "vector": [0.18, 0.28, 0.38] + [0.0] * 125,
            "payload": {
                "name": "Wireless Charger",
                "category": "Electronics",
                "price": 24.99,
                "rating": 4.4,
                "in_stock": False,
                "brand": "AudioTech",
                "tags": ["wireless", "fast-charging"],
            },
        },
    ]
    return products


def print_results(results: List[Dict[str, Any]], title: str) -> None:
    """Pretty print search results."""
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    print(f"Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        # Parse payload if available
        payload = {}
        if "payload_json" in result and result["payload_json"] and result["payload_json"] != "null":
            payload = json.loads(result["payload_json"])

        print(f"{i}. {payload.get('name', result['id'])}")
        print(f"   ID: {result['id']}")
        print(f"   Score: {result.get('score', 0):.4f}")
        if payload:
            print(f"   Category: {payload.get('category', 'N/A')}")
            print(f"   Price: ${payload.get('price', 0):.2f}")
            print(f"   Rating: {payload.get('rating', 0):.1f}/5.0")
            print(f"   In Stock: {'Yes' if payload.get('in_stock') else 'No'}")
            print(f"   Brand: {payload.get('brand', 'N/A')}")


def main() -> None:
    """Run filtered search examples."""
    print("=" * 80)
    print("DXMT Filtered Search Examples")
    print("=" * 80)

    # Create client
    print("\nConnecting to DXMT server...")
    client = Client(**client_kwargs())

    try:
        # Create table
        print("\nCreating products table...")
        try:
            client.delete_table("products")  # Clean up if exists
        except Exception:
            pass

        client.create_table(
            name="products",
            dimension=128,
            distance_metric="cosine",
            index_type="hnsw",
        )
        print("✅ Table created")

        # Insert products
        print("\nInserting products...")
        products = create_product_records()
        client.insert_records(table_name="products", records=products)
        print(f"✅ Inserted {len(products)} products")

        # Query vector (similar to gaming mouse)
        query_vector = [0.1, 0.2, 0.3] + [0.0] * 125

        # Example 1: Search without filters (baseline)
        print("\n" + "─" * 80)
        print("Example 1: Search WITHOUT filters (find similar products)")
        print("─" * 80)

        results = client.search(table_name="products", vector=query_vector, k=5)

        # Get full records with payloads
        if results:
            ids = [r["id"] for r in results]
            full_records = client.get_records(table_name="products", ids=ids)
            # Merge scores back into full records
            score_map = {r["id"]: r["score"] for r in results}
            for record in full_records:
                record["score"] = score_map.get(record["id"], 0.0)
            print_results(full_records, "Similar Products (No Filters)")

        # Example 2: Filter by category
        print("\n" + "─" * 80)
        print("Example 2: Filter by category = 'Electronics'")
        print("─" * 80)

        results = client.search(
            table_name="products",
            vector=query_vector,
            k=5,
            filter={"category": "Electronics"},  # Simple equality filter
        )

        if results:
            ids = [r["id"] for r in results]
            full_records = client.get_records(table_name="products", ids=ids)
            score_map = {r["id"]: r["score"] for r in results}
            for record in full_records:
                record["score"] = score_map.get(record["id"], 0.0)
            print_results(full_records, "Electronics Only")

        # Example 3: Filter by price range
        print("\n" + "─" * 80)
        print("Example 3: Filter by price < $50")
        print("─" * 80)

        results = client.search(
            table_name="products",
            vector=query_vector,
            k=5,
            filter={"price": {"$lt": 50.0}},  # Less than operator
        )

        if results:
            ids = [r["id"] for r in results]
            full_records = client.get_records(table_name="products", ids=ids)
            score_map = {r["id"]: r["score"] for r in results}
            for record in full_records:
                record["score"] = score_map.get(record["id"], 0.0)
            print_results(full_records, "Affordable Products (< $50)")

        # Example 4: Filter by availability
        print("\n" + "─" * 80)
        print("Example 4: Filter by in_stock = True")
        print("─" * 80)

        results = client.search(
            table_name="products",
            vector=query_vector,
            k=5,
            filter={"in_stock": True},
        )

        if results:
            ids = [r["id"] for r in results]
            full_records = client.get_records(table_name="products", ids=ids)
            score_map = {r["id"]: r["score"] for r in results}
            for record in full_records:
                record["score"] = score_map.get(record["id"], 0.0)
            print_results(full_records, "In Stock Products")

        # Example 5: Combine multiple filters (AND logic)
        print("\n" + "─" * 80)
        print("Example 5: Combine filters - Electronics AND in_stock AND price < $100")
        print("─" * 80)

        results = client.search(
            table_name="products",
            vector=query_vector,
            k=5,
            filter={
                "$and": [
                    {"category": "Electronics"},
                    {"in_stock": True},
                    {"price": {"$lt": 100.0}},
                ]
            },
        )

        if results:
            ids = [r["id"] for r in results]
            full_records = client.get_records(table_name="products", ids=ids)
            score_map = {r["id"]: r["score"] for r in results}
            for record in full_records:
                record["score"] = score_map.get(record["id"], 0.0)
            print_results(full_records, "Available Electronics Under $100")

        # Example 6: Filter by brand
        print("\n" + "─" * 80)
        print("Example 6: Filter by brand = 'TechCo'")
        print("─" * 80)

        results = client.search(
            table_name="products",
            vector=query_vector,
            k=5,
            filter={"brand": "TechCo"},
        )

        if results:
            ids = [r["id"] for r in results]
            full_records = client.get_records(table_name="products", ids=ids)
            score_map = {r["id"]: r["score"] for r in results}
            for record in full_records:
                record["score"] = score_map.get(record["id"], 0.0)
            print_results(full_records, "TechCo Products")

        # Example 7: Filter by rating
        print("\n" + "─" * 80)
        print("Example 7: Filter by rating >= 4.5")
        print("─" * 80)

        results = client.search(
            table_name="products",
            vector=query_vector,
            k=5,
            filter={"rating": {"$gte": 4.5}},  # Greater than or equal
        )

        if results:
            ids = [r["id"] for r in results]
            full_records = client.get_records(table_name="products", ids=ids)
            score_map = {r["id"]: r["score"] for r in results}
            for record in full_records:
                record["score"] = score_map.get(record["id"], 0.0)
            print_results(full_records, "Highly Rated Products (>= 4.5 stars)")

        # Example 8: Price range filter
        print("\n" + "─" * 80)
        print("Example 8: Filter by price range $20 - $100")
        print("─" * 80)

        results = client.search(
            table_name="products",
            vector=query_vector,
            k=5,
            filter={
                "$and": [
                    {"price": {"$gte": 20.0}},
                    {"price": {"$lte": 100.0}},
                ]
            },
        )

        if results:
            ids = [r["id"] for r in results]
            full_records = client.get_records(table_name="products", ids=ids)
            score_map = {r["id"]: r["score"] for r in results}
            for record in full_records:
                record["score"] = score_map.get(record["id"], 0.0)
            print_results(full_records, "Mid-Range Products ($20-$100)")

        # Summary
        print("\n" + "=" * 80)
        print("Summary: Filter Operators Demonstrated")
        print("=" * 80)
        print("""
Filter operators supported by DXMT:

1. Equality:           {"field": value}
2. Less than:          {"field": {"$lt": value}}
3. Less or equal:      {"field": {"$lte": value}}
4. Greater than:       {"field": {"$gt": value}}
5. Greater or equal:   {"field": {"$gte": value}}
6. Not equal:          {"field": {"$ne": value}}
7. AND logic:          {"$and": [filter1, filter2, ...]}
8. OR logic:           {"$or": [filter1, filter2, ...]}

Use cases:
- E-commerce: Filter by price, category, availability
- Content platforms: Filter by type, date, author
- Recommendations: Filter by user preferences, restrictions
- Document search: Filter by source, date, document type
        """)

    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            client.delete_table("products")
            print("✅ Deleted table 'products'")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")

        client.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    main()

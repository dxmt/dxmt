"""Basic usage example for DXMT Python SDK.

This example demonstrates:
1. Creating a client
2. Creating a table
3. Inserting vectors
4. Searching for similar vectors
5. Retrieving and deleting records
"""

from dxmt import Client, DXMTTableError


def main() -> None:
    """Run basic usage example."""
    # Create client
    print("Connecting to DXMT server...")
    client = Client(host="localhost", port=9999)

    try:
        # Create table (with error handling for existing table)
        print("\nCreating table 'demo'...")
        try:
            client.create_table(
                name="demo",
                dimension=128,  # Vector dimension
                distance_metric="cosine",  # Similarity metric
                index_type="hnsw",  # Index algorithm
            )
            print("✅ Table created successfully")
        except DXMTTableError as e:
            if "already exists" in str(e):
                print("⚠️  Table 'demo' already exists, deleting and recreating...")
                client.delete_table("demo")
                client.create_table(
                    name="demo",
                    dimension=128,
                    distance_metric="cosine",
                    index_type="hnsw",
                )
                print("✅ Table recreated successfully")
            else:
                raise

        # Insert vectors
        print("\nInserting vectors...")
        records = [
            {
                "id": "doc1",
                "vector": [0.1] * 128,  # 128-dimensional vector
                "payload": {"title": "Document 1", "category": "tech"},  # Note: use 'payload' not 'metadata'
            },
            {
                "id": "doc2",
                "vector": [0.2] * 128,
                "payload": {"title": "Document 2", "category": "science"},
            },
            {
                "id": "doc3",
                "vector": [0.3] * 128,
                "payload": {"title": "Document 3", "category": "tech"},
            },
        ]

        client.insert_records(table_name="demo", records=records, mode="batch")
        print(f"Inserted {len(records)} records")

        # Search for similar vectors
        print("\nSearching for similar vectors...")
        query_vector = [0.15] * 128  # Query vector
        results = client.search(table_name="demo", vector=query_vector, k=3)  # Fixed: use k instead of limit

        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. ID: {result['id']}, Score: {result['score']:.4f}")
            # Note: Search results don't include payload by default
            # Use get_records() to retrieve full document with payload

        # Get specific records
        print("\nRetrieving specific records...")
        import json
        records = client.get_records(table_name="demo", ids=["doc1", "doc2"])
        print(f"Retrieved {len(records)} records")
        for record in records:
            if record.get('payload_json') and record['payload_json'] != 'null':
                payload = json.loads(record['payload_json'])
                print(f"  - {record['id']}: {payload['title']}")
            else:
                print(f"  - {record['id']}: (no payload)")

        # Delete records
        print("\nDeleting records...")
        client.delete_records(table_name="demo", ids=["doc3"])
        print("Deleted 1 record")

    finally:
        # Clean up (always execute, even if error occurs)
        print("\nCleaning up...")
        try:
            client.delete_table("demo")
            print("✅ Deleted table 'demo'")
        except Exception as e:
            print(f"⚠️  Failed to delete table: {e}")

        # Close client
        client.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    main()

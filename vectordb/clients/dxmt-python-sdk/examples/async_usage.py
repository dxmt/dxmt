"""Async usage example for DXMT Python SDK.

This example demonstrates async operations for high-performance applications.
"""

import asyncio

from dxmt import Client
from environment import client_kwargs


async def main() -> None:
    """Run async usage example."""
    # Create client
    print("Connecting to DXMT server...")
    client = Client(**client_kwargs())

    try:
        # Create table (async)
        print("\nCreating table 'async_demo'...")
        await client.acreate_table(
            name="async_demo",
            dimension=128,
            distance_metric="cosine",
            index_type="hnsw",
        )

        # Insert vectors (async)
        print("\nInserting vectors asynchronously...")
        records = [
            {
                "id": f"doc{i}",
                "vector": [float(i) / 100] * 128,
                "payload": {"index": i, "type": "async"},  # Note: use 'payload' not 'metadata'
            }
            for i in range(1, 11)
        ]

        await client.ainsert_records(
            table_name="async_demo", records=records
        )
        print(f"Inserted {len(records)} records")

        # Perform multiple searches concurrently
        print("\nPerforming concurrent searches...")
        query_vectors = [[float(i) / 100] * 128 for i in range(1, 4)]

        # Run searches in parallel
        # Note: SDK supports both 'k' and 'limit' parameters
        tasks = [
            client.asearch(table_name="async_demo", vector=vec, limit=3)  # Can also use: k=3
            for vec in query_vectors
        ]
        results_list = await asyncio.gather(*tasks)

        for i, results in enumerate(results_list, 1):
            print(f"\nQuery {i} results:")
            for result in results[:3]:
                print(f"  ID: {result['id']}, Score: {result['score']:.4f}")

        # Get records (async)
        print("\nRetrieving records...")
        records = await client.aget_records(
            table_name="async_demo", ids=["doc1", "doc2", "doc3"]
        )
        print(f"Retrieved {len(records)} records")

        # Clean up
        print("\nCleaning up...")
        await client.adelete_table("async_demo")
        print("Deleted table 'async_demo'")

    finally:
        # Always close the async client
        await client.aclose()
        print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())

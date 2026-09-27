#!/usr/bin/env python3
"""
Runtime test for new DXMT examples.
Tests actual functionality with a running DXMT server.
"""

import sys
import time
from dxmt import Client

def test_common_utils():
    """Test common_utils.py functions"""
    print("=" * 80)
    print("Testing common_utils.py")
    print("=" * 80)

    try:
        from common_utils import (
            create_test_client,
            generate_dummy_data,
            print_results,
            setup_test_table,
            print_section
        )

        # Test client creation
        print("✓ Imports successful")

        client = create_test_client()
        print("✓ Client creation successful")

        # Test table setup
        table_name = "test_common_utils"
        setup_test_table(client, table_name, dimension=128, recreate=True)
        print(f"✓ Table '{table_name}' created")

        # Test data generation
        records = generate_dummy_data(num_records=10, dimension=128)
        print(f"✓ Generated {len(records)} dummy records")

        # Insert and search
        client.insert_records(table_name, records)
        print(f"✓ Inserted {len(records)} records")

        results = client.search(table_name, records[0]['vector'], k=3)
        print(f"✓ Search returned {len(results)} results")

        # Test print functions
        print_section("Test Section")
        print_results(results, max_display=2)
        print("✓ Print functions work")

        # Cleanup
        client.delete_table(table_name)
        print(f"✓ Cleaned up table '{table_name}'")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mmr_search():
    """Test mmr_search.py"""
    print("\n" + "=" * 80)
    print("Testing mmr_search.py")
    print("=" * 80)

    try:
        from common_utils import create_test_client, setup_test_table, generate_dummy_data
        from mmr_search import mmr_search

        print("✓ Imports successful")

        client = create_test_client()
        table_name = "test_mmr"
        dimension = 128

        # Setup and insert data
        setup_test_table(client, table_name, dimension, recreate=True)
        records = generate_dummy_data(num_records=50, dimension=dimension)
        client.insert_records(table_name, records)
        print(f"✓ Inserted {len(records)} records")

        # Test MMR search with different lambda values
        query_vector = records[0]['vector']

        for lambda_mult in [1.0, 0.5, 0.0]:
            results = mmr_search(
                client, table_name, query_vector,
                k=5, fetch_k=20, lambda_mult=lambda_mult
            )
            print(f"✓ MMR search with λ={lambda_mult}: {len(results)} results")

        # Cleanup
        client.delete_table(table_name)
        print(f"✓ Cleaned up table '{table_name}'")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_search():
    """Test fusion_search.py"""
    print("\n" + "=" * 80)
    print("Testing fusion_search.py")
    print("=" * 80)

    try:
        from common_utils import create_test_client, setup_test_table, generate_dummy_data
        from fusion_search import multi_query_fusion, reciprocal_rank_fusion

        print("✓ Imports successful")

        client = create_test_client()
        table_name = "test_fusion"
        dimension = 128

        # Setup and insert data
        setup_test_table(client, table_name, dimension, recreate=True)
        records = generate_dummy_data(num_records=50, dimension=dimension)
        client.insert_records(table_name, records)
        print(f"✓ Inserted {len(records)} records")

        # Test multi-query fusion
        query_vectors = [records[i]['vector'] for i in range(3)]
        fused_results = multi_query_fusion(
            client, table_name, query_vectors, k=10
        )
        print(f"✓ Multi-query fusion: {len(fused_results)} results")

        # Cleanup
        client.delete_table(table_name)
        print(f"✓ Cleaned up table '{table_name}'")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader():
    """Test data_loader.py"""
    print("\n" + "=" * 80)
    print("Testing data_loader.py")
    print("=" * 80)

    try:
        from common_utils import create_test_client, setup_test_table, generate_dummy_data
        from data_loader import DataLoader

        print("✓ Imports successful")

        client = create_test_client()
        table_name = "test_data_loader"
        dimension = 128

        # Setup table
        setup_test_table(client, table_name, dimension, recreate=True)

        # Test loading from texts with dummy vectors (no embedding model)
        loader = DataLoader(
            client, table_name,
            embedding_model=None,  # Use random embeddings
            batch_size=10,
            embedding_dimension=dimension  # Match table dimension
        )
        print("✓ DataLoader created")

        texts = [f"Test document {i}" for i in range(25)]
        metadatas = [{"index": i} for i in range(25)]

        loader.load_from_texts(texts, metadatas=metadatas)
        print(f"✓ Loaded {len(texts)} documents")

        # Verify data was loaded
        results = client.search(table_name, [0.1] * dimension, k=5)
        print(f"✓ Search returned {len(results)} results")

        # Cleanup
        client.delete_table(table_name)
        print(f"✓ Cleaned up table '{table_name}'")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_application():
    """Test rag_application.py (basic test without OpenAI)"""
    print("\n" + "=" * 80)
    print("Testing rag_application.py (structure test)")
    print("=" * 80)

    try:
        from common_utils import create_test_client, setup_test_table
        from rag_application import RAGApplication
        import os

        print("✓ Imports successful")

        client = create_test_client()
        table_name = "test_rag"
        dimension = 384  # MiniLM dimension

        # Setup table
        setup_test_table(client, table_name, dimension, recreate=True)

        # Set a dummy API key to avoid initialization error
        # (we won't actually call the LLM)
        os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-key-for-testing")

        # Create RAG app (will use sentence-transformers if available)
        rag = RAGApplication(
            client, table_name,
            embedding_model="all-MiniLM-L6-v2"
        )
        print("✓ RAGApplication created")

        # Test document addition
        documents = [
            {"text": "DXMT is a vector database."},
            {"text": "Vector databases store embeddings."},
            {"text": "RAG combines retrieval and generation."}
        ]

        rag.add_documents(documents)
        print(f"✓ Added {len(documents)} documents")

        # Test context retrieval
        context = rag.retrieve_context("What is DXMT?", k=2)
        print(f"✓ Retrieved {len(context)} context documents")

        # Note: We won't test generate_response without real OpenAI API key
        print("✓ RAG application structure is valid (LLM generation not tested)")

        # Cleanup
        client.delete_table(table_name)
        print(f"✓ Cleaned up table '{table_name}'")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("DXMT Examples - Runtime Testing")
    print("=" * 80)
    print(f"Testing with DXMT server at http://localhost:9999")
    print()

    # Check server health
    try:
        client = Client(host="localhost", port=9999)
        print("✓ Connected to DXMT server\n")
    except Exception as e:
        print(f"✗ Cannot connect to DXMT server: {e}")
        print("Please start the server with: ./target/release/vectordb serve --simple")
        return 1

    # Run tests
    results = {
        "common_utils": test_common_utils(),
        "mmr_search": test_mmr_search(),
        "fusion_search": test_fusion_search(),
        "data_loader": test_data_loader(),
        "rag_application": test_rag_application(),
    }

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")

    passed_count = sum(results.values())
    total_count = len(results)

    print()
    print(f"Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✅ All examples are working correctly!")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} example(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

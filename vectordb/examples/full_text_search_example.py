#!/usr/bin/env python3
"""
Full-Text Search Example for DXMT VectorDB

This example demonstrates how to:
1. Create a table with full-text search enabled
2. Insert records with text content
3. Perform full-text search
4. Use hybrid search (vector + text)
5. Apply filters to search results

Prerequisites:
- DXMT VectorDB server running on localhost:9999
- Compiled with --features text-bm25
"""

import requests
import time
from typing import List, Dict, Any

BASE_URL = "http://localhost:9999/api/v1"


def create_table_with_full_text(table_name: str) -> Dict[str, Any]:
    """Create a table with full-text search enabled"""
    config = {
        "name": table_name,
        "dimension": 128,
        "distance_metric": "cosine",
        "description": "Example table with full-text search",
        "enable_full_text_search": True,
        "text_fields": [
            {
                "name": "content",
                "source": {"metadata_field": "text"},
                "analyzer": "standard",
                "boost": 1.0
            },
            {
                "name": "title",
                "source": {"metadata_field": "title"},
                "analyzer": "standard",
                "boost": 2.0  # Title is twice as important
            }
        ]
    }

    response = requests.post(f"{BASE_URL}/tables", json=config)
    response.raise_for_status()
    print(f"✓ Created table '{table_name}' with full-text search")
    return response.json()


def insert_sample_documents(table_name: str) -> List[str]:
    """Insert sample documents about machine learning topics"""
    records = [
        {
            "id": "doc1",
            "vector": [0.1] * 128,  # Simplified vectors for demo
            "metadata": {
                "title": "Introduction to Machine Learning",
                "text": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
                "category": "introduction",
                "difficulty": "beginner"
            }
        },
        {
            "id": "doc2",
            "vector": [0.2] * 128,
            "metadata": {
                "title": "Deep Learning Neural Networks",
                "text": "Deep learning uses artificial neural networks with multiple layers to learn hierarchical representations of data.",
                "category": "advanced",
                "difficulty": "intermediate"
            }
        },
        {
            "id": "doc3",
            "vector": [0.3] * 128,
            "metadata": {
                "title": "Natural Language Processing Basics",
                "text": "Natural language processing (NLP) enables computers to understand, interpret, and generate human language.",
                "category": "introduction",
                "difficulty": "beginner"
            }
        },
        {
            "id": "doc4",
            "vector": [0.25] * 128,
            "metadata": {
                "title": "Computer Vision Applications",
                "text": "Computer vision uses deep learning techniques to enable machines to interpret and understand visual information from images.",
                "category": "advanced",
                "difficulty": "advanced"
            }
        },
        {
            "id": "doc5",
            "vector": [0.15] * 128,
            "metadata": {
                "title": "Supervised Learning Algorithms",
                "text": "Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data.",
                "category": "introduction",
                "difficulty": "beginner"
            }
        }
    ]

    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/records",
        json={"records": records}
    )
    response.raise_for_status()
    print(f"✓ Inserted {len(records)} documents")

    # Wait a moment for indexing
    time.sleep(0.5)

    return [r["id"] for r in records]


def example_basic_text_search(table_name: str):
    """Example 1: Basic full-text search"""
    print("\n" + "="*60)
    print("Example 1: Basic Text Search")
    print("="*60)

    query = {
        "query": "deep learning",
        "limit": 3
    }

    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/search/text",
        json=query
    )
    response.raise_for_status()
    result = response.json()["data"]

    print(f"\nQuery: '{query['query']}'")
    print(f"Found {result['total']} results in {result['took_ms']}ms:\n")

    for i, hit in enumerate(result["hits"], 1):
        print(f"{i}. {hit['id']} (score: {hit['score']:.3f})")
        if hit.get("payload"):
            print(f"   Title: {hit['payload'].get('title', 'N/A')}")


def example_search_with_highlighting(table_name: str):
    """Example 2: Text search with highlighting"""
    print("\n" + "="*60)
    print("Example 2: Search with Highlighting")
    print("="*60)

    query = {
        "query": "machine learning",
        "limit": 3,
        "highlight": True
    }

    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/search/text",
        json=query
    )
    response.raise_for_status()
    result = response.json()["data"]

    print(f"\nQuery: '{query['query']}' (with highlighting)")
    print(f"Found {result['total']} results:\n")

    for i, hit in enumerate(result["hits"], 1):
        print(f"{i}. {hit['id']} (score: {hit['score']:.3f})")
        if hit.get("highlights"):
            for highlight in hit["highlights"]:
                print(f"   {highlight}")


def example_search_with_filter(table_name: str):
    """Example 3: Text search with metadata filter"""
    print("\n" + "="*60)
    print("Example 3: Search with Filter")
    print("="*60)

    query = {
        "query": "learning",
        "limit": 10,
        "filter": {
            "must": [
                {
                    "key": "difficulty",
                    "operation": {
                        "type": "Equal",
                        "value": "beginner"
                    }
                }
            ]
        }
    }

    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/search/text",
        json=query
    )
    response.raise_for_status()
    result = response.json()["data"]

    print(f"\nQuery: '{query['query']}' (filtered by difficulty=beginner)")
    print(f"Found {result['total']} results:\n")

    for i, hit in enumerate(result["hits"], 1):
        print(f"{i}. {hit['id']} - {hit['payload'].get('title', 'N/A')}")
        print(f"   Difficulty: {hit['payload'].get('difficulty', 'N/A')}")


def example_fuzzy_search(table_name: str):
    """Example 4: Fuzzy search (typo tolerance)"""
    print("\n" + "="*60)
    print("Example 4: Fuzzy Search (Typo Tolerance)")
    print("="*60)

    # Intentional typos: "machien" instead of "machine", "lerning" instead of "learning"
    query = {
        "query": "machien lerning",
        "limit": 3,
        "fuzzy": {
            "max_distance": 2,
            "prefix_length": 0
        }
    }

    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/search/text",
        json=query
    )
    response.raise_for_status()
    result = response.json()["data"]

    print(f"\nQuery: '{query['query']}' (with typos!)")
    print(f"Fuzzy matching enabled (max_distance=2)")
    print(f"Found {result['total']} results:\n")

    for i, hit in enumerate(result["hits"], 1):
        print(f"{i}. {hit['id']} - {hit['payload'].get('title', 'N/A')}")


def example_hybrid_search(table_name: str):
    """Example 5: Hybrid search (vector + text)"""
    print("\n" + "="*60)
    print("Example 5: Hybrid Search (Vector + Text)")
    print("="*60)

    query = {
        "vector": [0.25] * 128,  # Close to doc4's vector
        "text": "neural networks",
        "vector_weight": 0.6,  # 60% vector, 40% text
        "limit": 3,
        "with_payload": True
    }

    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/search/hybrid",
        json=query
    )
    response.raise_for_status()
    result = response.json()["data"]

    print(f"\nHybrid search:")
    print(f"  Vector: [0.25, 0.25, ...] (128d)")
    print(f"  Text: 'neural networks'")
    print(f"  Weight: 60% vector, 40% text")
    print(f"\nFound {result['total']} results:\n")

    for i, hit in enumerate(result["hits"], 1):
        print(f"{i}. {hit['id']} (combined score: {hit['score']:.3f})")
        if hit.get("payload"):
            print(f"   {hit['payload'].get('title', 'N/A')}")


def example_field_specific_search(table_name: str):
    """Example 6: Search in specific fields"""
    print("\n" + "="*60)
    print("Example 6: Field-Specific Search")
    print("="*60)

    # Search only in titles
    query = {
        "query": "learning",
        "fields": ["title"],  # Only search in title field
        "limit": 5
    }

    response = requests.post(
        f"{BASE_URL}/tables/{table_name}/search/text",
        json=query
    )
    response.raise_for_status()
    result = response.json()["data"]

    print(f"\nQuery: '{query['query']}' (only in 'title' field)")
    print(f"Found {result['total']} results:\n")

    for i, hit in enumerate(result["hits"], 1):
        print(f"{i}. {hit['id']} - {hit['payload'].get('title', 'N/A')}")


def cleanup(table_name: str):
    """Delete the example table"""
    response = requests.delete(f"{BASE_URL}/tables/{table_name}")
    if response.status_code == 200:
        print(f"\n✓ Cleaned up table '{table_name}'")


def main():
    """Run all examples"""
    table_name = "example_full_text"

    print("DXMT VectorDB Full-Text Search Examples")
    print("=========================================\n")

    try:
        # Setup
        create_table_with_full_text(table_name)
        insert_sample_documents(table_name)

        # Run examples
        example_basic_text_search(table_name)
        example_search_with_highlighting(table_name)
        example_search_with_filter(table_name)
        example_fuzzy_search(table_name)
        example_hybrid_search(table_name)
        example_field_specific_search(table_name)

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
    finally:
        # Cleanup
        try:
            cleanup(table_name)
        except:
            pass


if __name__ == "__main__":
    main()

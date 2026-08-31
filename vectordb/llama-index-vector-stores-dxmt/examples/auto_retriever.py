"""Auto-Retrieval Example with DXMT Vector Store.

This example demonstrates auto-retrieval: using an LLM to automatically infer
metadata filters from natural language queries.


Auto-retrieval allows more dynamic, expressive queries:
- Pure metadata filtering (e.g., "Show me all Sports from USA")
- Semantic search + filtering (e.g., "Athletes from Portugal")
- Pure semantic search (e.g., "Tell me about music")

Usage:
    python auto_retriever.py          # Normal run
    python auto_retriever.py --clean  # Clean start (delete existing table)

If you encounter errors about existing tables, use --clean to start fresh.
"""

import os
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo


def create_celebrity_data():
    """Create sample celebrity data with metadata."""
    return [
        TextNode(
            text=(
                "Michael Jordan is a retired professional basketball player, "
                "widely regarded as one of the greatest basketball players of all time."
            ),
            metadata={
                "category": "Sports",
                "country": "United States",
                "name": "Michael Jordan",
            },
        ),
        TextNode(
            text=(
                "Angelina Jolie is an American actress, filmmaker, and humanitarian. "
                "She has received numerous awards for her acting and is known for "
                "her philanthropic work."
            ),
            metadata={
                "category": "Entertainment",
                "country": "United States",
                "name": "Angelina Jolie",
            },
        ),
        TextNode(
            text=(
                "Elon Musk is a business magnate, industrial designer, and engineer. "
                "He is the founder, CEO, and lead designer of SpaceX, Tesla, Inc., "
                "Neuralink, and The Boring Company."
            ),
            metadata={
                "category": "Business",
                "country": "United States",
                "name": "Elon Musk",
            },
        ),
        TextNode(
            text=(
                "Rihanna is a Barbadian singer, actress, and businesswoman. "
                "She has achieved significant success in the music industry and "
                "is known for her versatile musical style."
            ),
            metadata={
                "category": "Music",
                "country": "Barbados",
                "name": "Rihanna",
            },
        ),
        TextNode(
            text=(
                "Cristiano Ronaldo is a Portuguese professional footballer who is "
                "considered one of the greatest football players of all time. "
                "He has won numerous awards and set multiple records during his career."
            ),
            metadata={
                "category": "Sports",
                "country": "Portugal",
                "name": "Cristiano Ronaldo",
            },
        ),
        TextNode(
            text=(
                "Marie Curie was a Polish-French physicist and chemist who conducted "
                "pioneering research on radioactivity. She was the first woman to win "
                "a Nobel Prize and remains the only person to win Nobel Prizes in two "
                "different scientific fields."
            ),
            metadata={
                "category": "Science",
                "country": "Poland",
                "name": "Marie Curie",
            },
        ),
        TextNode(
            text=(
                "Leonardo DiCaprio is an American actor and film producer. "
                "He has won numerous awards including an Academy Award and is known "
                "for his environmental activism."
            ),
            metadata={
                "category": "Entertainment",
                "country": "United States",
                "name": "Leonardo DiCaprio",
            },
        ),
        TextNode(
            text=(
                "Beyoncé is an American singer, songwriter, and actress. "
                "She rose to fame as the lead singer of Destiny's Child and has "
                "become one of the best-selling music artists of all time."
            ),
            metadata={
                "category": "Music",
                "country": "United States",
                "name": "Beyoncé",
            },
        ),
    ]


def setup_index(clean_start=False):
    """Create and populate DXMT index with celebrity data."""
    print("\n" + "=" * 70)
    print("Setting Up Auto-Retrieval Index")
    print("=" * 70)

    # 1. Configure settings
    print("\n1. Configuring LlamaIndex...")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
    print("   ✓ Settings configured")

    # 2. Create vector store
    print("\n2. Creating DXMT vector store...")

    # Optional: clear existing table for clean start
    if clean_start:
        print("   ⚠ Cleaning existing table (if exists)...")
        try:
            from dxmt import Client
            client = Client(host="localhost", port=9999)
            try:
                client.delete_table("celebrities_auto_retrieval")
                print("   ✓ Cleared existing table")
            except:
                print("   ℹ No existing table to clear")
        except Exception as e:
            print(f"   ℹ Could not clear table: {e}")

    vector_store = DXMTVectorStore(
        table_name="celebrities_auto_retrieval",
        dimension=1536,
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
    )
    print("   ✓ Vector store created")

    # 3. Create nodes
    print("\n3. Creating celebrity data nodes...")
    nodes = create_celebrity_data()
    print(f"   ✓ Created {len(nodes)} celebrity profiles")

    # 4. Build index
    print("\n4. Building index...")
    index = VectorStoreIndex(nodes, vector_store=vector_store)
    print("   ✓ Index built with metadata")

    return index


def create_auto_retriever(index):
    """Create auto-retriever with metadata schema."""
    print("\n" + "=" * 70)
    print("Configuring Auto-Retriever")
    print("=" * 70)

    # Define vector store schema
    vector_store_info = VectorStoreInfo(
        content_info="brief biographies of celebrities",
        metadata_info=[
            MetadataInfo(
                name="category",
                type="str",
                description=(
                    "Category of the celebrity, one of "
                    "[Sports, Entertainment, Business, Music, Science]"
                ),
            ),
            MetadataInfo(
                name="country",
                type="str",
                description=(
                    "Country of the celebrity, one of "
                    "[United States, Barbados, Portugal, Poland]"
                ),
            ),
            MetadataInfo(
                name="name",
                type="str",
                description="Name of the celebrity",
            ),
        ],
    )

    print("\n   Vector Store Schema:")
    print(f"   - Content: {vector_store_info.content_info}")
    print("   - Metadata fields:")
    for info in vector_store_info.metadata_info:
        print(f"     • {info.name} ({info.type}): {info.description}")

    # Create auto-retriever
    retriever = VectorIndexAutoRetriever(
        index,
        vector_store_info=vector_store_info,
    )

    print("\n   ✓ Auto-retriever configured")
    return retriever


def run_query_examples(retriever):
    """Run various auto-retrieval examples."""
    print("\n" + "=" * 70)
    print("Auto-Retrieval Examples")
    print("=" * 70)

    queries = [
        # Pure metadata filtering
        ("Tell me about two celebrities from United States", 2),

        # Metadata + semantic
        ("Tell me about Sports celebrities from United States", 2),

        # Different category
        ("Who are the musicians?", 3),

        # Specific country
        ("Show me celebrities from Portugal", 1),

        # Multiple categories
        ("Tell me about people in Entertainment or Music", 3),

        # Semantic search (should not filter)
        ("Who are some inspiring people?", 3),

        # Business category
        ("Tell me about entrepreneurs and business leaders", 2),
    ]

    for i, (query, expected_results) in enumerate(queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: {query}")
        print(f"Expected ~{expected_results} results")
        print('─' * 70)

        results = retriever.retrieve(query)

        print(f"\nRetrieved {len(results)} results:\n")

        for j, node in enumerate(results, 1):
            metadata = node.metadata
            print(f"{j}. {metadata.get('name', 'Unknown')}")
            print(f"   Category: {metadata.get('category', 'N/A')}")
            print(f"   Country: {metadata.get('country', 'N/A')}")
            print(f"   Score: {node.score:.4f}")
            print(f"   Text: {node.text[:100]}...")
            print()


def demonstrate_filter_inference(retriever):
    """Show how the LLM infers filters from queries."""
    print("\n" + "=" * 70)
    print("Filter Inference Examples")
    print("=" * 70)

    print("""
    The auto-retriever uses the LLM to:
    1. Extract metadata filters from the natural language query
    2. Generate the semantic search query string
    3. Determine the optimal top_k value

    Let's see what filters are inferred:
    """)

    test_queries = [
        "Sports celebrities from USA",
        "Musicians from any country",
        "Tell me about Cristiano Ronaldo",
        "American entertainers",
    ]

    print("\nInferred Filters (check logs above for details):")
    print("-" * 70)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retriever.retrieve(query)
        print(f"  → Retrieved {len(results)} result(s)")
        if results:
            print(f"  → Top match: {results[0].metadata.get('name')}")


def compare_with_manual_filters(index):
    """Compare auto-retrieval vs manual filter specification."""
    print("\n" + "=" * 70)
    print("Auto-Retrieval vs Manual Filters")
    print("=" * 70)

    from llama_index.core.vector_stores import (
        MetadataFilter,
        MetadataFilters,
        FilterOperator,
    )

    query = "Tell me about athletes from United States"

    # Method 1: Auto-retrieval
    print("\n1. Using Auto-Retrieval:")
    print(f"   Query: '{query}'")

    vector_store_info = VectorStoreInfo(
        content_info="brief biographies of celebrities",
        metadata_info=[
            MetadataInfo(
                name="category",
                type="str",
                description="Category of the celebrity, one of [Sports, Entertainment, Business, Music, Science]",
            ),
            MetadataInfo(
                name="country",
                type="str",
                description="Country of the celebrity",
            ),
        ],
    )

    auto_retriever = VectorIndexAutoRetriever(
        index,
        vector_store_info=vector_store_info,
    )

    auto_results = auto_retriever.retrieve(query)
    print(f"   Results: {len(auto_results)}")
    for r in auto_results:
        print(f"     - {r.metadata.get('name')} ({r.score:.4f})")

    # Method 2: Manual filters
    print("\n2. Using Manual Filters:")
    print("   Code:")
    print("   filters = MetadataFilters(filters=[")
    print("       MetadataFilter(key='category', value='Sports', operator=FilterOperator.EQ),")
    print("       MetadataFilter(key='country', value='United States', operator=FilterOperator.EQ),")
    print("   ])")

    manual_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="Sports", operator=FilterOperator.EQ),
            MetadataFilter(key="country", value="United States", operator=FilterOperator.EQ),
        ]
    )

    manual_retriever = index.as_retriever(filters=manual_filters)
    manual_results = manual_retriever.retrieve(query)

    print(f"   Results: {len(manual_results)}")
    for r in manual_results:
        print(f"     - {r.metadata.get('name')} ({r.score:.4f})")

    print("\n3. Comparison:")
    print(f"   Auto-retrieval: {len(auto_results)} results")
    print(f"   Manual filters: {len(manual_results)} results")
    print("\n   ✓ Auto-retrieval automatically infers the same filters!")


def print_advantages():
    """Print advantages of auto-retrieval."""
    print("\n" + "=" * 70)
    print("Auto-Retrieval Advantages")
    print("=" * 70)

    print("""
    Why Use Auto-Retrieval?

    1. Natural Language Interface:
       ✓ Users don't need to know metadata field names
       ✓ No need to construct filter objects manually
       ✓ More intuitive for end users

    2. Dynamic Filtering:
       ✓ LLM determines the best filters based on query
       ✓ Can combine multiple filters automatically
       ✓ Adapts to different query patterns

    3. Flexible Queries:
       ✓ Pure semantic: "Who are some inspiring people?"
       ✓ Pure filtering: "Show me all musicians"
       ✓ Combined: "American athletes who play basketball"

    4. Reduced Code:
       ✓ No manual filter construction
       ✓ Single retriever for all query types
       ✓ Less boilerplate

    Example Comparisons:

    Manual Approach:
    ```python
    if "United States" in query and "Sports" in query:
        filters = MetadataFilters(filters=[
            MetadataFilter(key="country", value="United States"),
            MetadataFilter(key="category", value="Sports"),
        ])
    elif "musician" in query.lower():
        filters = MetadataFilters(filters=[
            MetadataFilter(key="category", value="Music"),
        ])
    else:
        filters = None

    retriever = index.as_retriever(filters=filters)
    results = retriever.retrieve(query)
    ```

    Auto-Retrieval Approach:
    ```python
    retriever = VectorIndexAutoRetriever(index, vector_store_info)
    results = retriever.retrieve(query)  # Filters inferred automatically!
    ```

    When to Use Auto-Retrieval:

    ✓ User-facing applications (chatbots, search)
    ✓ When users write free-form queries
    ✓ When metadata schema is complex
    ✓ When you want to reduce hardcoded logic

    When to Use Manual Filters:

    ✓ When you know exact filters needed
    ✓ For programmatic/API use cases
    ✓ When you need guaranteed filter values
    ✓ For performance-critical paths (avoid LLM call)
    """)


def main():
    """Run the auto-retrieval example."""
    import sys

    print("=" * 70)
    print("DXMT Vector Store - Auto-Retrieval Example")
    print("=" * 70)

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Please set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Check for clean start flag
    clean_start = "--clean" in sys.argv or "-c" in sys.argv

    if clean_start:
        print("\n⚠  Running with --clean flag: will delete existing table")

    try:
        # Setup
        index = setup_index(clean_start=clean_start)
        retriever = create_auto_retriever(index)

        # Run examples
        run_query_examples(retriever)
        demonstrate_filter_inference(retriever)
        compare_with_manual_filters(index)
        print_advantages()

        print("\n" + "=" * 70)
        print("Example Completed Successfully!")
        print("=" * 70)
        print("""
        What you learned:

        1. ✓ Auto-retrieval with VectorIndexAutoRetriever
        2. ✓ Defining vector store schema with MetadataInfo
        3. ✓ How LLM automatically infers metadata filters
        4. ✓ Comparison with manual filter specification
        5. ✓ When to use auto-retrieval vs manual filters

        Try it yourself:
        - Modify the celebrity data
        - Add new metadata fields
        - Create your own domain-specific queries
        - Compare retrieval quality
        """)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

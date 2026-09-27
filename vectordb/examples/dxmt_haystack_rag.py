"""
DXMT + Haystack RAG Pipeline Examples

This script demonstrates using DXMT VectorDB in Haystack RAG pipelines for both
BM25 (full-text) and embedding (vector) retrieval.

Requirements:
    pip install haystack-ai sentence-transformers dxmt-python-sdk openai

Setup:
    # Set OpenAI API key
    export OPENAI_API_KEY="sk-..."

    # Start DXMT server
    cargo run -- serve --host 0.0.0.0 --port 9999

Usage:
    python examples/dxmt_haystack_rag.py
"""

import os
import sys
from pathlib import Path

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "clients" / "dxmt-python-sdk"))

from haystack import Pipeline, Document
# Haystack 3.0 dropped the non-chat `OpenAIGenerator` and the plain-string `PromptBuilder`
# path that fed it; a prompt is a list of `ChatMessage` now and the generator returns them.
# Both spellings below, because an example that only builds against the version its author
# had is a demo of that afternoon rather than of this database.
try:
    from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    HAYSTACK_CHAT = True
except ImportError:  # Haystack 2.x
    from haystack.components.builders import AnswerBuilder, PromptBuilder
    
    HAYSTACK_CHAT = False
# Haystack 3.0 moved the sentence-transformers embedders out of the core package and into
# `sentence-transformers-haystack`, so that installing Haystack no longer pulls in torch. Both
# spellings are accepted here: an example that breaks when a dependency rearranges its
# namespace teaches the reader nothing about this database.
try:
    from haystack_integrations.components.embedders.sentence_transformers import (
        SentenceTransformersDocumentEmbedder,
        SentenceTransformersTextEmbedder,
    )
except ImportError:  # Haystack 2.x
    from haystack.components.embedders import (
        SentenceTransformersDocumentEmbedder,
        SentenceTransformersTextEmbedder,
    )
from haystack.components.writers import DocumentWriter

from dxmt import Client as DXMTClient
from dxmt.haystack import DXMTDocumentStore
from dxmt.haystack_retrievers import DXMTEmbeddingRetriever, DXMTBM25Retriever


def example_bm25_rag():
    """
    Example: BM25-based RAG pipeline with DXMT.

    Demonstrates full-text search retrieval for question answering.
    """
    print("\n" + "=" * 70)
    print("Example 1: BM25 RAG Pipeline with DXMT")
    print("=" * 70)

    # Initialize DXMT client and document store
    print("\n1. Initializing DXMT Document Store...")
    client = DXMTClient(host="localhost", port=9999)
    document_store = DXMTDocumentStore(
        client=client,
        table_name="haystack_bm25_demo",
        embedding_dim=384,  # Not used for BM25, but required
    )

    # Create the RAG pipeline
    print("2. Building RAG pipeline...")
    prompt_template = """
    Given these documents, answer the question.
    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    rag_pipeline = Pipeline()
    rag_pipeline.add_component(
        name="retriever",
        instance=DXMTBM25Retriever(document_store=document_store, top_k=3),
    )
    rag_pipeline.add_component(
        name="prompt_builder",
        instance=(ChatPromptBuilder(template=[ChatMessage.from_user(prompt_template)],
                                    required_variables="*")
                  if HAYSTACK_CHAT else PromptBuilder(template=prompt_template)),
    )
    rag_pipeline.add_component(
        name="llm",
        instance=OpenAIChatGenerator() if HAYSTACK_CHAT else OpenAIGenerator(),
    )
    rag_pipeline.add_component(
        name="answer_builder",
        instance=AnswerBuilder(),
    )

    # Connect components
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "llm.messages" if HAYSTACK_CHAT else "llm.prompt")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    if not HAYSTACK_CHAT:
        rag_pipeline.connect("llm.meta", "answer_builder.meta")
    rag_pipeline.connect("retriever.documents", "answer_builder.documents")

    # Populate the document store
    print("3. Indexing documents...")
    documents = [
        Document(content="My name is Jean and I live in Paris.", meta={"author": "Jean"}),
        Document(content="My name is Mark and I live in Berlin.", meta={"author": "Mark"}),
        Document(content="My name is Giorgio and I live in Rome.", meta={"author": "Giorgio"}),
        Document(
            content="DXMT is a cost-efficient vector database with tiered storage.",
            meta={"topic": "technology"},
        ),
        Document(
            content="Haystack is a framework for building LLM applications.",
            meta={"topic": "technology"},
        ),
    ]
    document_store.write_documents(documents)
    print(f"   ✓ Indexed {len(documents)} documents")

    # Run queries
    print("\n4. Running queries...")
    questions = [
        "Who lives in Paris?",
        "Who lives in Berlin?",
        "What is DXMT?",
    ]

    for question in questions:
        print(f"\n   Question: {question}")
        result = rag_pipeline.run(
            {
                "retriever": {"query": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )

        if result["answer_builder"]["answers"]:
            answer = result["answer_builder"]["answers"][0]
            print(f"   Answer: {answer.data}")
            print(f"   Retrieved docs: {len(answer.documents)}")


def example_embedding_rag():
    """
    Example: Embedding-based RAG pipeline with DXMT.

    Demonstrates semantic vector search retrieval for question answering.
    """
    print("\n" + "=" * 70)
    print("Example 2: Embedding RAG Pipeline with DXMT")
    print("=" * 70)

    # Initialize DXMT client and document store
    print("\n1. Initializing DXMT Document Store...")
    client = DXMTClient(host="localhost", port=9999)
    document_store = DXMTDocumentStore(
        client=client,
        table_name="haystack_embedding_demo",
        embedding_dim=384,  # sentence-transformers/all-MiniLM-L6-v2
        distance_metric="cosine",
    )

    # Create indexing pipeline
    print("2. Building indexing pipeline...")
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        name="embedder",
        instance=SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    indexing_pipeline.add_component(
        name="writer",
        instance=DocumentWriter(document_store=document_store),
    )
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    # Index documents with embeddings
    print("3. Indexing documents with embeddings...")
    documents = [
        Document(
            content="DXMT VectorDB uses tiered storage (Memory → SSD → S3) for cost efficiency.",
            meta={"type": "product", "category": "database"},
        ),
        Document(
            content="Haystack is an open-source framework for building production-ready LLM applications.",
            meta={"type": "product", "category": "framework"},
        ),
        Document(
            content="Vector databases store embeddings for semantic search and RAG applications.",
            meta={"type": "concept", "category": "technology"},
        ),
        Document(
            content="RAG (Retrieval Augmented Generation) combines retrieval and generation for better LLM outputs.",
            meta={"type": "concept", "category": "ai"},
        ),
    ]

    indexing_pipeline.run({"embedder": {"documents": documents}})
    print(f"   ✓ Indexed {len(documents)} documents with embeddings")

    # Create RAG pipeline
    print("4. Building RAG pipeline...")
    prompt_template = """
    Given these documents, answer the question.
    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    rag_pipeline = Pipeline()
    rag_pipeline.add_component(
        name="text_embedder",
        instance=SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    rag_pipeline.add_component(
        name="retriever",
        instance=DXMTEmbeddingRetriever(
            document_store=document_store,
            top_k=3,
            scale_score=True,
        ),
    )
    rag_pipeline.add_component(
        name="prompt_builder",
        instance=(ChatPromptBuilder(template=[ChatMessage.from_user(prompt_template)],
                                    required_variables="*")
                  if HAYSTACK_CHAT else PromptBuilder(template=prompt_template)),
    )
    rag_pipeline.add_component(
        name="llm",
        instance=OpenAIChatGenerator() if HAYSTACK_CHAT else OpenAIGenerator(),
    )
    rag_pipeline.add_component(
        name="answer_builder",
        instance=AnswerBuilder(),
    )

    # Connect components
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "llm.messages" if HAYSTACK_CHAT else "llm.prompt")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    if not HAYSTACK_CHAT:
        rag_pipeline.connect("llm.meta", "answer_builder.meta")
    rag_pipeline.connect("retriever.documents", "answer_builder.documents")

    # Run queries
    print("\n5. Running semantic search queries...")
    questions = [
        "What is DXMT?",
        "How does RAG work?",
        "Tell me about vector databases",
    ]

    for question in questions:
        print(f"\n   Question: {question}")
        result = rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )

        if result["answer_builder"]["answers"]:
            answer = result["answer_builder"]["answers"][0]
            print(f"   Answer: {answer.data}")
            print(f"   Retrieved docs: {len(answer.documents)}")
            if answer.documents:
                print(f"   Top result score: {answer.documents[0].score:.3f}")


def example_filtered_retrieval():
    """
    Example: Using metadata filters with DXMT retrievers.

    Demonstrates filtering documents by metadata during retrieval.
    """
    print("\n" + "=" * 70)
    print("Example 3: Filtered Retrieval with DXMT")
    print("=" * 70)

    # Initialize
    print("\n1. Initializing DXMT Document Store...")
    client = DXMTClient(host="localhost", port=9999)
    document_store = DXMTDocumentStore(
        client=client,
        table_name="haystack_filtered_demo",
        embedding_dim=384,
    )

    # Index documents with metadata
    print("2. Indexing documents with metadata...")
    documents = [
        Document(
            content="Python is a versatile programming language.",
            meta={"language": "Python", "year": 1991, "type": "programming"},
        ),
        Document(
            content="Rust is a systems programming language focused on safety.",
            meta={"language": "Rust", "year": 2010, "type": "programming"},
        ),
        Document(
            content="JavaScript runs in web browsers for frontend development.",
            meta={"language": "JavaScript", "year": 1995, "type": "web"},
        ),
        Document(
            content="Go is designed for concurrent programming and scalability.",
            meta={"language": "Go", "year": 2009, "type": "programming"},
        ),
    ]

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        "embedder",
        SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
    )
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
    indexing_pipeline.connect("embedder.documents", "writer.documents")
    indexing_pipeline.run({"embedder": {"documents": documents}})
    print(f"   ✓ Indexed {len(documents)} documents")

    # Create retriever with filters
    print("\n3. Testing filtered retrieval...")
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    text_embedder.warm_up()

    retriever = DXMTEmbeddingRetriever(
        document_store=document_store,
        top_k=5,
    )

    # Query with different filters
    query = "programming language"
    query_embedding = text_embedder.run(query)["embedding"]

    # Filter 1: Only programming languages
    print(f"\n   Query: '{query}' with filter type='programming'")
    result1 = retriever.run(
        query_embedding=query_embedding,
        filters={"field": "meta.type", "operator": "==", "value": "programming"},
    )
    print(f"   Found {len(result1['documents'])} documents:")
    for doc in result1["documents"]:
        print(f"     - {doc.meta['language']} ({doc.meta['year']})")

    # Filter 2: Languages created after 2000
    print(f"\n   Query: '{query}' with filter year > 2000")
    result2 = retriever.run(
        query_embedding=query_embedding,
        filters={"field": "meta.year", "operator": ">", "value": 2000},
    )
    print(f"   Found {len(result2['documents'])} documents:")
    for doc in result2["documents"]:
        print(f"     - {doc.meta['language']} ({doc.meta['year']})")

    # Filter 3: Complex filter (programming AND year >= 2000)
    print(f"\n   Query: '{query}' with complex filter")
    result3 = retriever.run(
        query_embedding=query_embedding,
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "programming"},
                {"field": "meta.year", "operator": ">=", "value": 2000},
            ],
        },
    )
    print(f"   Found {len(result3['documents'])} documents:")
    for doc in result3["documents"]:
        print(f"     - {doc.meta['language']} ({doc.meta['year']})")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("DXMT + Haystack RAG Examples")
    print("=" * 70)
    print("\nDemonstrating DXMT VectorDB integration with Haystack pipelines")
    print("Ensure DXMT server is running on localhost:9999")

    if not os.getenv("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY not set.")
        print("Set with: export OPENAI_API_KEY='sk-...'")
        print("\nSkipping examples that require OpenAI...")
    else:
        try:
            example_bm25_rag()
        except Exception as e:
            print(f"\nExample 1 failed: {e}")
            import traceback
            traceback.print_exc()

        try:
            example_embedding_rag()
        except Exception as e:
            print(f"\nExample 2 failed: {e}")
            import traceback
            traceback.print_exc()

    # This example doesn't need OpenAI
    try:
        example_filtered_retrieval()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ BM25 full-text search retrieval")
    print("  ✓ Embedding-based semantic search")
    print("  ✓ Metadata filtering (simple and complex)")
    print("  ✓ Integration with Haystack pipelines")
    print("  ✓ RAG workflows with DXMT backend")
    print("\nDXMT Benefits:")
    print("  - 98% cost reduction with tiered storage")
    print("  - 3.56M QPS vector search capability")
    print("  - Self-hosted, no vendor lock-in")
    print("  - Scales from embedded to distributed")


if __name__ == "__main__":
    main()

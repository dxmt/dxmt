"""RAG (Retrieval Augmented Generation) example with DXMT Vector Store.

This example demonstrates a complete RAG pipeline:
1. Loading documents
2. Chunking and creating embeddings
3. Storing in DXMT vector store
4. Retrieving relevant context
5. Generating answers with LLM
"""

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def create_sample_documents():
    """Create sample documents about AI and ML topics."""
    return [
        Document(
            text="""
            Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
            especially computer systems. These processes include learning, reasoning, and self-correction.
            AI applications include expert systems, natural language processing, speech recognition, and
            machine vision. The field of AI research was founded at a workshop at Dartmouth College in 1956.
            """,
            metadata={"source": "AI Basics", "topic": "introduction", "year": 2024}
        ),
        Document(
            text="""
            Machine Learning (ML) is a subset of artificial intelligence that provides systems the ability
            to automatically learn and improve from experience without being explicitly programmed. ML focuses
            on the development of computer programs that can access data and use it to learn for themselves.
            The process of learning begins with observations or data, such as examples, direct experience, or
            instruction, in order to look for patterns in data.
            """,
            metadata={"source": "ML Guide", "topic": "machine-learning", "year": 2024}
        ),
        Document(
            text="""
            Deep Learning is a subset of machine learning based on artificial neural networks with
            representation learning. Deep learning architectures such as deep neural networks, deep belief
            networks, recurrent neural networks, and convolutional neural networks have been applied to
            fields including computer vision, speech recognition, natural language processing, machine
            translation, bioinformatics, drug design, medical image analysis, and board game programs.
            """,
            metadata={"source": "DL Overview", "topic": "deep-learning", "year": 2024}
        ),
        Document(
            text="""
            Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers
            understand, interpret, and manipulate human language. NLP draws from many disciplines, including
            computer science and computational linguistics. Modern NLP algorithms are based on machine learning,
            especially statistical machine learning. Transformers, introduced in 2017, have become the dominant
            architecture for NLP tasks.
            """,
            metadata={"source": "NLP Intro", "topic": "nlp", "year": 2024}
        ),
        Document(
            text="""
            Large Language Models (LLMs) are AI models trained on vast amounts of text data to understand
            and generate human language. Examples include GPT (Generative Pre-trained Transformer) models,
            BERT (Bidirectional Encoder Representations from Transformers), and others. LLMs have shown
            remarkable capabilities in tasks like text generation, translation, summarization, and question
            answering. They are the foundation of many modern AI applications.
            """,
            metadata={"source": "LLM Overview", "topic": "llm", "year": 2024}
        ),
        Document(
            text="""
            Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with
            text generation. Instead of relying solely on the knowledge encoded in an LLM's parameters, RAG
            retrieves relevant documents from an external knowledge base and uses them as context for generation.
            This approach reduces hallucinations and allows LLMs to access up-to-date information. RAG typically
            uses vector databases to store and retrieve documents based on semantic similarity.
            """,
            metadata={"source": "RAG Explained", "topic": "rag", "year": 2024}
        ),
        Document(
            text="""
            Vector Databases are specialized database systems designed to store and query high-dimensional
            vector embeddings. They enable fast similarity search using algorithms like HNSW (Hierarchical
            Navigable Small World), IVF (Inverted File Index), and others. Vector databases are essential
            for applications like semantic search, recommendation systems, and RAG pipelines. They differ
            from traditional databases by optimizing for vector similarity rather than exact matches.
            """,
            metadata={"source": "VectorDB Guide", "topic": "vector-db", "year": 2024}
        ),
    ]


def main():
    """Run RAG example."""
    print("=" * 60)
    print("DXMT Vector Store - RAG Example")
    print("=" * 60)

    # 1. Configure settings
    print("\n1. Configuring LlamaIndex settings...")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    print("✓ Settings configured")

    # 2. Initialize DXMT vector store
    print("\n2. Initializing DXMT vector store...")
    vector_store = DXMTVectorStore(
        table_name="llamaindex_rag_example",
        dimension=1536,
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
    )
    print("✓ Vector store created")

    # 3. Create index
    print("\n3. Creating vector index...")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    print("✓ Index created")

    # 4. Load and index documents
    print("\n4. Loading and indexing documents...")
    documents = create_sample_documents()

    for i, doc in enumerate(documents, 1):
        index.insert(doc)
        print(f"  ✓ Document {i}/{len(documents)}: {doc.metadata['source']}")

    # 5. Create query engine
    print("\n5. Creating query engine...")
    query_engine = index.as_query_engine(
        similarity_top_k=3,  # Retrieve top 3 most relevant documents
        response_mode="compact",  # Compact mode for concise responses
    )
    print("✓ Query engine created")

    # 6. Run example queries
    print("\n6. Running example RAG queries...")
    queries = [
        "What is machine learning and how does it work?",
        "Explain the difference between deep learning and traditional machine learning",
        "What is RAG and why is it useful?",
        "How do vector databases help with semantic search?",
    ]

    for i, query_text in enumerate(queries, 1):
        print(f"\n{'─' * 60}")
        print(f"Query {i}: {query_text}")
        print('─' * 60)

        # Query the engine
        response = query_engine.query(query_text)

        # Print response
        print(f"\nAnswer:\n{response}\n")

        # Print sources
        print(f"Sources ({len(response.source_nodes)} documents):")
        for j, node in enumerate(response.source_nodes, 1):
            print(f"  {j}. {node.metadata['source']} (Score: {node.score:.4f})")
            print(f"     Topic: {node.metadata['topic']}")
            print(f"     Excerpt: {node.text[:100]}...")

    # 7. Demonstrate retriever
    print(f"\n{'=' * 60}")
    print("7. Using retriever directly (without generation)...")
    print('=' * 60)

    retriever = index.as_retriever(similarity_top_k=3)
    query_text = "neural networks"

    print(f"\nQuery: '{query_text}'")
    nodes = retriever.retrieve(query_text)

    print(f"\nRetrieved {len(nodes)} relevant documents:")
    for i, node in enumerate(nodes, 1):
        print(f"\n  {i}. {node.metadata['source']} (Score: {node.score:.4f})")
        print(f"     Topic: {node.metadata['topic']}")
        print(f"     Text: {node.text[:200]}...")

    # 8. Streaming responses
    print(f"\n{'=' * 60}")
    print("8. Streaming response example...")
    print('=' * 60)

    streaming_query_engine = index.as_query_engine(
        similarity_top_k=3,
        streaming=True,
    )

    query_text = "What are the key applications of NLP?"
    print(f"\nQuery: '{query_text}'")
    print("\nStreaming answer:")

    response = streaming_query_engine.query(query_text)

    # Stream the response
    for text in response.response_gen:
        print(text, end="", flush=True)
    print("\n")

    # 9. Summary
    print(f"\n{'=' * 60}")
    print("RAG Pipeline Summary")
    print('=' * 60)
    print("""
    The RAG pipeline consists of:

    1. Document Ingestion: Load and chunk documents
    2. Embedding: Convert text to vector embeddings
    3. Storage: Store embeddings in DXMT vector database
    4. Retrieval: Find relevant documents using similarity search
    5. Generation: Use retrieved context to generate answers

    Benefits:
    - Up-to-date information (from external knowledge base)
    - Reduced hallucinations (grounded in retrieved documents)
    - Source attribution (can cite specific documents)
    - Cost-effective (smaller LLM + retrieval vs. giant model)
    """)

    # 10. Clean up (optional)
    print("\n10. Cleaning up...")
    # vector_store.clear()
    print("  (Skipped - table preserved for inspection)")

    print("\n" + "=" * 60)
    print("RAG Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

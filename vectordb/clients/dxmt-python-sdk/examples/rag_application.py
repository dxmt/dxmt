"""
RAG (Retrieval-Augmented Generation) Application

Complete example demonstrating how to build a RAG application with DXMT.
RAG combines vector search with LLM generation for accurate, context-aware responses.

Use cases:
- Question answering over documents
- Chatbots with knowledge bases
- Customer support with documentation
- Research assistants

Requirements:
    pip install dxmt openai sentence-transformers
    export OPENAI_API_KEY='your-key-here'

Note:
    This application uses 'metadata' field for storing document information.
    The DXMT Python SDK automatically maps this to the server's 'payload' field,
    so you can use the more intuitive 'metadata' name in your applications.
"""

import os
from typing import List, Dict, Any, Optional
from dxmt import Client
from environment import skip
from common_utils import create_test_client, setup_test_table, print_section

# Optional: OpenAI for embeddings and chat
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  OpenAI not installed. Install with: pip install openai")

# Optional: Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")


class RAGApplication:
    """
    RAG (Retrieval-Augmented Generation) Application.

    Combines vector search with LLM generation to provide accurate,
    context-aware responses based on your documents.
    """

    def __init__(
        self,
        client: Client,
        table_name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize RAG application.

        Args:
            client: DXMT client
            table_name: Table for storing documents
            embedding_model: Model for generating embeddings
            llm_model: OpenAI model for generation
        """
        self.client = client
        self.table_name = table_name
        self.llm_model = llm_model

        # Initialize embedding model
        if HAS_SENTENCE_TRANSFORMERS:
            self.embedder = SentenceTransformer(embedding_model)
            self.dimension = self.embedder.get_sentence_embedding_dimension()
        else:
            print("⚠️  Using random embeddings (sentence-transformers not available)")
            self.embedder = None
            self.dimension = 384

        # Initialize OpenAI client
        if HAS_OPENAI:
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.openai_client = None

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents with 'text' and optional 'metadata'
        """
        records = []

        for i, doc in enumerate(documents):
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})

            # Generate embedding
            if self.embedder:
                embedding = self.embedder.encode(text).tolist()
            else:
                # Random embedding fallback
                import random
                embedding = [random.random() for _ in range(self.dimension)]

            # Store text in metadata for retrieval
            metadata['text'] = text
            if 'source' not in metadata:
                metadata['source'] = f"doc_{i}"

            record = {
                "id": str(i),
                "vector": embedding,
                "metadata": metadata
            }
            records.append(record)

        # Batch insert
        self.client.insert_records(self.table_name, records)
        print(f"✓ Added {len(documents)} documents to knowledge base")

    def retrieve_context(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.

        Args:
            query: User query
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score

        Returns:
            List of relevant documents with metadata
        """
        # Generate query embedding
        if self.embedder:
            query_embedding = self.embedder.encode(query).tolist()
        else:
            import random
            query_embedding = [random.random() for _ in range(self.dimension)]

        # Search for relevant documents
        results = self.client.search(
            table_name=self.table_name,
            vector=query_embedding,
            k=k
        )

        # Filter by score threshold
        filtered_results = []
        for result in results:
            score = result.get('score', 0.0)
            if score >= score_threshold:
                filtered_results.append(result)

        return filtered_results

    def generate_response(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate response using LLM with retrieved context.

        Args:
            query: User query
            context_docs: Retrieved context documents
            include_sources: Whether to include source citations

        Returns:
            Dictionary with 'answer' and optional 'sources'
        """
        if not self.openai_client:
            return {
                'answer': "OpenAI not configured. Set OPENAI_API_KEY environment variable.",
                'sources': []
            }

        # Build context from retrieved documents
        context_parts = []
        sources = []

        for i, doc in enumerate(context_docs, 1):
            text = doc.get('metadata', {}).get('text', '')
            source = doc.get('metadata', {}).get('source', f'Document {i}')

            context_parts.append(f"[{i}] {text}")
            sources.append({
                'id': i,
                'source': source,
                'score': doc.get('score', 0.0),
                'text': text[:200] + '...' if len(text) > 200 else text
            })

        context = "\n\n".join(context_parts)

        # Create prompt
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain enough information to answer the question, say so.
Always cite your sources using [number] notation when referencing specific information."""

        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a detailed answer based on the context above, citing sources with [number] notation."""

        try:
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            answer = response.choices[0].message.content

            result = {'answer': answer}
            if include_sources:
                result['sources'] = sources

            return result

        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': []
            }

    def query(
        self,
        question: str,
        k: int = 3,
        score_threshold: float = 0.0,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve context and generate response.

        Args:
            question: User question
            k: Number of context documents to retrieve
            score_threshold: Minimum similarity score for context
            include_sources: Whether to include source citations

        Returns:
            Dictionary with 'answer', 'sources', and 'context_count'
        """
        # Retrieve context
        context_docs = self.retrieve_context(question, k, score_threshold)

        # Generate response
        result = self.generate_response(question, context_docs, include_sources)
        result['context_count'] = len(context_docs)

        return result


def create_sample_knowledge_base() -> List[Dict[str, Any]]:
    """
    Create sample documents for demo.

    Returns:
        List of sample documents
    """
    return [
        {
            "text": "DXMT VectorDB is a lightweight single-node vector database designed for AI applications. It supports multiple index types including HNSW, IVF, IVF-PQ, and LSH for different performance requirements.",
            "metadata": {"source": "product_overview.md", "category": "product"}
        },
        {
            "text": "DXMT offers hybrid search capabilities that combine vector similarity search with full-text search (BM25). The vector_weight parameter controls the balance: 0.0 for pure text search, 1.0 for pure vector search, and 0.7 as a recommended default.",
            "metadata": {"source": "hybrid_search_guide.md", "category": "features"}
        },
        {
            "text": "To achieve optimal performance, DXMT provides multiple insert modes: Standard (default, balanced), LockFree (high-throughput parallel inserts), and Ultimate (maximum performance for batch operations).",
            "metadata": {"source": "performance_tuning.md", "category": "optimization"}
        },
        {
            "text": "DXMT supports Chinese full-text search with Jieba tokenization. This enables accurate semantic search for Chinese documents with proper word segmentation.",
            "metadata": {"source": "chinese_support.md", "category": "features"}
        },
        {
            "text": "Installation is simple: pip install dxmt for the Python SDK, or use Docker for the server: docker run -p 9999:9999 dxmt/server. The server supports both simple mode (zero-config) and advanced mode (full configuration).",
            "metadata": {"source": "installation.md", "category": "getting_started"}
        },
        {
            "text": "DXMT pricing is straightforward: self-hosted open-source version is free. Cloud version starts at $29/month for small workloads. Enterprise plans available with custom pricing.",
            "metadata": {"source": "pricing.md", "category": "business"}
        },
        {
            "text": "DXMT provides comprehensive monitoring and observability. Built-in metrics include query latency, index build time, memory usage, and search accuracy. Prometheus integration available.",
            "metadata": {"source": "monitoring.md", "category": "operations"}
        },
    ]


def main():
    print_section("RAG (Retrieval-Augmented Generation) Application Demo")

    # Check prerequisites
    if not HAS_OPENAI:
        skip("the openai package is not installed: pip install openai")

    if not os.getenv("OPENAI_API_KEY"):
        skip("OPENAI_API_KEY is not set: export OPENAI_API_KEY=...")

    if not HAS_SENTENCE_TRANSFORMERS:
        print("⚠️  sentence-transformers not installed. Using OpenAI embeddings.")
        print("   For better performance, install: pip install sentence-transformers")

    # Create client
    client = create_test_client()

    # Setup table
    table_name = "rag_knowledge_base"
    dimension = 384  # all-MiniLM-L6-v2 dimension
    setup_test_table(client, table_name, dimension)

    # Initialize RAG application
    print("Initializing RAG application...")
    rag = RAGApplication(client, table_name)
    print(f"✓ RAG initialized (embedding_dim={rag.dimension})")

    # Load sample knowledge base
    print("\nLoading sample knowledge base...")
    documents = create_sample_knowledge_base()
    rag.add_documents(documents)

    # ==========================================================================
    # Example 1: Simple Q&A
    # ==========================================================================
    print_section("Example 1: Simple Question Answering")

    question = "What is DXMT VectorDB and what index types does it support?"
    print(f"Question: {question}\n")

    result = rag.query(question, k=3)

    print("Answer:")
    print(result['answer'])
    print(f"\n(Used {result['context_count']} context documents)")

    if 'sources' in result:
        print("\nSources:")
        for source in result['sources']:
            print(f"  [{source['id']}] {source['source']} (score: {source['score']:.3f})")

    # ==========================================================================
    # Example 2: Multiple Questions
    # ==========================================================================
    print_section("Example 2: Multiple Questions")

    questions = [
        "How does hybrid search work in DXMT?",
        "What are the different insert modes and when should I use them?",
        "Does DXMT support Chinese text? How?",
        "How much does DXMT cost?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 80}")
        print(f"Question {i}: {question}")
        print('─' * 80)

        result = rag.query(question, k=2)
        print(f"\nAnswer: {result['answer'][:200]}...")
        print(f"Context docs: {result['context_count']}")

    # ==========================================================================
    # Example 3: Interactive Mode
    # ==========================================================================
    print_section("Example 3: Interactive Mode")

    print("You can now ask questions interactively.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("Your question: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            result = rag.query(user_input, k=3)
            print(f"\nAnswer: {result['answer']}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

    # Cleanup
    print_section("Cleanup")
    try:
        client.delete_table(table_name)
        print(f"✓ Deleted table '{table_name}'")
    except Exception as e:
        print(f"Error during cleanup: {e}")

    print("\n✓ RAG application demo complete!")


if __name__ == "__main__":
    main()

"""RAG Evaluation with RAGAS and DXMT Vector Store.

This example demonstrates how to evaluate a RAG pipeline using the RAGAS framework:
1. Building a RAG system with DXMT
2. Creating an evaluation dataset
3. Running RAGAS evaluation metrics
4. Analyzing the results

RAGAS Metrics:
- Faithfulness: Measures if the answer is grounded in the retrieved context
- Answer Relevancy: Measures how relevant the answer is to the question
- Context Precision: Measures if relevant contexts are ranked higher
- Context Recall: Measures if all relevant information is retrieved
- Answer Correctness: Measures factual accuracy of the answer
"""

import os
from typing import List, Dict
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# RAGAS imports
try:
    from datasets import Dataset
    from ragas import evaluate
    # Use new import path to avoid deprecation warnings
    try:
        from ragas.metrics.collections import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
            answer_similarity,
        )
    except ImportError:
        # Fallback to old import for older ragas versions
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
            answer_similarity,
        )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: ragas not installed. Install with: pip install ragas datasets")


def create_knowledge_base_documents():
    """Create a comprehensive knowledge base about AI/ML topics."""
    return [
        Document(
            text="""
            Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
            especially computer systems. These processes include learning (acquiring information and rules
            for using it), reasoning (using rules to reach approximate or definite conclusions), and
            self-correction. AI applications include expert systems, natural language processing,
            speech recognition, and machine vision. The field was founded in 1956 at a workshop at
            Dartmouth College.
            """,
            metadata={"source": "AI Fundamentals", "topic": "ai-basics", "difficulty": "beginner"}
        ),
        Document(
            text="""
            Machine Learning (ML) is a subset of artificial intelligence that provides systems the ability
            to automatically learn and improve from experience without being explicitly programmed. ML
            algorithms build a mathematical model based on sample data, known as training data, to make
            predictions or decisions. Common ML approaches include supervised learning (labeled data),
            unsupervised learning (unlabeled data), and reinforcement learning (reward-based).
            """,
            metadata={"source": "ML Essentials", "topic": "machine-learning", "difficulty": "intermediate"}
        ),
        Document(
            text="""
            Deep Learning is a subset of machine learning based on artificial neural networks with
            multiple layers (deep neural networks). These networks can learn hierarchical representations
            of data. Popular architectures include Convolutional Neural Networks (CNNs) for image processing,
            Recurrent Neural Networks (RNNs) for sequential data, and Transformers for language tasks.
            Deep learning has revolutionized computer vision, speech recognition, and natural language
            processing.
            """,
            metadata={"source": "Deep Learning Guide", "topic": "deep-learning", "difficulty": "advanced"}
        ),
        Document(
            text="""
            Natural Language Processing (NLP) enables computers to understand, interpret, and generate
            human language. NLP combines computational linguistics with machine learning and deep learning.
            Key tasks include text classification, named entity recognition, sentiment analysis, machine
            translation, and question answering. Modern NLP is dominated by Transformer-based models like
            BERT, GPT, and T5, which use attention mechanisms to process text.
            """,
            metadata={"source": "NLP Overview", "topic": "nlp", "difficulty": "intermediate"}
        ),
        Document(
            text="""
            Large Language Models (LLMs) are neural networks trained on massive amounts of text data to
            understand and generate human-like text. Examples include GPT-4, Claude, and PaLM. LLMs use
            the Transformer architecture with billions of parameters. They excel at tasks like text
            generation, summarization, translation, and question answering. However, they can produce
            hallucinations (incorrect but plausible-sounding information) and may lack access to current
            information.
            """,
            metadata={"source": "LLM Deep Dive", "topic": "llm", "difficulty": "advanced"}
        ),
        Document(
            text="""
            Retrieval Augmented Generation (RAG) combines information retrieval with text generation to
            address LLM limitations. RAG retrieves relevant documents from a knowledge base and provides
            them as context to the LLM for generation. This approach reduces hallucinations, provides
            access to up-to-date information, and enables source attribution. The RAG pipeline typically
            includes document chunking, embedding generation, vector storage, semantic retrieval, and
            context-aware generation.
            """,
            metadata={"source": "RAG Explained", "topic": "rag", "difficulty": "advanced"}
        ),
        Document(
            text="""
            Vector Databases store and index high-dimensional vector embeddings for fast similarity search.
            They use specialized algorithms like HNSW (Hierarchical Navigable Small World), IVF (Inverted
            File Index), and FAISS. Vector databases enable semantic search by finding documents with
            similar meaning rather than exact keyword matches. They are essential for RAG systems,
            recommendation engines, and similarity-based applications. Popular vector databases include
            Pinecone, Weaviate, Qdrant, and DXMT.
            """,
            metadata={"source": "Vector DB Guide", "topic": "vector-db", "difficulty": "intermediate"}
        ),
        Document(
            text="""
            Embeddings are dense vector representations of data (text, images, audio) that capture
            semantic meaning in a continuous vector space. Similar items have similar embeddings.
            Text embeddings are typically generated by models like OpenAI's text-embedding-ada-002,
            Sentence-BERT, or E5. Embeddings enable semantic search, clustering, and similarity
            comparisons. The dimensionality typically ranges from 384 to 1536 dimensions.
            """,
            metadata={"source": "Embeddings Primer", "topic": "embeddings", "difficulty": "intermediate"}
        ),
    ]


def create_evaluation_dataset() -> Dict[str, List]:
    """Create a ground truth dataset for evaluation.

    Each entry includes:
    - question: The query
    - ground_truth: The expected/reference answer
    - contexts: Relevant text passages (optional, for context recall)
    """
    return {
        "question": [
            "What is machine learning and what are its main approaches?",
            "How does RAG help reduce hallucinations in LLMs?",
            "What algorithms do vector databases use for similarity search?",
            "What is the difference between deep learning and traditional machine learning?",
            "What are embeddings and why are they important?",
        ],
        "ground_truth": [
            "Machine learning is a subset of AI that enables systems to learn from experience without "
            "explicit programming. Its main approaches are supervised learning with labeled data, "
            "unsupervised learning with unlabeled data, and reinforcement learning which is reward-based.",

            "RAG reduces hallucinations by retrieving relevant documents from a knowledge base and "
            "providing them as context to the LLM. This grounds the generation in factual information "
            "and enables source attribution.",

            "Vector databases use specialized algorithms like HNSW (Hierarchical Navigable Small World), "
            "IVF (Inverted File Index), and FAISS for fast similarity search.",

            "Deep learning uses artificial neural networks with multiple layers (deep neural networks) "
            "to learn hierarchical representations, while traditional machine learning typically uses "
            "simpler models. Deep learning has revolutionized fields like computer vision and NLP.",

            "Embeddings are dense vector representations that capture semantic meaning in continuous "
            "vector space. They enable semantic search, clustering, and similarity comparisons by "
            "representing similar items with similar vectors.",
        ],
    }


def build_rag_system():
    """Build and return a RAG system with DXMT."""
    print("\n" + "=" * 70)
    print("Building RAG System")
    print("=" * 70)

    # 1. Configure LlamaIndex
    print("\n1. Configuring LlamaIndex settings...")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    print("   ✓ Embedding model: text-embedding-ada-002")
    print("   ✓ LLM: gpt-3.5-turbo")

    # 2. Initialize DXMT vector store
    print("\n2. Initializing DXMT vector store...")
    vector_store = DXMTVectorStore(
        table_name="ragas_evaluation_kb",
        dimension=1536,
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
    )
    print("   ✓ Vector store initialized")

    # 3. Create index
    print("\n3. Creating vector index...")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    print("   ✓ Index created")

    # 4. Load documents
    print("\n4. Indexing knowledge base documents...")
    documents = create_knowledge_base_documents()

    for i, doc in enumerate(documents, 1):
        index.insert(doc)
        print(f"   ✓ Document {i}/{len(documents)}: {doc.metadata['source']}")

    # 5. Create query engine
    print("\n5. Creating query engine...")
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
    )
    print("   ✓ Query engine ready")

    return query_engine, index


def run_rag_queries(query_engine, eval_dataset: Dict[str, List]) -> Dict[str, List]:
    """Run RAG queries and collect results for RAGAS evaluation."""
    print("\n" + "=" * 70)
    print("Running RAG Queries")
    print("=" * 70)

    questions = eval_dataset["question"]
    answers = []
    contexts_list = []

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Query: {question[:60]}...")

        # Query the RAG system
        response = query_engine.query(question)

        # Extract answer
        answer = str(response)
        answers.append(answer)

        # Extract contexts from source nodes
        contexts = [node.text.strip() for node in response.source_nodes]
        contexts_list.append(contexts)

        print(f"   ✓ Answer: {answer[:80]}...")
        print(f"   ✓ Retrieved {len(contexts)} context passages")

    # Build result dataset
    result_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": eval_dataset["ground_truth"],
    }

    return result_data


def evaluate_with_ragas(result_data: Dict[str, List]):
    """Evaluate RAG system using RAGAS metrics."""
    if not RAGAS_AVAILABLE:
        print("\n❌ RAGAS is not available. Please install with:")
        print("   pip install ragas datasets")
        return

    print("\n" + "=" * 70)
    print("RAGAS Evaluation")
    print("=" * 70)

    # Create dataset
    print("\n1. Creating evaluation dataset...")
    dataset = Dataset.from_dict(result_data)
    print(f"   ✓ Dataset created with {len(dataset)} examples")

    # Configure RAGAS with LangChain models
    print("\n2. Configuring RAGAS evaluators...")
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        # Initialize LangChain models for RAGAS
        ragas_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        ragas_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        print("   ✓ RAGAS evaluators configured with LangChain models")
    except ImportError:
        print("   ⚠ LangChain not installed, using default RAGAS configuration")
        print("   Note: Some metrics may not work without LangChain")
        print("   Install with: pip install langchain-openai")
        ragas_llm = None
        ragas_embeddings = None

    # Define metrics
    print("\n3. Defining evaluation metrics...")

    # Use only metrics that work reliably without special configuration
    metrics = [
        faithfulness,           # Answer grounded in context
        context_precision,      # Relevant contexts ranked higher
        context_recall,         # All relevant info retrieved
    ]

    # Add metrics that require embeddings only if available
    if ragas_embeddings is not None:
        try:
            metrics.extend([
                answer_relevancy,       # Answer relevance to question (needs embeddings)
                answer_correctness,     # Factual accuracy (needs embeddings)
                answer_similarity,      # Similarity to ground truth (needs embeddings)
            ])
        except Exception as e:
            print(f"   ⚠ Could not add embedding-based metrics: {e}")

    metric_descriptions = {
        "faithfulness": "Measures if the answer is grounded in the retrieved context",
        "answer_relevancy": "Measures how relevant the answer is to the question",
        "context_precision": "Measures if relevant contexts are ranked higher",
        "context_recall": "Measures if all relevant information is retrieved",
        "answer_correctness": "Measures factual accuracy of the answer",
        "answer_similarity": "Measures semantic similarity to ground truth",
    }

    for metric in metrics:
        print(f"   ✓ {metric.name}: {metric_descriptions.get(metric.name, 'N/A')}")

    # Run evaluation
    print("\n4. Running evaluation (this may take a minute)...")
    print("   Note: This will make API calls to OpenAI for evaluation")

    try:
        # Prepare evaluation kwargs
        eval_kwargs = {"metrics": metrics}

        # Add LLM and embeddings if available
        if ragas_llm is not None:
            eval_kwargs["llm"] = ragas_llm
        if ragas_embeddings is not None:
            eval_kwargs["embeddings"] = ragas_embeddings

        results = evaluate(dataset, **eval_kwargs)

        print("\n" + "=" * 70)
        print("Evaluation Results")
        print("=" * 70)

        # Display overall scores
        print("\n📊 Overall Scores:")
        print("-" * 70)

        results_df = results.to_pandas()

        for metric in metrics:
            score = results_df[metric.name].mean()
            print(f"   {metric.name:25s}: {score:.4f} {'✓' if score >= 0.7 else '⚠'}")

        # Detailed per-question results
        print("\n📋 Per-Question Results:")
        print("-" * 70)

        for i, question in enumerate(result_data["question"]):
            print(f"\nQuestion {i+1}: {question[:60]}...")
            for metric in metrics:
                score = results_df[metric.name].iloc[i]
                print(f"   {metric.name:20s}: {score:.4f}")

        # Analysis and recommendations
        print("\n" + "=" * 70)
        print("Analysis & Recommendations")
        print("=" * 70)

        avg_scores = {metric.name: results_df[metric.name].mean() for metric in metrics}

        print("\n🎯 Strengths:")
        for metric_name, score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            if score >= 0.7:
                print(f"   ✓ {metric_name}: {score:.4f} - Excellent performance")

        print("\n⚠️  Areas for Improvement:")
        for metric_name, score in sorted(avg_scores.items(), key=lambda x: x[1])[:3]:
            if score < 0.7:
                print(f"   ! {metric_name}: {score:.4f} - Needs attention")

                if metric_name == "context_recall":
                    print("     → Increase similarity_top_k to retrieve more contexts")
                elif metric_name == "context_precision":
                    print("     → Improve ranking with reranking or better embeddings")
                elif metric_name == "faithfulness":
                    print("     → Adjust LLM temperature or use stricter prompts")
                elif metric_name == "answer_relevancy":
                    print("     → Refine retrieval or use query rewriting")

        # Save results
        print("\n4. Saving results...")
        output_file = "ragas_evaluation_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"   ✓ Results saved to: {output_file}")

        return results

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print(f"   Make sure you have set OPENAI_API_KEY environment variable")
        return None


def print_tips():
    """Print tips for improving RAG performance."""
    print("\n" + "=" * 70)
    print("Tips for Improving RAG Performance")
    print("=" * 70)
    print("""
    1. Chunking Strategy:
       - Optimal chunk size: 256-512 tokens
       - Use overlap (10-20%) to preserve context
       - Respect document boundaries (sentences, paragraphs)

    2. Retrieval Optimization:
       - Increase similarity_top_k for better recall
       - Use hybrid search (vector + keyword)
       - Implement reranking for better precision
       - Consider query rewriting for complex questions

    3. Generation Quality:
       - Lower temperature (0.0-0.3) for factual tasks
       - Use system prompts to enforce faithfulness
       - Implement citation mechanisms
       - Add answer verification steps

    4. Embedding Selection:
       - Better embeddings → better retrieval
       - Consider domain-specific models
       - Experiment with different dimensions

    5. Evaluation-Driven Iteration:
       - Use RAGAS metrics to identify weak points
       - A/B test different configurations
       - Monitor metrics in production
       - Collect user feedback

    6. DXMT-Specific Optimizations:
       - Use HNSW for high recall
       - Consider IVF-PQ for large-scale datasets
       - Tune distance metrics (cosine vs euclidean)
       - Leverage metadata filtering for precision
    """)


def main():
    """Run the complete RAGAS evaluation example."""
    print("=" * 70)
    print("DXMT Vector Store - RAGAS Evaluation Example")
    print("=" * 70)

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Please set it with: export OPENAI_API_KEY='your-key-here'")
        return

    if not RAGAS_AVAILABLE:
        print("\n❌ Error: RAGAS is not installed")
        print("   Please install with: pip install ragas datasets")
        return

    # Build RAG system
    query_engine, index = build_rag_system()

    # Create evaluation dataset
    eval_dataset = create_evaluation_dataset()

    # Run queries
    result_data = run_rag_queries(query_engine, eval_dataset)

    # Evaluate with RAGAS
    results = evaluate_with_ragas(result_data)

    # Print tips
    if results is not None:
        print_tips()

    # Cleanup
    print("\n" + "=" * 70)
    print("Cleanup")
    print("=" * 70)
    print("  (Table preserved for inspection)")

    print("\n" + "=" * 70)
    print("RAGAS Evaluation Example Completed!")
    print("=" * 70)
    print("""
    Summary:
    ✓ Built RAG system with DXMT
    ✓ Indexed 8 knowledge base documents
    ✓ Ran 5 evaluation queries
    ✓ Evaluated with 6 RAGAS metrics
    ✓ Generated performance report

    Next Steps:
    - Review ragas_evaluation_results.csv for detailed metrics
    - Iterate on RAG configuration based on weak metrics
    - Add more evaluation questions for comprehensive testing
    - Deploy the optimized RAG system
    """)


if __name__ == "__main__":
    main()

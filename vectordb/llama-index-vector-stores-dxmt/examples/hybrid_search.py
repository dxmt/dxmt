"""Hybrid search example for DXMT Vector Store with LlamaIndex.

This example demonstrates:
1. Enabling full-text search
2. Performing hybrid searches (vector + text)
3. Adjusting search weights with alpha parameter
4. Comparing different search modes
"""

from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from embedding import default_embed_model, embed_dimension, has_llm
import statistics


def print_search_results(results, mode_name, show_score_stats=True):
    """Helper function to print search results with statistics."""
    if len(results.nodes) == 0:
        print(f"    ⚠️  Warning: No {mode_name} results!")
        return

    for i, (node, score) in enumerate(zip(results.nodes, results.similarities), 1):
        print(f"    {i}. Score: {score:.4f}")
        print(f"       {node.text[:80]}...")

    # Check for duplicates
    unique_texts = set(node.text for node in results.nodes)
    if len(unique_texts) < len(results.nodes):
        print(f"    ⚠️  Warning: Detected {len(results.nodes) - len(unique_texts)} duplicate result(s)")

    # Show score statistics
    if show_score_stats and len(results.similarities) > 1:
        scores = results.similarities
        print(f"    📊 Scores: range=[{min(scores):.4f}, {max(scores):.4f}], "
              f"std={statistics.stdev(scores):.4f}")


def test_query(query_text, query_type, expected_keywords, vector_store, embed_model):
    """Test a single query across all search modes."""
    print(f"\n  {'='*70}")
    print(f"  Query Type: {query_type}")
    print(f"  Query: '{query_text}'")
    print(f"  Expected keywords: {expected_keywords}")
    print(f"  {'='*70}")

    # Get embedding
    query_embedding = embed_model.get_text_embedding(query_text)

    # A. Vector Search
    print("\n  A. Vector Search Only:")
    vector_query = VectorStoreQuery(
        query_embedding=query_embedding,
        mode=VectorStoreQueryMode.DEFAULT,
        similarity_top_k=3,
    )
    vector_results = vector_store.query(vector_query)
    print_search_results(vector_results, "vector search")

    # Check if expected keywords appear in top result
    if vector_results.nodes:
        top_text = vector_results.nodes[0].text.lower()
        found = any(kw.lower() in top_text for kw in expected_keywords)
        if found:
            print(f"    ✓ Top result contains expected keywords")
        else:
            print(f"    ⚠️ Top result may not be most relevant")

    # B. Text Search (BM25)
    print("\n  B. Text Search (BM25):")
    text_query = VectorStoreQuery(
        query_str=query_text,
        mode=VectorStoreQueryMode.TEXT_SEARCH,
        similarity_top_k=3,
    )
    text_results = vector_store.query(text_query)
    print_search_results(text_results, "text search")

    if text_results.nodes:
        top_text = text_results.nodes[0].text.lower()
        found = any(kw.lower() in top_text for kw in expected_keywords)
        if found:
            print(f"    ✓ Top result contains expected keywords")

    # C. Hybrid Search (Alpha = 0.5)
    print("\n  C. Hybrid Search (Alpha=0.5, 50% vector + 50% text):")
    hybrid_query = VectorStoreQuery(
        query_embedding=query_embedding,
        query_str=query_text,
        mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=3,
        alpha=0.5,
    )
    hybrid_results = vector_store.query(hybrid_query)
    print_search_results(hybrid_results, "hybrid search")

    if hybrid_results.nodes:
        top_text = hybrid_results.nodes[0].text.lower()
        found = any(kw.lower() in top_text for kw in expected_keywords)
        if found:
            print(f"    ✓ Top result contains expected keywords")


def main():
    """Run hybrid search example."""
    print("=" * 60)
    print("DXMT Vector Store - Hybrid Search Example")
    print("=" * 60)

    embed_model = default_embed_model()
    table_name = f"llamaindex_hybrid_example_{embed_dimension(embed_model)}"

    # 0. Clean up existing table to avoid duplicates. It has to be the name this run will use:
    # the table carries the embedding model's dimension, and deleting a different one left the
    # documents from the previous run in place -- every run then showed each document twice.
    print("\n0. Cleaning up existing table...")
    try:
        from dxmt import Client
        client = Client(host="localhost", port=9999)
        client.delete_table(table_name)
        print("✓ Existing table deleted")
    except Exception:
        print("ℹ No existing table to clean up")

    # 1. Initialize vector store with full-text search enabled
    print("\n1. Initializing DXMT vector store with full-text search...")
    vector_store = DXMTVectorStore(
        table_name=table_name,
        dimension=embed_dimension(embed_model),
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
        enable_full_text_search=True,  # Enable BM25 full-text search
        text_fields=[
            {
                "name": "content",
                "source": {"metadata_field": "_node_content"},  # LlamaIndex stores text in _node_content
                "analyzer": "standard",
                "boost": 1.0,
            }
        ],
    )
    print("✓ Vector store created with full-text search")

    # 2. Create index
    print("\n2. Creating LlamaIndex index...")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    print("✓ Index created")

    # 3. Add documents (expanded to 50+ for better testing)
    print("\n3. Adding documents about machine learning and AI...")
    documents = [
        # Core ML Concepts (10 docs)
        Document(
            text="Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn patterns from data.",
            metadata={"category": "ML", "difficulty": "beginner", "topic": "basics"}
        ),
        Document(
            text="Deep learning uses neural networks with multiple layers to learn hierarchical representations of data.",
            metadata={"category": "DL", "difficulty": "intermediate", "topic": "neural_networks"}
        ),
        Document(
            text="Supervised learning requires labeled training data, where the algorithm learns to map inputs to known outputs.",
            metadata={"category": "ML", "difficulty": "beginner", "topic": "supervised"}
        ),
        Document(
            text="Unsupervised learning discovers hidden patterns in unlabeled data without explicit guidance.",
            metadata={"category": "ML", "difficulty": "intermediate", "topic": "unsupervised"}
        ),
        Document(
            text="Reinforcement learning trains agents through trial and error, using rewards to guide behavior.",
            metadata={"category": "RL", "difficulty": "advanced", "topic": "reinforcement"}
        ),
        Document(
            text="Semi-supervised learning combines small amounts of labeled data with large amounts of unlabeled data during training.",
            metadata={"category": "ML", "difficulty": "intermediate", "topic": "semi_supervised"}
        ),
        Document(
            text="Transfer learning allows models to apply knowledge learned from one task to improve performance on related tasks.",
            metadata={"category": "ML", "difficulty": "advanced", "topic": "transfer"}
        ),
        Document(
            text="Feature engineering is the process of selecting and transforming raw data into meaningful features for machine learning models.",
            metadata={"category": "ML", "difficulty": "intermediate", "topic": "features"}
        ),
        Document(
            text="Overfitting occurs when a model learns training data too well, including noise, and fails to generalize to new data.",
            metadata={"category": "ML", "difficulty": "intermediate", "topic": "optimization"}
        ),
        Document(
            text="Cross-validation is a technique to assess model performance by partitioning data into training and validation sets multiple times.",
            metadata={"category": "ML", "difficulty": "intermediate", "topic": "validation"}
        ),

        # Neural Networks (10 docs)
        Document(
            text="Convolutional neural networks (CNNs) excel at processing grid-like data such as images using convolutional layers.",
            metadata={"category": "DL", "difficulty": "advanced", "topic": "cnn"}
        ),
        Document(
            text="Recurrent neural networks (RNNs) process sequential data by maintaining hidden states that capture temporal dependencies.",
            metadata={"category": "DL", "difficulty": "advanced", "topic": "rnn"}
        ),
        Document(
            text="Long short-term memory (LSTM) networks address the vanishing gradient problem in RNNs using gating mechanisms.",
            metadata={"category": "DL", "difficulty": "advanced", "topic": "lstm"}
        ),
        Document(
            text="Transformers revolutionized NLP by using self-attention mechanisms to process sequences in parallel.",
            metadata={"category": "NLP", "difficulty": "advanced", "topic": "transformers"}
        ),
        Document(
            text="Attention mechanisms allow models to focus on relevant parts of input sequences when making predictions.",
            metadata={"category": "DL", "difficulty": "advanced", "topic": "attention"}
        ),
        Document(
            text="Generative adversarial networks (GANs) consist of generator and discriminator networks competing against each other.",
            metadata={"category": "DL", "difficulty": "advanced", "topic": "gan"}
        ),
        Document(
            text="Autoencoders learn compressed representations of data through encoder-decoder architectures.",
            metadata={"category": "DL", "difficulty": "advanced", "topic": "autoencoder"}
        ),
        Document(
            text="Batch normalization improves training stability by normalizing layer inputs across mini-batches.",
            metadata={"category": "DL", "difficulty": "advanced", "topic": "optimization"}
        ),
        Document(
            text="Dropout is a regularization technique that randomly deactivates neurons during training to prevent overfitting.",
            metadata={"category": "DL", "difficulty": "intermediate", "topic": "regularization"}
        ),
        Document(
            text="Residual networks (ResNets) use skip connections to enable training of very deep neural networks.",
            metadata={"category": "DL", "difficulty": "advanced", "topic": "architectures"}
        ),

        # NLP (10 docs)
        Document(
            text="Natural language processing (NLP) enables computers to understand and generate human language using ML techniques.",
            metadata={"category": "NLP", "difficulty": "intermediate", "topic": "basics"}
        ),
        Document(
            text="Word embeddings like Word2Vec and GloVe represent words as dense vectors capturing semantic relationships.",
            metadata={"category": "NLP", "difficulty": "intermediate", "topic": "embeddings"}
        ),
        Document(
            text="BERT (Bidirectional Encoder Representations from Transformers) uses bidirectional context for language understanding.",
            metadata={"category": "NLP", "difficulty": "advanced", "topic": "bert"}
        ),
        Document(
            text="GPT (Generative Pre-trained Transformer) models generate coherent text using autoregressive language modeling.",
            metadata={"category": "NLP", "difficulty": "advanced", "topic": "gpt"}
        ),
        Document(
            text="Named entity recognition (NER) identifies and classifies entities like persons, organizations, and locations in text.",
            metadata={"category": "NLP", "difficulty": "intermediate", "topic": "ner"}
        ),
        Document(
            text="Sentiment analysis determines the emotional tone or opinion expressed in text using classification techniques.",
            metadata={"category": "NLP", "difficulty": "beginner", "topic": "sentiment"}
        ),
        Document(
            text="Machine translation converts text from one language to another using sequence-to-sequence models.",
            metadata={"category": "NLP", "difficulty": "advanced", "topic": "translation"}
        ),
        Document(
            text="Text summarization condenses long documents into shorter versions while preserving key information.",
            metadata={"category": "NLP", "difficulty": "intermediate", "topic": "summarization"}
        ),
        Document(
            text="Question answering systems retrieve or generate answers to natural language questions from knowledge sources.",
            metadata={"category": "NLP", "difficulty": "advanced", "topic": "qa"}
        ),
        Document(
            text="Tokenization splits text into smaller units like words or subwords for processing by NLP models.",
            metadata={"category": "NLP", "difficulty": "beginner", "topic": "preprocessing"}
        ),

        # Computer Vision (10 docs)
        Document(
            text="Computer vision enables machines to interpret and understand visual information from images and videos.",
            metadata={"category": "CV", "difficulty": "beginner", "topic": "basics"}
        ),
        Document(
            text="Object detection identifies and localizes multiple objects in images using bounding boxes and class labels.",
            metadata={"category": "CV", "difficulty": "intermediate", "topic": "detection"}
        ),
        Document(
            text="Image segmentation partitions images into meaningful regions or assigns class labels to each pixel.",
            metadata={"category": "CV", "difficulty": "advanced", "topic": "segmentation"}
        ),
        Document(
            text="YOLO (You Only Look Once) performs real-time object detection by predicting bounding boxes and probabilities simultaneously.",
            metadata={"category": "CV", "difficulty": "advanced", "topic": "yolo"}
        ),
        Document(
            text="R-CNN and its variants use region proposals for accurate object detection in images.",
            metadata={"category": "CV", "difficulty": "advanced", "topic": "rcnn"}
        ),
        Document(
            text="Image classification assigns category labels to entire images based on their visual content.",
            metadata={"category": "CV", "difficulty": "beginner", "topic": "classification"}
        ),
        Document(
            text="Facial recognition systems identify individuals by analyzing facial features using deep learning models.",
            metadata={"category": "CV", "difficulty": "intermediate", "topic": "face_recognition"}
        ),
        Document(
            text="Optical character recognition (OCR) converts images of text into machine-readable text format.",
            metadata={"category": "CV", "difficulty": "intermediate", "topic": "ocr"}
        ),
        Document(
            text="Style transfer applies artistic styles from one image to the content of another using neural networks.",
            metadata={"category": "CV", "difficulty": "advanced", "topic": "style_transfer"}
        ),
        Document(
            text="Image augmentation artificially expands training datasets by applying transformations like rotation and flipping.",
            metadata={"category": "CV", "difficulty": "intermediate", "topic": "augmentation"}
        ),

        # Applications & Tools (10 docs)
        Document(
            text="TensorFlow is an open-source machine learning framework developed by Google for building and training models.",
            metadata={"category": "Tools", "difficulty": "intermediate", "topic": "tensorflow"}
        ),
        Document(
            text="PyTorch is a popular deep learning framework known for its dynamic computation graphs and ease of use.",
            metadata={"category": "Tools", "difficulty": "intermediate", "topic": "pytorch"}
        ),
        Document(
            text="Scikit-learn provides simple and efficient tools for data mining and machine learning in Python.",
            metadata={"category": "Tools", "difficulty": "beginner", "topic": "sklearn"}
        ),
        Document(
            text="Keras is a high-level neural network API that runs on top of TensorFlow with user-friendly interfaces.",
            metadata={"category": "Tools", "difficulty": "beginner", "topic": "keras"}
        ),
        Document(
            text="Gradient boosting machines like XGBoost and LightGBM achieve state-of-the-art results on structured data.",
            metadata={"category": "ML", "difficulty": "advanced", "topic": "boosting"}
        ),
        Document(
            text="Random forests combine multiple decision trees to improve prediction accuracy and reduce overfitting.",
            metadata={"category": "ML", "difficulty": "intermediate", "topic": "random_forest"}
        ),
        Document(
            text="Support vector machines (SVMs) find optimal hyperplanes that maximize margins between different classes.",
            metadata={"category": "ML", "difficulty": "advanced", "topic": "svm"}
        ),
        Document(
            text="K-means clustering partitions data into K groups by minimizing within-cluster variance.",
            metadata={"category": "ML", "difficulty": "intermediate", "topic": "clustering"}
        ),
        Document(
            text="Principal component analysis (PCA) reduces data dimensionality by projecting it onto principal components.",
            metadata={"category": "ML", "difficulty": "advanced", "topic": "pca"}
        ),
        Document(
            text="Hyperparameter tuning optimizes model performance by searching for the best configuration of learning parameters.",
            metadata={"category": "ML", "difficulty": "intermediate", "topic": "tuning"}
        ),
    ]

    for i, doc in enumerate(documents, 1):
        index.insert(doc)
        if i % 10 == 0 or i == len(documents):
            print(f"  ✓ {i}/{len(documents)} documents added")

    # Print dataset statistics
    print(f"\n  Dataset Statistics:")
    print(f"  - Total documents: {len(documents)}")
    categories = {}
    difficulties = {}
    for doc in documents:
        cat = doc.metadata.get("category", "Unknown")
        diff = doc.metadata.get("difficulty", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
    print(f"  - Categories: {dict(categories)}")
    print(f"  - Difficulty levels: {dict(difficulties)}")

    # 4. Test with multiple diverse queries
    print("\n4. Testing with diverse query types...")

    # Test Query 1: Exact keyword match (BM25 should excel)
    test_query(
        query_text="deep learning neural networks",
        query_type="Exact Keyword Match",
        expected_keywords=["deep learning", "neural networks"],
        vector_store=vector_store,
        embed_model=embed_model
    )

    # Test Query 2: Semantic understanding (Vector should excel)
    test_query(
        query_text="How do machines acquire knowledge from examples?",
        query_type="Semantic Understanding",
        expected_keywords=["supervised", "learning", "training", "labeled"],
        vector_store=vector_store,
        embed_model=embed_model
    )

    # Test Query 3: Mixed query (Hybrid should excel)
    test_query(
        query_text="transformers attention mechanisms NLP",
        query_type="Mixed Query (Keywords + Concepts)",
        expected_keywords=["transformers", "attention", "NLP"],
        vector_store=vector_store,
        embed_model=embed_model
    )

    # 5. Alpha parameter comparison (using first query)
    print("\n5. Alpha parameter comparison (using 'neural networks and deep learning')...")
    query_text = "neural networks and deep learning"
    query_embedding = embed_model.get_text_embedding(query_text)

    for alpha_value in [0.3, 0.5, 0.7]:
        weight_desc = f"{int(alpha_value*100)}% vector + {int((1-alpha_value)*100)}% text"
        print(f"\n  Alpha = {alpha_value} ({weight_desc}):")

        hybrid_query = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=query_text,
            mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=3,
            alpha=alpha_value,
        )
        results = vector_store.query(hybrid_query)

        for i, (node, score) in enumerate(zip(results.nodes, results.similarities), 1):
            print(f"    {i}. Score: {score:.4f}")
            print(f"       {node.text[:80]}...")

        # Show score statistics
        if len(results.similarities) > 1:
            scores = results.similarities
            print(f"    📊 Score range: [{min(scores):.4f}, {max(scores):.4f}], "
                  f"std: {statistics.stdev(scores):.4f}")

    # 6. Query engine demonstration. Generating an answer needs an LLM; the retrieval under
    # it does not, so without a key this shows the nodes DXMT returned.
    print("\n6. Using query engine for RAG...")
    if has_llm():
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query("What is deep learning?")
        print(f"\n  Query: 'What is deep learning?'")
        print(f"  Response: {response}")
    else:
        print("\n  Query: 'What is deep learning?' (retrieval only -- no OPENAI_API_KEY)")
        for node in index.as_retriever(similarity_top_k=3).retrieve("What is deep learning?"):
            print(f"    {node.score:.3f}  {node.node.text[:70]}...")

    # 7. Summary and Analysis
    print("\n7. Summary and Analysis:")
    print("  ✓ Tested with 50 diverse documents across 5 categories")
    print("  ✓ Evaluated 3 different query types:")
    print("    - Exact keyword matching → BM25 excels")
    print("    - Semantic understanding → Vector search excels")
    print("    - Mixed queries → Hybrid search excels")
    print("")
    print("  Key Findings:")
    print("  - Vector search: Best for semantic/conceptual queries")
    print("  - BM25 search: Best for exact keyword matching")
    print("  - Hybrid search: Combines strengths of both")
    print("")
    print("  Alpha Parameter Guide:")
    print("    * Alpha = 0.7-1.0: More weight on semantic similarity")
    print("    * Alpha = 0.4-0.6: Balanced approach (recommended)")
    print("    * Alpha = 0.0-0.3: More weight on keyword matching")
    print("")
    print("  💡 Tip: Choose alpha based on your query type!")

    # 8. Clean up (optional)
    print("\n8. Cleanup...")
    # vector_store.clear()
    print("  (Skipped - table preserved for inspection)")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

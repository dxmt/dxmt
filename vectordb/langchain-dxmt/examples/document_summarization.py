"""Intelligent Document Summarization using DXMT and LangChain.

This example demonstrates how to:
- Split large documents into manageable chunks
- Store chunks in DXMT vector store
- Retrieve relevant sections for targeted summarization
- Generate comprehensive summaries using LLMs

Prerequisites:
- DXMT server running on localhost:9999
- OpenAI API key set in environment: export OPENAI_API_KEY='sk-...'

Install dependencies:
    pip install langchain-dxmt langchain-openai langchain langchain-community langchain-classic langchain-text-splitters

Use case:
    Summarize long documents (research papers, legal docs, reports),
    extract key information, or generate executive summaries.
"""

import os
from typing import List

from langchain_classic.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_dxmt import DXMTVectorDB


def get_sample_document() -> str:
    """Get a sample long document for demonstration."""
    # Sample: A mock AI research paper abstract + introduction
    return """
    Title: Advances in Large Language Models: A Comprehensive Survey

    Abstract:
    Large Language Models (LLMs) have revolutionized natural language processing
    and artificial intelligence. This survey provides a comprehensive overview of
    recent advances in LLM development, training methodologies, and applications.
    We discuss architectural innovations, scaling laws, emergent capabilities, and
    the challenges facing the field. Our analysis covers models from GPT-3 to
    GPT-4, PaLM, LLaMA, and beyond, examining their performance across diverse tasks.

    1. Introduction

    The past five years have witnessed unprecedented progress in natural language
    processing, driven primarily by the development of increasingly large neural
    language models. These models, collectively known as Large Language Models (LLMs),
    have demonstrated remarkable capabilities in understanding and generating human
    language. Beginning with BERT in 2018 and accelerating with GPT-3 in 2020, the
    field has seen exponential growth in both model size and capabilities.

    LLMs are typically based on the Transformer architecture, introduced by Vaswani
    et al. in 2017. The key innovation of Transformers is the self-attention mechanism,
    which allows the model to weigh the importance of different words in a sequence
    when processing each word. This parallel processing capability, combined with
    massive scale, has proven extraordinarily effective for language tasks.

    2. Model Architectures

    Modern LLMs employ various architectural designs, but most share common foundations.
    The encoder-decoder architecture, used in models like T5 and BART, processes input
    and output through separate components. Decoder-only models like GPT series focus
    solely on autoregressive generation, predicting the next token given previous context.
    Encoder-only models like BERT excel at understanding tasks but cannot generate text.

    Recent innovations include sparse attention mechanisms that reduce computational
    complexity from O(n²) to O(n log n) or O(n). Models like Longformer and BigBird
    enable processing of much longer contexts. Mixture-of-Experts (MoE) architectures,
    used in models like Switch Transformer, activate only a subset of parameters for
    each input, allowing even larger effective model sizes while maintaining reasonable
    computational costs.

    3. Training Methodologies

    Training LLMs requires massive computational resources and carefully curated datasets.
    Pre-training typically uses self-supervised learning on large text corpora, with
    objectives like masked language modeling (BERT) or next-token prediction (GPT).
    The scale of training has grown dramatically: GPT-3 used 175 billion parameters
    and was trained on 300 billion tokens, while recent models exceed 500 billion
    parameters and train on trillions of tokens.

    Fine-tuning adapts pre-trained models to specific tasks. Supervised fine-tuning
    uses labeled data, while reinforcement learning from human feedback (RLHF) aligns
    model outputs with human preferences. RLHF has proven crucial for creating helpful,
    harmless, and honest AI assistants. Parameter-efficient fine-tuning methods like
    LoRA and prefix tuning enable adaptation with minimal computational cost.

    4. Emergent Capabilities

    A fascinating phenomenon in LLMs is the emergence of capabilities not explicitly
    trained for. As models scale, they spontaneously develop abilities like few-shot
    learning, chain-of-thought reasoning, and even basic arithmetic. These emergent
    capabilities suggest that language modeling at sufficient scale captures deeper
    patterns of reasoning and knowledge representation.

    Few-shot learning allows models to perform tasks from just a few examples in the
    prompt, without any parameter updates. Chain-of-thought prompting, where models
    are encouraged to show their reasoning steps, dramatically improves performance
    on complex reasoning tasks. Recent work shows that models can even learn to use
    external tools like calculators, search engines, and code interpreters.

    5. Applications and Impact

    LLMs have found applications across virtually every domain involving text. In
    software development, tools like GitHub Copilot assist programmers by suggesting
    code completions and entire functions. In content creation, models generate articles,
    marketing copy, and creative writing. Customer service chatbots powered by LLMs
    provide natural, helpful responses at scale.

    The scientific community uses LLMs for literature review, hypothesis generation,
    and even experimental design. Education applications include personalized tutoring,
    automated essay grading, and interactive learning experiences. In healthcare, LLMs
    assist with medical documentation, patient communication, and clinical decision
    support, though with appropriate caution given the stakes.

    6. Challenges and Limitations

    Despite impressive capabilities, LLMs face significant challenges. Hallucination—
    generating plausible but incorrect information—remains a persistent problem. Models
    lack true understanding and can be fooled by adversarial inputs. Bias in training
    data propagates to model outputs, raising concerns about fairness and representation.

    Computational costs are staggering, both financial and environmental. Training
    the largest models costs tens of millions of dollars and produces significant
    carbon emissions. Inference costs limit deployment at scale. Memory requirements
    make running large models accessible only to well-resourced organizations.

    Privacy and security concerns arise from models' ability to memorize training data,
    potentially leaking sensitive information. The technology's potential for misuse
    in generating misinformation, deepfakes, and malicious content necessitates
    thoughtful governance and safety measures.

    7. Future Directions

    The field continues to evolve rapidly. Research directions include improving
    efficiency through better architectures and compression techniques. Multimodal
    models that process text, images, and other modalities together show promise.
    Retrieval-augmented generation, combining LLMs with external knowledge bases,
    addresses hallucination and knowledge staleness.

    Interpretability research aims to understand what models learn and how they make
    decisions. Alignment research focuses on ensuring AI systems reliably pursue
    intended goals. Constitutional AI and other approaches explore embedding values
    and constraints directly into model training.

    8. Conclusion

    Large Language Models represent a transformative technology with vast potential
    and significant challenges. As capabilities continue to improve and costs decrease,
    LLMs will likely become ubiquitous in our digital infrastructure. Responsible
    development, deployment, and governance will be essential to realizing benefits
    while mitigating risks. The next decade promises continued rapid progress in this
    exciting field.
    """


def main():
    """Run the document summarization example."""
    print("=" * 80)
    print("Document Summarization with DXMT")
    print("=" * 80)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        return

    # 1. Get document
    print("\n1. Loading document...")
    doc_text = get_sample_document()
    doc_length = len(doc_text)
    word_count = len(doc_text.split())

    print(f"   Document length: {doc_length:,} characters")
    print(f"   Word count: {word_count:,} words")
    print(f"   ✅ Document loaded")

    # 2. Split document into chunks
    print("\n2. Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Characters per chunk
        chunk_overlap=50,  # Overlap for context preservation
        separators=["\n\n", "\n", ". ", " ", ""]  # Split on paragraphs first
    )

    texts = text_splitter.split_text(doc_text)
    print(f"   Created {len(texts)} chunks")

    # Create Document objects with metadata
    chunks = [
        Document(
            page_content=text,
            metadata={
                "chunk_id": i,
                "source": "ai_survey_paper.txt",
                "total_chunks": len(texts)
            }
        )
        for i, text in enumerate(texts)
    ]
    print(f"   ✅ Document split into {len(chunks)} chunks")

    # 3. Store in DXMT vector store
    print("\n3. Storing chunks in DXMT vector store...")
    embeddings = OpenAIEmbeddings()

    vectorstore = DXMTVectorDB(
        table_name="doc_summarization_demo",
        embedding=embeddings,
        dimension=1536,
        host="localhost",
        port=9999,
        distance_metric="cosine",
        index_type="hnsw",
    )

    ids = vectorstore.add_documents(chunks)
    print(f"   ✅ Stored {len(ids)} chunks in vector store")

    # 4. Initialize LLM
    print("\n4. Initializing LLM...")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,  # Deterministic for consistency
    )
    print("   ✅ LLM ready")

    # 5. Method 1: Summarize entire document (Map-Reduce)
    print("\n" + "=" * 80)
    print("Method 1: Full Document Summary (Map-Reduce)")
    print("=" * 80)
    print("\n💡 This processes all chunks and combines into one summary\n")

    # Use stuff chain for simple concatenation (works for moderate-sized docs)
    summarize_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",  # Summarize each chunk, then summarize summaries
        verbose=False
    )

    print("Generating summary...")
    full_summary = summarize_chain.invoke({"input_documents": chunks})
    print("\n📄 Full Document Summary:")
    print("─" * 80)
    print(full_summary["output_text"])
    print("─" * 80)

    # 6. Method 2: Targeted summarization using retrieval
    print("\n" + "=" * 80)
    print("Method 2: Targeted Summarization (Query-Focused)")
    print("=" * 80)
    print("\n💡 Retrieve relevant sections, then summarize only those\n")

    queries = [
        "What are the main architectural innovations in LLMs?",
        "What are the challenges and limitations of LLMs?",
        "What applications do LLMs have in practice?",
    ]

    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("─" * 80)

        # Retrieve relevant chunks
        relevant_chunks = vectorstore.similarity_search(query, k=3)

        print(f"Retrieved {len(relevant_chunks)} relevant chunks:")
        for i, chunk in enumerate(relevant_chunks, 1):
            chunk_id = chunk.metadata.get("chunk_id", "N/A")
            preview = chunk.page_content[:80] + "..."
            print(f"   {i}. Chunk {chunk_id}: {preview}")

        # Summarize retrieved chunks
        targeted_chain = load_summarize_chain(
            llm,
            chain_type="stuff",  # Concatenate chunks
            verbose=False
        )

        summary_result = targeted_chain.invoke({"input_documents": relevant_chunks})
        print(f"\n📝 Summary:\n{summary_result['output_text']}\n")

    # 7. Method 3: Custom prompt for structured summary
    print("\n" + "=" * 80)
    print("Method 3: Structured Summary with Custom Prompt")
    print("=" * 80)
    print("\n💡 Use custom prompts for specific summary formats\n")

    # Define custom prompt
    custom_prompt = PromptTemplate(
        template="""Write a structured summary of the following text with these sections:
- Main Topic (one sentence)
- Key Points (3-5 bullet points)
- Conclusion (one sentence)

Text:
{text}

Structured Summary:""",
        input_variables=["text"]
    )

    custom_chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=custom_prompt,
        verbose=False
    )

    # Get chunks about "emergent capabilities"
    query_ec = "emergent capabilities and few-shot learning"
    relevant_ec = vectorstore.similarity_search(query_ec, k=2)

    print(f"🔍 Topic: Emergent Capabilities in LLMs")
    print("─" * 80)

    structured_summary = custom_chain.invoke({"input_documents": relevant_ec})
    print(structured_summary["output_text"])
    print("─" * 80)

    # 8. Method 4: Extractive summary (show actual quotes)
    print("\n" + "=" * 80)
    print("Method 4: Extractive Summary (Key Sentences)")
    print("=" * 80)
    print("\n💡 Extract most important sentences directly from text\n")

    query_key = "main findings and future directions"
    key_chunks = vectorstore.similarity_search_with_score(query_key, k=3)

    print(f"🔍 Extracting key sentences about: \"{query_key}\"")
    print("─" * 80)
    print("\n📌 Key Excerpts:\n")

    for i, (chunk, score) in enumerate(key_chunks, 1):
        # Get first 2 sentences from each chunk
        sentences = chunk.page_content.split(". ")[:2]
        excerpt = ". ".join(sentences) + "."

        print(f"{i}. (Relevance: {score:.4f})")
        print(f"   {excerpt}\n")

    # 9. Performance comparison
    print("\n" + "=" * 80)
    print("Summary Methods Comparison")
    print("=" * 80)
    print("""
    ┌─────────────────────┬──────────────┬────────────┬─────────────────┐
    │ Method              │ Speed        │ Coverage   │ Best For        │
    ├─────────────────────┼──────────────┼────────────┼─────────────────┤
    │ Map-Reduce          │ Moderate     │ Complete   │ Full documents  │
    │ Targeted Retrieval  │ Fast         │ Focused    │ Specific topics │
    │ Custom Prompt       │ Fast         │ Structured │ Formatted output│
    │ Extractive          │ Fastest      │ Quotes     │ Key passages    │
    └─────────────────────┴──────────────┴────────────┴─────────────────┘
    """)

    # 10. Cleanup
    print("\n" + "=" * 80)
    print("Cleanup")
    print("=" * 80)

    print("\nDeleting document chunks...")
    vectorstore.delete(ids)
    print("✅ Chunks deleted")

    print("\n" + "=" * 80)
    print("💡 Key Takeaways")
    print("=" * 80)
    print("""
    ✅ Document Chunking:
       - Split large docs into manageable pieces
       - Preserve context with chunk overlap
       - Store metadata for tracking

    ✅ Vector Storage:
       - Enable semantic search across chunks
       - Retrieve most relevant sections quickly
       - Scale to millions of documents

    ✅ Multiple Summary Methods:
       1. Map-Reduce: Comprehensive, processes all content
       2. Targeted: Fast, focuses on specific topics
       3. Custom Prompts: Structured output formats
       4. Extractive: Direct quotes, no generation

    💡 Production Tips:
    1. Choose chunk size based on document type (papers: 500-1000, books: 1000-2000)
    2. Use metadata for filtering (date, author, section)
    3. Cache summaries for frequently accessed documents
    4. Implement incremental updates as documents change
    5. Add quality checks for generated summaries
    6. Consider cost: targeted summaries use fewer tokens

    💡 Use Cases:
    - Research paper analysis
    - Legal document review
    - News article digests
    - Meeting transcript summaries
    - Technical documentation overviews
    - Book chapter summaries
    """)


if __name__ == "__main__":
    main()

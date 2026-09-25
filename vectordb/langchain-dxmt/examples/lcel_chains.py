"""
LCEL (LangChain Expression Language) Chains Example

This example demonstrates modern LangChain chain building using LCEL syntax.
LCEL is the recommended way to build chains in LangChain (2024+).

Benefits of LCEL:
- Clearer syntax with pipe (|) operator
- Built-in streaming support
- Better error handling
- Easier debugging and introspection
- Type safety with Pydantic
- More composable and testable

Patterns demonstrated:
1. Simple RAG chain
2. Conversational RAG with history
3. Conditional routing
4. Parallel execution
5. Custom runnables
"""

import os
from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage
from embedding import default_embeddings, embed_dimension, needs_an_llm

# `langchain_openai` is imported where it is used, not here: without the
# package this example died at import time, before it could say what it needed.
from langchain_dxmt import DXMTVectorDB


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def format_docs(docs: List[Document]) -> str:
    """Format documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    # Check for OpenAI API key
    if not needs_an_llm("this example"):
        return

    from langchain_openai import ChatOpenAI

    print_section("LCEL (LangChain Expression Language) Modern Chains")

    # Initialize components
    embeddings = default_embeddings()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Create vector store
    print("Setting up DXMT vector store...")
    vectorstore = DXMTVectorDB(
        table_name="lcel_demo",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
    )

    # Clean up
    try:
        from dxmt import Client
        client = Client(host="localhost", port=9999)
        client.delete_table("lcel_demo")
        print("   ✓ Cleaned existing table")
    except:
        pass

    # Recreate
    vectorstore = DXMTVectorDB(
        table_name="lcel_demo",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
    )

    # Add sample documents
    docs = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, web development, and automation.",
            metadata={"topic": "python", "category": "programming"}
        ),
        Document(
            page_content="Machine learning is a subset of AI that enables systems to learn from data. Popular frameworks include TensorFlow, PyTorch, and scikit-learn.",
            metadata={"topic": "ml", "category": "ai"}
        ),
        Document(
            page_content="Docker containers package applications with their dependencies for consistent deployment. Kubernetes orchestrates containers at scale.",
            metadata={"topic": "devops", "category": "infrastructure"}
        ),
        Document(
            page_content="REST APIs use HTTP methods like GET, POST, PUT, DELETE for CRUD operations. GraphQL offers a flexible alternative with precise data fetching.",
            metadata={"topic": "api", "category": "web"}
        ),
    ]

    print("Adding documents...")
    vectorstore.add_documents(docs)
    print(f"   ✓ Added {len(docs)} documents\n")

    # ==========================================================================
    # Example 1: Simple RAG Chain with LCEL
    # ==========================================================================
    print_section("Example 1: Simple RAG Chain (LCEL Syntax)")

    print("Building chain with pipe operator...\n")

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Define prompt template
    template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer: """

    prompt = ChatPromptTemplate.from_template(template)

    # Build chain using LCEL (pipe operator)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Chain structure:")
    print("   Input → Retriever → Format → Prompt → LLM → Output Parser → Answer\n")

    # Test the chain
    question = "What is Python used for?"
    print(f"Question: {question}\n")

    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")

    # ==========================================================================
    # Example 2: Conversational RAG with History
    # ==========================================================================
    print_section("Example 2: Conversational RAG with History")

    print("Building conversational chain with memory...\n")

    # Contextualize question prompt
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # History-aware retriever
    history_aware_retriever = (
        contextualize_q_prompt
        | llm
        | StrOutputParser()
        | retriever
    )

    # QA prompt
    qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Keep the answer concise.

Context: {context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Build conversational chain
    conversational_chain = (
        RunnablePassthrough.assign(
            context=history_aware_retriever | format_docs
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # Simulate conversation
    chat_history = []

    # Turn 1
    question1 = "What is Python?"
    print(f"User: {question1}")
    answer1 = conversational_chain.invoke({
        "input": question1,
        "chat_history": chat_history
    })
    print(f"Assistant: {answer1}\n")

    chat_history.extend([
        HumanMessage(content=question1),
        AIMessage(content=answer1)
    ])

    # Turn 2 (references previous context)
    question2 = "What is it used for?"  # "it" refers to Python
    print(f"User: {question2}")
    answer2 = conversational_chain.invoke({
        "input": question2,
        "chat_history": chat_history
    })
    print(f"Assistant: {answer2}")

    # ==========================================================================
    # Example 3: Parallel Execution
    # ==========================================================================
    print_section("Example 3: Parallel Execution with RunnableParallel")

    print("Executing multiple retrievals in parallel...\n")

    # Create a chain that retrieves and generates summary in parallel
    parallel_chain = RunnableParallel(
        {
            "retrieval_result": itemgetter("question") | retriever | format_docs,
            "direct_answer": ChatPromptTemplate.from_template(
                "Answer this question directly: {question}"
            ) | llm | StrOutputParser(),
        }
    )

    question = "What is machine learning?"
    print(f"Question: {question}\n")

    result = parallel_chain.invoke({"question": question})

    print("Parallel results:")
    print(f"\n1. Retrieved context (from vector store):")
    print(f"   {result['retrieval_result'][:100]}...\n")
    print(f"2. Direct answer (from LLM):")
    print(f"   {result['direct_answer']}")

    # ==========================================================================
    # Example 4: Conditional Routing
    # ==========================================================================
    print_section("Example 4: Conditional Routing")

    print("Routing questions to different handlers...\n")

    def route_question(input_dict):
        """Route based on question type."""
        question = input_dict["question"].lower()

        if "python" in question or "programming" in question:
            return "programming"
        elif "machine learning" in question or "ai" in question:
            return "ai"
        else:
            return "general"

    # Different chains for different topics
    programming_chain = (
        ChatPromptTemplate.from_template(
            "As a programming expert, answer: {question}"
        )
        | llm
        | StrOutputParser()
    )

    ai_chain = (
        ChatPromptTemplate.from_template(
            "As an AI expert, answer: {question}"
        )
        | llm
        | StrOutputParser()
    )

    general_chain = rag_chain  # Use RAG for general questions

    # Routing logic
    def route_chain(input_dict):
        route = route_question(input_dict)
        print(f"   Routing to: {route}")

        if route == "programming":
            return programming_chain.invoke(input_dict["question"])
        elif route == "ai":
            return ai_chain.invoke(input_dict["question"])
        else:
            return general_chain.invoke(input_dict["question"])

    routing_chain = RunnableLambda(route_chain)

    # Test routing
    test_questions = [
        "How does Python handle memory management?",
        "Explain machine learning algorithms",
        "What is Docker used for?",
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        answer = routing_chain.invoke({"question": q})
        print(f"Answer: {answer[:100]}...")

    # ==========================================================================
    # Example 5: Chain Introspection
    # ==========================================================================
    print_section("Example 5: Chain Introspection and Debugging")

    print("LCEL chains are inspectable and debuggable...\n")

    # Get chain schema
    print("Chain schema (input/output types):")
    print(f"   Input schema: {rag_chain.input_schema.schema()}")
    print(f"   Output schema: {rag_chain.output_schema.schema()}\n")

    # Stream output (token by token)
    print("Streaming output (real-time generation):")
    print("Question: What is Docker?\n")
    print("Answer: ", end="", flush=True)

    for chunk in rag_chain.stream("What is Docker?"):
        print(chunk, end="", flush=True)
    print("\n")

    # Batch processing
    print("Batch processing multiple questions:")
    questions_batch = [
        "What is Python?",
        "What is Docker?",
        "What is REST API?",
    ]

    print(f"   Processing {len(questions_batch)} questions in batch...\n")
    answers = rag_chain.batch(questions_batch)

    for q, a in zip(questions_batch, answers):
        print(f"   Q: {q}")
        print(f"   A: {a[:80]}...\n")

    # ==========================================================================
    # Best Practices Summary
    # ==========================================================================
    print_section("LCEL Best Practices Summary")

    print("✅ Key LCEL advantages:")
    print("   • Clean, readable syntax with pipe (|) operator")
    print("   • Built-in streaming, async, and batch support")
    print("   • Automatic schema validation")
    print("   • Easy to test and debug")
    print("   • Composable and reusable components")
    print("")
    print("🎯 Common patterns:")
    print("   • RunnablePassthrough: Pass data through unchanged")
    print("   • RunnableParallel: Execute multiple steps in parallel")
    print("   • RunnableLambda: Custom Python functions")
    print("   • RunnableSequence: Chain multiple runnables")
    print("")
    print("💡 Tips:")
    print("   • Use .stream() for real-time output")
    print("   • Use .batch() for multiple inputs")
    print("   • Use .invoke() for single synchronous calls")
    print("   • Use .ainvoke() for async operations")
    print("")
    print("📚 Migration from old chains:")
    print("   • RetrievalQA → LCEL with retriever | prompt | llm")
    print("   • ConversationalRetrievalChain → LCEL with history")
    print("   • LLMChain → LCEL with prompt | llm")

    # Cleanup
    print_section("Cleanup")
    try:
        client.delete_table("lcel_demo")
        print("   ✓ Cleaned up demo table")
    except Exception as e:
        print(f"   ℹ Cleanup: {e}")

    print("\n" + "=" * 80)
    print("  Key Takeaway")
    print("=" * 80)
    print("""
LCEL is the modern, recommended way to build LangChain applications:
✓ More readable and maintainable
✓ Better performance with streaming and batching
✓ Easier to debug and test
✓ Future-proof (LangChain's direction)

Start using LCEL for all new projects!
    """)


if __name__ == "__main__":
    main()

"""
Agent with Vector Store Tool Example

This example demonstrates how to use DXMT vector store as a tool within
a LangChain Agent, enabling the LLM to decide when and how to search
the knowledge base.

Difference from RAG:
- RAG: Always retrieves, then answers
- Agent: Decides whether to retrieve, what to search, and how to combine sources

How it works:
1. Agent receives user question
2. Agent decides if it needs to search the knowledge base
3. Agent formulates search query (may differ from user question)
4. Agent uses tool to retrieve documents
5. Agent synthesizes answer from multiple sources

Use cases:
- Complex queries needing multi-step reasoning
- Questions requiring multiple searches
- Combining vector search with other tools (web search, calculator, etc.)
- Dynamic decision-making about when to use knowledge base
"""

import os
from typing import Optional, Type

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from embedding import default_embeddings, embed_dimension, needs_an_llm

# `langchain_openai` is imported where it is used, not here: without the
# package this example died at import time, before it could say what it needed.
from langchain_dxmt import DXMTVectorDB
from pydantic import BaseModel, Field

# Try multiple import paths for different langchain versions
try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
except ImportError:
    try:
        from langchain_community.agents import AgentExecutor
        from langchain_community.agent_toolkits import create_openai_tools_agent
    except ImportError:
        try:
            from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
        except ImportError:
            print("⚠️ Agent tools not available.")
            print("Tried import paths:")
            print("  - langchain.agents")
            print("  - langchain_community.agents")
            print("  - langchain_classic.agents")
            print("")
            print("This example requires langchain and agent support.")
            print("Install with: pip install langchain langchain-community")
            exit(1)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


# Define tool input schema
class VectorSearchInput(BaseModel):
    """Input schema for vector search tool."""
    query: str = Field(
        description="The search query to find relevant documents in the knowledge base"
    )
    k: int = Field(
        default=3,
        description="Number of results to return (default: 3)"
    )


class DXMTVectorSearchTool(BaseTool):
    """Tool for searching DXMT vector store."""

    name: str = "knowledge_base_search"
    description: str = """Search the company knowledge base for information about products, features, and documentation.
    Use this when you need specific facts that might be in our documentation.
    Input should be a clear search query."""
    args_schema: Type[BaseModel] = VectorSearchInput

    vectorstore: DXMTVectorDB = Field(exclude=True)

    def _run(self, query: str, k: int = 3) -> str:
        """Execute the search."""
        results = self.vectorstore.similarity_search(query, k=k)

        if not results:
            return "No relevant documents found."

        # Format results
        formatted = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[Source {i}: {source}]\n{doc.page_content}")

        return "\n\n".join(formatted)

    async def _arun(self, query: str, k: int = 3) -> str:
        """Async version."""
        # For simplicity, using sync version
        return self._run(query, k)


def main():
    # Check for OpenAI API key
    if not needs_an_llm("this example"):
        return

    from langchain_openai import ChatOpenAI

    print_section("Agent with Vector Store Tool")

    # Initialize components
    embeddings = default_embeddings()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Create vector store
    print("Setting up DXMT vector store...")
    vectorstore = DXMTVectorDB(
        table_name="agent_demo",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
    )

    # Clean up
    try:
        from dxmt import Client
        client = Client(host="localhost", port=9999)
        client.delete_table("agent_demo")
        print("   ✓ Cleaned existing table")
    except:
        pass

    # Recreate
    vectorstore = DXMTVectorDB(
        table_name="agent_demo",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
    )

    # Add knowledge base documents
    docs = [
        Document(
            page_content="DXMT VectorDB supports three index types: HNSW for balanced performance, IVF for large-scale datasets, and BruteForce for small datasets with exact results.",
            metadata={"source": "Product Documentation", "topic": "indexes"}
        ),
        Document(
            page_content="DXMT pricing: Starter plan at $99/month (up to 1M vectors), Pro plan at $299/month (up to 10M vectors), Enterprise plan with custom pricing for unlimited vectors.",
            metadata={"source": "Pricing Page", "topic": "pricing"}
        ),
        Document(
            page_content="Hybrid search in DXMT combines vector similarity with full-text search. Set vector_weight between 0-1, where 0.7 means 70% vector search and 30% text search. Enable full-text indexing when creating tables.",
            metadata={"source": "Feature Guide", "topic": "hybrid-search"}
        ),
        Document(
            page_content="DXMT installation: pip install dxmt, then start server with 'dxmt serve --port 9999'. Requires Python 3.8+. For production, use Docker: docker run -p 9999:9999 dxmt/server",
            metadata={"source": "Installation Guide", "topic": "installation"}
        ),
        Document(
            page_content="DXMT performance benchmarks: 3.5M queries per second on standard hardware, 5-10ms P95 latency, sub-5min recovery time from snapshots. Tiered storage reduces costs by 70%.",
            metadata={"source": "Performance Docs", "topic": "performance"}
        ),
        Document(
            page_content="DXMT supports multiple distance metrics: cosine (for normalized vectors), euclidean (for absolute distances), and dot product (for similarity scoring). Choose based on your embedding model.",
            metadata={"source": "Configuration Guide", "topic": "distance-metrics"}
        ),
    ]

    print("Adding documents to knowledge base...")
    vectorstore.add_documents(docs)
    print(f"   ✓ Added {len(docs)} documents\n")

    # Create vector search tool
    vector_tool = DXMTVectorSearchTool(vectorstore=vectorstore)

    # Could add more tools here
    tools = [vector_tool]

    # Create agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to a knowledge base about DXMT VectorDB.

When answering questions:
1. Decide if you need to search the knowledge base
2. If yes, formulate a clear search query
3. Use the search results to provide accurate answers
4. Cite your sources when using the knowledge base
5. If you don't need the knowledge base, answer from your general knowledge

Be concise and helpful."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # ==========================================================================
    # Example 1: Question requiring knowledge base
    # ==========================================================================
    print_section("Example 1: Question Requiring Knowledge Base")

    question1 = "What index types does DXMT support and when should I use each?"
    print(f"Question: {question1}\n")
    print("Agent thinking process:")
    print("-" * 80)

    result1 = agent_executor.invoke({"input": question1})

    print("-" * 80)
    print(f"\nFinal Answer:\n{result1['output']}")

    # ==========================================================================
    # Example 2: General knowledge question (no search needed)
    # ==========================================================================
    print_section("Example 2: General Knowledge Question")

    question2 = "What is a vector database?"
    print(f"Question: {question2}\n")
    print("Agent thinking process:")
    print("-" * 80)

    result2 = agent_executor.invoke({"input": question2})

    print("-" * 80)
    print(f"\nFinal Answer:\n{result2['output']}")

    # ==========================================================================
    # Example 3: Multi-step reasoning
    # ==========================================================================
    print_section("Example 3: Multi-Step Reasoning")

    question3 = "I want to deploy DXMT for a project with 5M vectors. What's the cost and performance I can expect?"
    print(f"Question: {question3}\n")
    print("Agent thinking process:")
    print("-" * 80)

    result3 = agent_executor.invoke({"input": question3})

    print("-" * 80)
    print(f"\nFinal Answer:\n{result3['output']}")

    # ==========================================================================
    # Example 4: Question requiring multiple searches
    # ==========================================================================
    print_section("Example 4: Multiple Searches")

    question4 = "Compare the installation process with the pricing plans. Which plan should a beginner choose?"
    print(f"Question: {question4}\n")
    print("Agent thinking process:")
    print("-" * 80)

    result4 = agent_executor.invoke({"input": question4})

    print("-" * 80)
    print(f"\nFinal Answer:\n{result4['output']}")

    # ==========================================================================
    # Example 5: Interactive mode
    # ==========================================================================
    print_section("Example 5: Interactive Agent Chat")

    print("You can now chat with the agent interactively.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Predefined demo questions for non-interactive mode
    demo_questions = [
        "What is hybrid search?",
        "How much does the Pro plan cost?",
    ]

    print("Demo mode - using predefined questions:\n")
    for q in demo_questions:
        print(f"User: {q}")
        response = agent_executor.invoke({"input": q})
        print(f"Agent: {response['output']}\n")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print_section("Agent vs RAG Comparison")

    print("📊 When to use Agent vs RAG:\n")
    print("Use Agent when:")
    print("   ✓ Need multi-step reasoning")
    print("   ✓ Combining multiple tools (search, calculator, API calls)")
    print("   ✓ Complex queries requiring decision-making")
    print("   ✓ Need to search knowledge base selectively")
    print("   ✓ Questions may not need retrieval at all")
    print("")
    print("Use RAG when:")
    print("   ✓ Every question needs document context")
    print("   ✓ Simple Q&A over documents")
    print("   ✓ Speed is critical (no agent reasoning overhead)")
    print("   ✓ Lower cost (fewer LLM calls)")
    print("   ✓ More predictable behavior")
    print("")
    print("Trade-offs:")
    print("   Agent: Smarter but slower and more expensive")
    print("   RAG: Faster and cheaper but less flexible")

    print_section("Advanced Agent Patterns")

    print("🔧 Enhancements you can add:\n")
    print("1. Multiple tools:")
    print("   • Vector search tool (knowledge base)")
    print("   • Web search tool (current info)")
    print("   • Calculator tool (math)")
    print("   • API tools (external data)")
    print("")
    print("2. Memory:")
    print("   • Add conversation history")
    print("   • Track what has been searched")
    print("   • Learn from user feedback")
    print("")
    print("3. Custom routing:")
    print("   • Different agents for different tasks")
    print("   • Supervisor agent coordinating sub-agents")
    print("   • Parallel agent execution")
    print("")
    print("4. Advanced retrieval:")
    print("   • Combine with multi-query retrieval")
    print("   • Use MMR for diverse results")
    print("   • Apply contextual compression")
    print("")
    print("5. Observability:")
    print("   • Log agent decisions")
    print("   • Track tool usage")
    print("   • Monitor costs per query")

    # Cleanup
    print_section("Cleanup")
    try:
        client.delete_table("agent_demo")
        print("   ✓ Cleaned up demo table")
    except Exception as e:
        print(f"   ℹ Cleanup: {e}")

    print("\n" + "=" * 80)
    print("  Key Takeaway")
    print("=" * 80)
    print("""
Agents with vector store tools enable:
✓ Intelligent decision-making about when to search
✓ Multi-step reasoning with knowledge base access
✓ Combining multiple information sources
✓ More flexible and powerful than simple RAG

Use when you need the LLM to intelligently decide how to use tools,
not just blindly retrieve every time.
    """)


if __name__ == "__main__":
    main()

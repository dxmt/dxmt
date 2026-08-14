# LangChain-DXMT Examples

Welcome! This directory contains comprehensive examples demonstrating how to use DXMT with LangChain for various AI applications.

## 📚 Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Example Categories](#example-categories)
  - [Basic Examples](#basic-examples)
  - [Advanced Use Cases](#advanced-use-cases)
  - [Production Examples](#production-examples)
- [Running Examples](#running-examples)
- [Troubleshooting](#troubleshooting)
- [Learning Path](#learning-path)

---

## Quick Start

```bash
# 1. Install dependencies
pip install langchain-dxmt langchain-openai langchain

# 2. Set your OpenAI API key
export OPENAI_API_KEY='sk-...'

# 3. Start DXMT server (in another terminal)
docker run -p 9999:9999 dxmt/server
# Or if installed locally:
dxmt serve

# 4. Run an example
python examples/basic_usage.py
```

---

## Prerequisites

### Required

1. **DXMT Server** running on `localhost:9999`
   ```bash
   # Option 1: Docker (recommended)
   docker run -p 9999:9999 dxmt/server

   # Option 2: Local installation
   pip install dxmt
   dxmt serve
   ```

2. **Python Packages**
   ```bash
   pip install langchain-dxmt langchain-openai langchain
   ```

3. **OpenAI API Key** (for most examples)
   ```bash
   export OPENAI_API_KEY='sk-your-key-here'
   ```
   Get your key from: https://platform.openai.com/api-keys

### Optional

- **Alternative Embeddings**: Some examples can be adapted to use local embeddings
- **Custom LLMs**: Replace OpenAI with other LangChain-supported LLMs

---

## Example Categories

### Basic Examples

Learn the fundamentals of using DXMT with LangChain.

#### 1. **basic_usage.py** - Getting Started
- ⏱️ **Time**: 5 minutes
- 🎯 **Level**: Beginner
- 📖 **Learn**: Core operations (add, search, retrieve, delete)

**What you'll learn:**
- Initialize DXMT vector store
- Add documents with metadata
- Perform similarity searches
- Filter by metadata
- Use as LangChain retriever
- Delete documents

**Run it:**
```bash
python examples/basic_usage.py
```

---

#### 2. **hybrid_search.py** - Combining Vector + Text Search
- ⏱️ **Time**: 5 minutes
- 🎯 **Level**: Beginner
- 📖 **Learn**: Hybrid search (semantic + keyword matching)

**What you'll learn:**
- Pure vector similarity search
- Hybrid search with adjustable weights
- When to use vector vs text vs hybrid
- Metadata filtering with hybrid search

**Run it:**
```bash
python examples/hybrid_search.py
```

---

#### 3. **async_usage.py** - High-Performance Async Operations
- ⏱️ **Time**: 5 minutes
- 🎯 **Level**: Intermediate
- 📖 **Learn**: Async/await patterns, concurrent operations

**What you'll learn:**
- Async document operations
- Concurrent batch processing
- Performance comparison (sync vs async)
- Best practices for async LangChain

**Run it:**
```bash
python examples/async_usage.py
```

---

### Advanced Use Cases

Real-world applications showing DXMT in production scenarios.

#### 4. **rag_qa_system.py** - RAG Question-Answering System
- ⏱️ **Time**: 10 minutes
- 🎯 **Level**: Intermediate
- 📖 **Learn**: Retrieval-Augmented Generation (RAG)

**What you'll learn:**
- Build a Q&A system over your documents
- Combine retrieval with LLM generation
- Return source documents for transparency
- Interactive Q&A mode

**Use cases:**
- Internal knowledge bases
- Product documentation Q&A
- Research paper analysis
- Customer support automation

**Run it:**
```bash
python examples/rag_qa_system.py
```

**Example queries:**
- "What is DXMT VectorDB and what index types does it support?"
- "How does hybrid search work in DXMT?"
- "How much does DXMT cost?"

---

#### 5. **chatbot_with_memory.py** - Conversational AI with Context
- ⏱️ **Time**: 10 minutes
- 🎯 **Level**: Intermediate
- 📖 **Learn**: Conversation memory, context-aware responses

**What you'll learn:**
- Maintain conversation history
- Context-aware responses (understands "it", "that")
- Retrieve relevant knowledge on-the-fly
- Natural follow-up questions

**Use cases:**
- Customer support chatbots
- Personal assistants
- Interactive documentation
- Multi-turn dialogues

**Run it:**
```bash
python examples/chatbot_with_memory.py
```

**Try it:**
The example includes an interactive mode where you can have natural conversations!

---

#### 6. **semantic_search_engine.py** - Intelligent Search
- ⏱️ **Time**: 10 minutes
- 🎯 **Level**: Intermediate
- 📖 **Learn**: Semantic understanding, metadata filtering

**What you'll learn:**
- Meaning-based search (not just keywords)
- Rank results by relevance
- Filter by metadata (category, author, date)
- Compare semantic vs keyword search

**Use cases:**
- Internal document search
- Content recommendation
- E-commerce product discovery
- Code search
- Academic paper exploration

**Run it:**
```bash
python examples/semantic_search_engine.py
```

**Example searches:**
- "How do I deploy containerized apps at scale?" → Finds Kubernetes article
- "best practices for coding" + filter by category
- Find all articles by specific author

---

#### 7. **document_summarization.py** - Smart Summarization
- ⏱️ **Time**: 10 minutes
- 🎯 **Level**: Advanced
- 📖 **Learn**: Document chunking, multiple summarization methods

**What you'll learn:**
- Split large documents into chunks
- Map-Reduce summarization
- Targeted summaries (query-focused)
- Custom prompts for structured output
- Extractive vs abstractive summarization

**Use cases:**
- Research paper summaries
- Legal document review
- Meeting transcript digests
- News article summaries
- Technical documentation overviews

**Run it:**
```bash
python examples/document_summarization.py
```

**Methods demonstrated:**
1. Full document summary (Map-Reduce)
2. Targeted summary (query-focused)
3. Structured summary (custom prompts)
4. Extractive summary (key quotes)

---

### Advanced Retrieval Techniques

Master modern LangChain retrieval patterns for production-grade applications.

#### 10. **mmr_diversity_search.py** - Maximal Marginal Relevance
- ⏱️ **Time**: 10 minutes
- 🎯 **Level**: Intermediate
- 📖 **Learn**: Diverse search results, avoiding redundancy

**What you'll learn:**
- Balance relevance and diversity with MMR
- Control lambda parameter for different use cases
- Avoid redundant/similar results
- Use cases for content recommendations
- Optimize for exploration vs precision

**When to use:**
- Content recommendations (avoid similar items)
- Research paper discovery (diverse perspectives)
- News aggregation (variety of sources)
- Product search (show different options)

**Run it:**
```bash
python examples/mmr_diversity_search.py
```

---

#### 11. **multi_query_retriever.py** - Multi-Query Retrieval
- ⏱️ **Time**: 10 minutes
- 🎯 **Level**: Intermediate
- 📖 **Learn**: Generate multiple query variations for better recall

**What you'll learn:**
- Improve retrieval quality with query variations
- LLM-generated alternative phrasings
- De-duplicate and rank combined results
- When multi-query beats single-query
- Cost-benefit analysis

**Benefits:**
- 15-25% better recall on complex queries
- More robust to query phrasing
- Better for ambiguous questions
- Critical for high-accuracy Q&A systems

**Run it:**
```bash
python examples/multi_query_retriever.py
```

**Note:** Works with LangChain 0.x, 1.0-1.1, and 1.2.6+

---

#### 12. **contextual_compression.py** - Reducing Retrieval Noise
- ⏱️ **Time**: 10 minutes
- 🎯 **Level**: Advanced
- 📖 **Learn**: Extract only relevant passages, reduce token usage

**What you'll learn:**
- Compress long documents to relevant passages
- 40-70% token reduction
- Improve answer quality by removing noise
- Balance compression cost vs generation savings
- Production optimization strategies

**Use cases:**
- Long documents with mixed content
- Cost optimization (reduce LLM tokens)
- Improving RAG answer focus
- Working with limited context windows

**Run it:**
```bash
python examples/contextual_compression.py
```

**Note:** Works with LangChain 0.x, 1.0-1.1, and 1.2.6+

---

#### 13. **lcel_chains.py** - Modern LCEL Syntax
- ⏱️ **Time**: 15 minutes
- 🎯 **Level**: Intermediate to Advanced
- 📖 **Learn**: LangChain Expression Language (LCEL) patterns

**What you'll learn:**
- Modern chain building with pipe (|) operator
- Streaming, batching, and async support
- Conversational RAG with history
- Parallel execution with RunnableParallel
- Conditional routing
- Chain introspection and debugging

**Why LCEL:**
- Cleaner, more readable syntax
- Built-in streaming and async
- Better performance
- Future-proof (LangChain's direction)

**Run it:**
```bash
python examples/lcel_chains.py
```

---

#### 14. **agent_with_vectorstore_tool.py** - Intelligent Agents
- ⏱️ **Time**: 15 minutes
- 🎯 **Level**: Advanced
- 📖 **Learn**: LLM agents with vector store tools

**What you'll learn:**
- Vector store as an agent tool
- Agent decides when to search knowledge base
- Multi-step reasoning with retrieval
- Combining multiple tools
- Agent vs RAG comparison

**Use cases:**
- Complex queries needing multi-step reasoning
- Combining multiple information sources
- Questions that may not need retrieval
- Dynamic tool selection

**Run it:**
```bash
python examples/agent_with_vectorstore_tool.py
```

---

### Production Examples

Learn advanced configurations and optimizations.

#### 8. **chinese_text_search.py** - Chinese Language Support
- ⏱️ **Time**: 5 minutes
- 🎯 **Level**: Intermediate
- 📖 **Learn**: Chinese text processing with Jieba

**What you'll learn:**
- Chinese document indexing
- Jieba tokenization
- Hybrid search for Chinese text
- Semantic vs keyword search in Chinese

**Run it:**
```bash
python examples/chinese_text_search.py
```

---

#### 9. **advanced_configuration.py** - Performance Tuning
- ⏱️ **Time**: 15 minutes
- 🎯 **Level**: Advanced
- 📖 **Learn**: Index types, distance metrics, insert modes

**What you'll learn:**
- Compare index types (HNSW, IVF, IVF-PQ, LSH)
- Compare distance metrics (cosine, euclidean, dot)
- Batch insert optimization strategies
- Production configurations
- Performance benchmarking

**Run it:**
```bash
python examples/advanced_configuration.py
```

---

## Running Examples

### Method 1: Run Individual Examples

```bash
cd /path/to/langchain-dxmt

# Make sure DXMT server is running
# Make sure OPENAI_API_KEY is set

python examples/basic_usage.py
python examples/rag_qa_system.py
python examples/chatbot_with_memory.py
# ... etc
```

### Method 2: Run All Examples

```bash
# Run all examples in sequence
for file in examples/*.py; do
    echo "Running $file..."
    python "$file"
    echo "---"
done
```

### Method 3: Interactive Mode

Some examples have interactive modes. Simply run them and follow the prompts:

```bash
python examples/chatbot_with_memory.py
# Type questions, get answers, see conversation history

python examples/semantic_search_engine.py
# Search interactively, explore results
```

---

## Troubleshooting

### Issue: "Connection refused" or "DXMT server not responding"

**Solution:** Make sure DXMT server is running:
```bash
# Check if server is running
curl http://localhost:9999/health

# If not, start it:
docker run -p 9999:9999 dxmt/server
```

---

### Issue: "OPENAI_API_KEY not set"

**Solution:** Export your OpenAI API key:
```bash
export OPENAI_API_KEY='sk-your-key-here'

# Or add to your ~/.bashrc or ~/.zshrc:
echo "export OPENAI_API_KEY='sk-your-key-here'" >> ~/.bashrc
source ~/.bashrc
```

---

### Issue: "Module not found: langchain_dxmt"

**Solution:** Install the package:
```bash
pip install langchain-dxmt langchain-openai langchain
```

---

### Issue: "Table already exists" error

**Solution:** Delete the existing table or use a different table name:
```bash
# Using Python
from dxmt import Client
client = Client(host="localhost", port=9999)
client.delete_table("your_table_name")
```

---

### Issue: Slow performance

**Solutions:**
1. Use `index_type="hnsw"` for fast queries (default)
2. Use `insert_mode="lock_free"` for fast inserts
3. Check DXMT server resources (CPU, memory)
4. Reduce `k` (number of results) in searches

---

## Learning Path

### For Beginners

**Goal:** Learn DXMT basics with LangChain

1. ✅ Start with `basic_usage.py` - Get familiar with core operations
2. ✅ Try `hybrid_search.py` - Understand different search methods
3. ✅ Explore `rag_qa_system.py` - Build your first RAG application
4. ✅ Experiment with `semantic_search_engine.py` - See semantic search in action

**Estimated time:** 30-40 minutes

---

### For Intermediate Users

**Goal:** Build production-ready applications

1. ✅ `chatbot_with_memory.py` - Learn conversation management
2. ✅ `document_summarization.py` - Handle large documents
3. ✅ `async_usage.py` - Optimize for performance
4. ✅ `advanced_configuration.py` - Tune for your workload

**Estimated time:** 45-60 minutes

---

### For Advanced Users

**Goal:** Master advanced retrieval patterns and optimization

1. ✅ **mmr_diversity_search.py** - Learn diversity search
2. ✅ **multi_query_retriever.py** - Improve retrieval quality (15-25% better recall)
3. ✅ **contextual_compression.py** - Optimize token usage (40-70% reduction)
4. ✅ **lcel_chains.py** - Modern LangChain patterns
5. ✅ **agent_with_vectorstore_tool.py** - Build intelligent agents
6. ✅ **advanced_configuration.py** - Tune performance
7. ✅ Adapt examples for your specific use case
8. ✅ Implement custom embeddings (local models)
9. ✅ Build multi-tenant applications
10. ✅ Add monitoring and observability

**Estimated time:** 1-2 hours for new examples, then custom based on your needs

---

## Example Feature Matrix

| Example | RAG | Memory | Hybrid | Async | Filters | Advanced |
|---------|-----|--------|--------|-------|---------|----------|
| basic_usage.py | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| hybrid_search.py | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| async_usage.py | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| rag_qa_system.py | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| chatbot_with_memory.py | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| semantic_search_engine.py | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| document_summarization.py | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| chinese_text_search.py | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| advanced_configuration.py | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **mmr_diversity_search.py** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ MMR |
| **multi_query_retriever.py** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ Multi-Query |
| **contextual_compression.py** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ Compression |
| **lcel_chains.py** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ LCEL |
| **agent_with_vectorstore_tool.py** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Agent |

**Note:** All examples work with LangChain 0.x, 1.0-1.1, and 1.2.6+ through fallback imports

---

## Common Patterns

### Pattern 1: Initialize Vector Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_dxmt import DXMTVectorDB

embeddings = OpenAIEmbeddings()

vectorstore = DXMTVectorDB(
    table_name="my_documents",
    embedding=embeddings,
    dimension=1536,  # OpenAI ada-002
    host="localhost",
    port=9999,
)
```

### Pattern 2: Add Documents

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content="Your text here",
        metadata={"source": "doc1.txt", "author": "Alice"}
    ),
    # ... more docs
]

ids = vectorstore.add_documents(docs)
```

### Pattern 3: Search

```python
# Simple search
results = vectorstore.similarity_search("query", k=5)

# With scores
results_with_scores = vectorstore.similarity_search_with_score("query", k=5)

# With filters
results_filtered = vectorstore.similarity_search(
    "query",
    k=5,
    filter={"author": "Alice"}
)

# Hybrid search (DXMT-specific)
results_hybrid = vectorstore.hybrid_search(
    "query",
    k=5,
    vector_weight=0.7  # 70% semantic, 30% keyword
)
```

### Pattern 4: Use as Retriever

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Use with LangChain chains
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)

result = qa_chain.invoke({"query": "Your question"})
```

---

## Next Steps

1. 📖 **Read the main README**: `../README.md`
2. 🧪 **Run the tests**: `pytest tests/`
3. 📚 **Check the API docs**: https://docs.dxmt.io
4. 💬 **Join the community**: https://github.com/dxmt/vectordb/discussions
5. 🐛 **Report issues**: https://github.com/dxmt/vectordb/issues

---

## Additional Resources

### Documentation
- [LangChain Documentation](https://python.langchain.com/docs/)
- [DXMT VectorDB Docs](https://docs.dxmt.io)
- [OpenAI API Reference](https://platform.openai.com/docs/)

### Tutorials
- [LangChain RAG Tutorial 2026](https://langchain-tutorials.github.io/)
- [Building LLM Applications](https://docs.langchain.com/oss/python/use_cases/)

### Community
- [LangChain Discord](https://discord.gg/langchain)
- [DXMT GitHub Discussions](https://github.com/dxmt/vectordb/discussions)

---

## Contributing

Found a bug in an example? Have an idea for a new example?

1. Open an issue: https://github.com/dxmt/vectordb/issues
2. Submit a PR: https://github.com/dxmt/vectordb/pulls

We especially welcome examples for:
- Domain-specific applications (healthcare, legal, finance)
- Integration with other LangChain tools
- Advanced optimization techniques
- Multi-lingual applications

---

## License

These examples are part of the langchain-dxmt project and are released under the Apache License 2.0.

---

**Happy building with LangChain + DXMT! 🚀**

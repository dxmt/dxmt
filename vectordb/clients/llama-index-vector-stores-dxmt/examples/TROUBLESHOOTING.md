# Examples Troubleshooting Guide

Common issues and solutions when running the DXMT LlamaIndex examples.

---

## Issue 1: 500 Internal Server Error

**Symptoms:**
```
httpx.HTTPStatusError: Server error '500 Internal Server Error'
for url 'http://localhost:9999/api/v1/tables/paul_graham_essay/search'
```

**Cause:**
- Table already exists with different configuration
- Index state mismatch
- Server-side error

**Solutions:**

### Solution 1: Use --clean flag (Recommended)
```bash
python paul_graham_essay.py --clean
python auto_retriever.py --clean
python crud_operations.py --clean
```

This deletes the existing table and starts fresh.

### Solution 2: Use different table name
```python
# In the example file, change the table name:
vector_store = DXMTVectorStore(
    table_name="paul_graham_essay_v2",  # Different name
    dimension=1536,
    host="localhost",
    port=9999,
)
```

### Solution 3: Manually delete table
```python
from dxmt import DXMTClient

client = DXMTClient(host="localhost", port=9999)
client.delete_table("paul_graham_essay")
```

Or using curl:
```bash
curl -X DELETE http://localhost:9999/api/v1/tables/paul_graham_essay
```

---

## Issue 2: Connection Refused

**Symptoms:**
```
Connection refused (localhost:9999)
```

**Cause:**
DXMT server is not running.

**Solution:**
Start the DXMT server:
```bash
cd /path/to/vectordb
cargo run -- serve
```

Or in a separate terminal:
```bash
# Terminal 1: Server
cargo run -- serve

# Terminal 2: Run examples
cd llama-index-vector-stores-dxmt/examples
python paul_graham_essay.py
```

---

## Issue 3: OpenAI API Key Not Set

**Symptoms:**
```
❌ Error: OPENAI_API_KEY environment variable not set
```

**Solution:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or add to your shell profile:
```bash
echo "export OPENAI_API_KEY='your-key'" >> ~/.bashrc
source ~/.bashrc
```

---

## Issue 4: Module Not Found

**Symptoms:**
```
ModuleNotFoundError: No module named 'llama_index'
```

**Solution:**
Install required packages:
```bash
pip install llama-index-core
pip install llama-index-vector-stores-dxmt
pip install llama-index-embeddings-openai
pip install llama-index-llms-openai
```

Or use the setup script:
```bash
cd examples
./setup_examples.sh
```

---

## Issue 5: Dimension Mismatch

**Symptoms:**
```
ValueError: Dimension mismatch: expected 1536, got 384
```

**Cause:**
Using different embedding model than specified in vector store configuration.

**Solution:**
Match the dimension to your embedding model:

```python
# For text-embedding-ada-002 (OpenAI default)
dimension=1536

# For sentence-transformers models
dimension=384  # or 768

# For text-embedding-3-small
dimension=512

# For text-embedding-3-large
dimension=3072
```

---

## Issue 6: Examples Run Slowly

**Symptoms:**
- Long wait times
- Slow queries

**Solutions:**

### 1. Check server resources
```bash
cargo run -- serve
```

### 2. Use smaller dataset
```python
# In examples, reduce the amount of data
documents = documents[:10]  # Only first 10 docs
```

### 3. Use faster embedding model
```python
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"  # Faster than ada-002
)
```

### 4. Reduce top_k
```python
query_engine = index.as_query_engine(
    similarity_top_k=3,  # Reduce from default
)
```

---

## Issue 7: RAGAS Evaluation Returns NaN

**Symptoms:**
```
answer_relevancy         : nan ⚠
answer_correctness       : nan ⚠
```

**Solution:**
Install langchain-openai:
```bash
pip install langchain-openai
```

---

## Issue 8: Table Already Exists

**Symptoms:**
```
Table 'my_table' already exists
```

**Solutions:**

### Option 1: Use existing table
```python
# Don't create new table, use existing one
index = VectorStoreIndex.from_vector_store(vector_store)
# Instead of:
# index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
```

### Option 2: Delete existing table
```bash
# Use --clean flag
python paul_graham_essay.py --clean
```

### Option 3: Use different table name
```python
vector_store = DXMTVectorStore(
    table_name="my_table_v2",  # Different name
    # ... other params
)
```

---

## Issue 9: Port Already in Use

**Symptoms:**
```
Address already in use (port 9999)
```

**Solution:**

### Find what's using the port
```bash
lsof -i :9999
```

### Kill the process
```bash
kill -9 <PID>
```

### Or use a different port
```bash
# Start server on different port
cargo run -- serve --port 9998

# Update examples
vector_store = DXMTVectorStore(
    table_name="my_table",
    dimension=1536,
    port=9998,  # Match server port
)
```

---

## Issue 10: Delete Operations Not Working

**Observation:**
Documents are not being deleted, or deletions don't seem to take effect.

**Cause:**
- Delete API may have limitations in current version
- Incorrect ref_doc_id being used
- Index not updating after deletion

**Current Status:**
Delete operations are being improved in DXMT. The CRUD example demonstrates
the intended usage pattern, but actual deletion may be limited.

**Workaround:**
Use soft deletes instead:
```python
# Instead of deleting
# index.delete_ref_doc("doc_id")

# Use soft delete (mark as deleted)
doc = Document(
    text=original_text,
    metadata={**original_metadata, "deleted": True, "deleted_at": "2024-01-24"},
    doc_id=original_id,
)
index.insert(doc)  # Upsert

# Then filter out deleted documents
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="deleted", value=True, operator=FilterOperator.NE)
    ]
)
retriever = index.as_retriever(filters=filters)
```

---

## Issue 11: Data Not Persisting

**Observation:**
Data disappears after restarting examples.

**Cause:**
This shouldn't happen with DXMT! Server persists data automatically.

**Debug steps:**

1. Check server is still running:
   ```bash
   curl http://localhost:9999/health
   ```

2. List tables:
   ```python
   from dxmt import DXMTClient
   client = DXMTClient(host="localhost", port=9999)
   tables = client.list_tables()
   print(tables)
   ```

3. Check table contents:
   ```python
   # Table definition and record count
   print(client.get_table("your_table_name"))
   print(client.count("your_table_name"))
   ```

---

## Debug Mode

Enable debug logging to see more details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:
```bash
export DXMT_LOG_LEVEL=debug
python paul_graham_essay.py
```

---

## Getting More Help

1. **Check server logs**: The DXMT server prints detailed logs
2. **GitHub Issues**: https://github.com/dxmt/dxmt/issues
3. **Examples README**: See `README.md` in this directory
4. **Documentation**: Check individual example files for detailed docstrings

---

## Quick Checklist

When running examples, make sure:

- [ ] DXMT server is running
- [ ] OPENAI_API_KEY is set
- [ ] Required packages are installed
- [ ] Not using conflicting table names
- [ ] Server has enough resources (memory)
- [ ] Port 9999 is available (or using custom port)
- [ ] Python version is 3.9+

---

## Common Command Reference

```bash
# Start server
cargo run -- serve

# Set API key
export OPENAI_API_KEY='your-key'

# Install dependencies
pip install llama-index-embeddings-openai llama-index-llms-openai

# Run example with clean start
python paul_graham_essay.py --clean

# Check server health
curl http://localhost:9999/health

# List tables
curl http://localhost:9999/api/v1/tables

# Delete table
curl -X DELETE http://localhost:9999/api/v1/tables/table_name
```

---

## Still Having Issues?

If none of the above helps:

1. **Minimal reproduction**: Create a minimal example that reproduces the issue
2. **Collect information**:
   - Server logs
   - Python version
   - Package versions (`pip list`)
   - Full error traceback
3. **Report issue**: https://github.com/dxmt/dxmt/issues

Include:
- What you tried
- What you expected
- What actually happened
- Error messages and logs

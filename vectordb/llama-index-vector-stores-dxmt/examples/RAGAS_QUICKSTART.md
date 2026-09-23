# RAGAS Evaluation Quick Start Guide

This guide helps you get started with evaluating your RAG system using the RAGAS framework and DXMT.

## What is RAGAS?

RAGAS (Retrieval Augmented Generation Assessment) is a framework for evaluating RAG pipelines. It provides automated metrics to assess both retrieval quality and generation quality.

## Why Evaluate RAG Systems?

Without evaluation, you can't:
- Know if your RAG system is performing well
- Compare different configurations
- Identify bottlenecks (retrieval vs generation)
- Optimize systematically
- Monitor quality in production

## Quick Start

### 1. Install Dependencies

```bash
# Option A: Use the setup script
cd examples
./setup_examples.sh

# Option B: Manual installation
pip install ragas datasets llama-index-embeddings-openai llama-index-llms-openai langchain-openai
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 3. Start DXMT Server

```bash
# In the vectordb directory
cd ../../
cargo run -- serve --simple --memory-limit 1024
```

### 4. Run the Evaluation

```bash
cd llama-index-vector-stores-dxmt/examples
python ragas_evaluation.py
```

## Understanding RAGAS Metrics

### 1. Faithfulness (0.0 - 1.0)

**What it measures:** Is the answer grounded in the retrieved context?

**Interpretation:**
- **1.0**: Perfect - all claims are supported by context
- **0.8-0.9**: Good - minor unsupported claims
- **<0.7**: Poor - answer has hallucinations

**How to improve:**
- Lower LLM temperature (0.0-0.2)
- Use stricter system prompts
- Implement fact-checking mechanisms
- Add citation requirements

### 2. Answer Relevancy (0.0 - 1.0)

**What it measures:** How relevant is the answer to the question?

**Interpretation:**
- **1.0**: Perfect - directly answers the question
- **0.8-0.9**: Good - mostly relevant
- **<0.7**: Poor - answer is off-topic

**How to improve:**
- Improve query understanding
- Use query rewriting
- Better retrieval (retrieve more relevant docs)
- Fine-tune LLM prompts

### 3. Context Precision (0.0 - 1.0)

**What it measures:** Are relevant contexts ranked higher than irrelevant ones?

**Interpretation:**
- **1.0**: Perfect - all top results are relevant
- **0.8-0.9**: Good - most top results relevant
- **<0.7**: Poor - many irrelevant results in top-k

**How to improve:**
- Use reranking (e.g., cross-encoder)
- Better embeddings (domain-specific)
- Hybrid search (vector + keyword)
- Optimize distance metric

### 4. Context Recall (0.0 - 1.0)

**What it measures:** Did we retrieve all relevant information?

**Interpretation:**
- **1.0**: Perfect - all needed info retrieved
- **0.8-0.9**: Good - most info retrieved
- **<0.7**: Poor - missing important context

**How to improve:**
- Increase `similarity_top_k`
- Better chunking strategy
- Improve embeddings
- Use query expansion

### 5. Answer Correctness (0.0 - 1.0)

**What it measures:** Factual accuracy compared to ground truth

**Interpretation:**
- **1.0**: Perfect match with ground truth
- **0.8-0.9**: Mostly correct
- **<0.7**: Significant errors

**How to improve:**
- Better retrieval (improve context)
- Fine-tune prompts
- Use stronger LLM
- Add verification steps

### 6. Answer Similarity (0.0 - 1.0)

**What it measures:** Semantic similarity to ground truth answer

**Interpretation:**
- **1.0**: Identical meaning
- **0.8-0.9**: Very similar
- **<0.7**: Different meaning

**How to improve:**
- Analyze ground truth patterns
- Adjust generation style
- Improve context selection

## Example Output Interpretation

```
📊 Overall Scores:
   faithfulness             : 0.9234 ✓  # Excellent - minimal hallucinations
   answer_relevancy         : 0.8876 ✓  # Good - answers are on-topic
   context_precision        : 0.8456 ✓  # Good - relevant docs ranked high
   context_recall           : 0.7654 ✓  # Acceptable - some info missing
   answer_correctness       : 0.8123 ✓  # Good - mostly accurate
   answer_similarity        : 0.7891 ✓  # Acceptable - similar to ground truth
```

**Analysis:**
- ✅ **Strengths**: Faithfulness (0.92) - Low hallucination rate
- ⚠️ **Needs work**: Context Recall (0.76) - Missing some relevant info
- **Action**: Increase `similarity_top_k` from 3 to 5

## Optimization Workflow

### Step 1: Establish Baseline

```bash
python ragas_evaluation.py
```

Note your baseline scores.

### Step 2: Identify Weak Points

Look for scores < 0.7. Common patterns:

**Low Context Recall?**
→ Retrieval problem - increase top_k, improve chunking

**Low Context Precision?**
→ Ranking problem - add reranking, use hybrid search

**Low Faithfulness?**
→ Generation problem - stricter prompts, lower temperature

**Low Answer Relevancy?**
→ Understanding problem - query rewriting, better prompts

### Step 3: Experiment with Configuration

```python
# Example: Improve context recall
query_engine = index.as_query_engine(
    similarity_top_k=5,  # Increased from 3
    response_mode="tree_summarize",  # Better synthesis
)
```

### Step 4: Re-evaluate

```bash
python ragas_evaluation.py
```

Compare new scores with baseline.

### Step 5: Iterate

Repeat steps 2-4 until all metrics > 0.8.

## Common Optimization Strategies

### Strategy 1: Increase Context

```python
# Before
similarity_top_k=3

# After
similarity_top_k=5
```

**Impact:** Improves context_recall, may reduce precision

### Strategy 2: Add Reranking

```python
from llama_index.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    top_n=3
)

query_engine = index.as_query_engine(
    similarity_top_k=10,  # Retrieve more
    node_postprocessors=[reranker]  # Rerank top 3
)
```

**Impact:** Improves context_precision dramatically

### Strategy 3: Lower Temperature

```python
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0  # More deterministic
)
```

**Impact:** Improves faithfulness

### Strategy 4: Better Chunking

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,     # Increased from 256
    chunk_overlap=100,  # More overlap
)
```

**Impact:** Improves context_recall and answer_correctness

### Strategy 5: Hybrid Search

```python
# Combine vector + keyword search
# See hybrid_search.py example
```

**Impact:** Improves both precision and recall

## Production Monitoring

### Create a Test Suite

```python
# test_suite.py
evaluation_questions = [
    # Domain-specific questions
    "What is the refund policy?",
    "How do I reset my password?",
    # ... more questions
]

ground_truths = [
    "Refunds are available within 30 days...",
    "Click 'Forgot Password' and follow...",
    # ... corresponding answers
]
```

### Automate Evaluation

```bash
# Run daily or on each deployment
python ragas_evaluation.py > eval_results_$(date +%Y%m%d).log
```

### Set Alerts

```python
# In your evaluation script
if faithfulness_score < 0.8:
    send_alert("Faithfulness score dropped below 0.8!")
```

### Track Trends

```python
# Save scores to database
import sqlite3

conn = sqlite3.connect("ragas_metrics.db")
conn.execute("""
    INSERT INTO metrics (date, faithfulness, answer_relevancy, ...)
    VALUES (?, ?, ?, ...)
""", (datetime.now(), faithfulness, answer_relevancy, ...))
```

## Advanced Topics

### Custom Metrics

```python
from ragas.metrics import Metric

class CustomMetric(Metric):
    def score(self, row):
        # Your custom scoring logic
        return score
```

### Domain-Specific Ground Truth

```python
# For your domain, create realistic QA pairs
qa_pairs = [
    {
        "question": "What are the ingredients in Product X?",
        "ground_truth": "Product X contains: ...",
        "contexts": ["...relevant context..."]
    },
    # ... more pairs
]
```

### Synthetic Data Generation

```python
# Use LLM to generate evaluation questions
from llama_index.core.evaluation import DatasetGenerator

generator = DatasetGenerator(...)
eval_questions = generator.generate_questions_from_nodes()
```

## Troubleshooting

### Issue: All scores are very low

**Cause:** Ground truth doesn't match your RAG output style

**Solution:** Review and adjust ground truth to be realistic

### Issue: Evaluation is slow

**Cause:** RAGAS makes LLM calls for each metric

**Solution:**
- Use smaller test set for iteration
- Use faster LLM (gpt-3.5-turbo instead of gpt-4)
- Cache embeddings

### Issue: Inconsistent scores

**Cause:** LLM temperature > 0

**Solution:** Set temperature=0 for deterministic evaluation

## Best Practices

1. ✅ **Start small**: Test with 5-10 questions first
2. ✅ **Realistic ground truth**: Match your domain and style
3. ✅ **Track baselines**: Always compare to baseline
4. ✅ **One change at a time**: Isolate impact of changes
5. ✅ **Monitor in production**: Continuous evaluation
6. ✅ **User feedback**: Complement metrics with real users

## Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [Evaluation Strategies Blog Post](https://www.llamaindex.ai/blog/evaluating-rag)
- [DXMT Documentation](https://github.com/dxmt/dxmt)

## Next Steps

1. Run the example: `python ragas_evaluation.py`
2. Create your own evaluation dataset
3. Experiment with different configurations
4. Set up production monitoring
5. Iterate based on metrics

Happy evaluating! 🚀

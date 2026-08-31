"""Compare OpenAI vs SentenceTransformer embeddings for abbreviation understanding.

This script demonstrates the difference between OpenAI and open-source models
in understanding Chinese-English mixed queries and abbreviations.

Prerequisites:
    pip install openai sentence-transformers
    export OPENAI_API_KEY="your-api-key"

Run:
    python3 compare_embeddings.py
"""

import os
import numpy as np
from typing import List, Tuple

# Check dependencies
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not available. Install: pip install openai")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("⚠️  SentenceTransformer not available. Install: pip install sentence-transformers")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_openai_embeddings():
    """Test OpenAI embeddings understanding of abbreviations."""
    if not OPENAI_AVAILABLE:
        print("\n❌ Skipping OpenAI test (not installed)")
        return None

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ Skipping OpenAI test (OPENAI_API_KEY not set)")
        return None

    print("\n" + "=" * 80)
    print("测试 OpenAI Embeddings (text-embedding-3-small)")
    print("=" * 80)

    client = OpenAI()

    test_cases = [
        ("NLP技术", "自然语言处理技术", "NLP vs 自然语言处理"),
        ("AI应用", "人工智能应用", "AI vs 人工智能"),
        ("ML模型", "机器学习模型", "ML vs 机器学习"),
        ("NLP技术", "HNSW算法", "NLP vs 不相关术语"),
    ]

    results = []

    for query1, query2, description in test_cases:
        # Get embeddings
        response = client.embeddings.create(
            input=[query1, query2],
            model="text-embedding-3-small"
        )

        vec1 = response.data[0].embedding
        vec2 = response.data[1].embedding

        similarity = cosine_similarity(vec1, vec2)
        results.append((description, similarity))

        # Evaluate
        if "vs 不相关" in description:
            status = "✅" if similarity < 0.5 else "⚠️"
            expected = "(应该低)"
        else:
            status = "✅" if similarity > 0.7 else "⚠️"
            expected = "(应该高)"

        print(f"{status} {description}: {similarity:.4f} {expected}")

    return results


def test_sentence_transformer():
    """Test SentenceTransformer understanding of abbreviations."""
    if not ST_AVAILABLE:
        print("\n❌ Skipping SentenceTransformer test (not installed)")
        return None

    print("\n" + "=" * 80)
    print("测试 SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2)")
    print("=" * 80)

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    test_cases = [
        ("NLP技术", "自然语言处理技术", "NLP vs 自然语言处理"),
        ("AI应用", "人工智能应用", "AI vs 人工智能"),
        ("ML模型", "机器学习模型", "ML vs 机器学习"),
        ("NLP技术", "HNSW算法", "NLP vs 不相关术语"),
    ]

    results = []

    for query1, query2, description in test_cases:
        vec1 = model.encode(query1, convert_to_numpy=True)
        vec2 = model.encode(query2, convert_to_numpy=True)

        similarity = cosine_similarity(vec1.tolist(), vec2.tolist())
        results.append((description, similarity))

        # Evaluate
        if "vs 不相关" in description:
            status = "✅" if similarity < 0.5 else "⚠️"
            expected = "(应该低)"
        else:
            status = "✅" if similarity > 0.7 else "⚠️"
            expected = "(应该高)"

        print(f"{status} {description}: {similarity:.4f} {expected}")

    return results


def compare_results(openai_results: List[Tuple], st_results: List[Tuple]):
    """Compare results from both models."""
    if not openai_results or not st_results:
        return

    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)

    print("\n{:<30} {:>15} {:>15} {:>10}".format(
        "测试场景", "OpenAI", "SentenceT", "差异"
    ))
    print("-" * 80)

    improvements = []

    for (desc_openai, sim_openai), (desc_st, sim_st) in zip(openai_results, st_results):
        diff = sim_openai - sim_st
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"

        # Determine if improvement is significant
        if "vs 不相关" not in desc_openai:
            # For related pairs, higher is better
            improvement = diff > 0.1
        else:
            # For unrelated pairs, lower is better (so negative diff is good)
            improvement = diff < -0.1

        symbol = "🎯" if improvement else "  "

        print(f"{symbol} {desc_openai:<30} {sim_openai:>15.4f} {sim_st:>15.4f} {diff_str:>10}")

        if improvement:
            improvements.append(desc_openai)

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)

    if improvements:
        print(f"\n✅ OpenAI在以下场景表现更好:")
        for desc in improvements:
            print(f"   • {desc}")

        print(f"\n💡 建议:")
        print(f"   • 对于中英混合查询和技术缩写，OpenAI显著优于开源模型")
        print(f"   • 如果预算允许，推荐使用OpenAI embeddings")
        print(f"   • 成本非常低：$0.02 / 1M tokens（约500万个汉字）")
    else:
        print("\n⚠️  两个模型表现相当")

    print("\n" + "=" * 80)
    print("性价比分析")
    print("=" * 80)

    print("""
| 指标 | SentenceTransformer | OpenAI |
|------|---------------------|--------|
| 成本 | 免费 | $0.02/1M tokens (~$0.004/1000查询) |
| 缩写理解 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 离线使用 | ✅ 可以 | ❌ 需要网络 |
| 首次启动 | 慢（下载模型） | 快（无需下载） |
| 查询延迟 | ~10ms | ~50-100ms |

推荐方案:
• 生产环境 → OpenAI (质量高，成本可忽略)
• 离线环境 → SentenceTransformer (唯一选择)
• 预算极度紧张 → SentenceTransformer
• 技术文档/缩写多 → OpenAI (显著提升)
    """)


def main():
    """Run comparison tests."""
    print("=" * 80)
    print("OpenAI vs SentenceTransformer 对比测试")
    print("=" * 80)
    print("\n测试目标: 评估模型对中英缩写的理解能力")
    print("场景: NLP↔自然语言处理, AI↔人工智能, ML↔机器学习\n")

    # Test both models
    openai_results = test_openai_embeddings()
    st_results = test_sentence_transformer()

    # Compare
    if openai_results and st_results:
        compare_results(openai_results, st_results)
    else:
        print("\n⚠️  无法完成对比（缺少必要的库或配置）")
        print("\n安装方法:")
        if not OPENAI_AVAILABLE:
            print("  pip install openai")
        if not ST_AVAILABLE:
            print("  pip install sentence-transformers")
        if OPENAI_AVAILABLE and not os.environ.get("OPENAI_API_KEY"):
            print("  export OPENAI_API_KEY='your-api-key'")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

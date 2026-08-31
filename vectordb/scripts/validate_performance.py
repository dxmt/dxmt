#!/usr/bin/env python3
"""
Performance Threshold Validation Script

Validates benchmark results against defined performance thresholds.
Used in CI to automatically fail builds if performance degrades.

Usage:
    python scripts/validate_performance.py benchmark_results.json

Exit codes:
    0: All thresholds passed
    1: One or more thresholds failed
    2: Invalid input or error
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Default thresholds (sync with src/benchmarks/thresholds.rs)
DEFAULT_THRESHOLDS = {
    "HNSW": {
        "min_recall_at_k": 0.95,
        "recall_k": 10,
        "min_qps": 10000.0,
        "max_p99_latency_ms": 10.0,
        "min_precision_at_k": 0.95,
        "min_ndcg_at_k": 0.95,
    },
    "DiskANN": {
        "min_recall_at_k": 0.90,
        "recall_k": 10,
        "min_qps": 5000.0,
        "max_p99_latency_ms": 50.0,
        "min_precision_at_k": 0.90,
        "min_ndcg_at_k": 0.90,
    },
    "IVF-PQ": {
        "min_recall_at_k": 0.85,
        "recall_k": 10,
        "min_qps": 8000.0,
        "max_p99_latency_ms": 20.0,
        "min_precision_at_k": 0.85,
        "min_ndcg_at_k": 0.85,
    },
    "HNSW-PQ": {
        "min_recall_at_k": 0.90,
        "recall_k": 10,
        "min_qps": 12000.0,
        "max_p99_latency_ms": 15.0,
        "min_precision_at_k": 0.90,
        "min_ndcg_at_k": 0.90,
    },
    "BruteForce": {
        "min_recall_at_k": 1.0,
        "recall_k": 10,
        "min_qps": 100.0,
        "max_p99_latency_ms": 1000.0,
        "min_precision_at_k": 1.0,
        "min_ndcg_at_k": 1.0,
    },
}


class ThresholdValidator:
    """Validates benchmark metrics against performance thresholds"""

    def __init__(self, thresholds: Dict = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def validate(self, index_type: str, metrics: Dict) -> Dict:
        """
        Validate metrics against thresholds for an index type

        Returns:
            {
                "passed": bool,
                "index_type": str,
                "failures": List[str],
                "metrics": Dict
            }
        """
        if index_type not in self.thresholds:
            return {
                "passed": False,
                "index_type": index_type,
                "failures": [f"No threshold defined for {index_type}"],
                "metrics": metrics,
            }

        threshold = self.thresholds[index_type]
        failures = []

        # Validate recall
        if metrics.get("recall_at_k", 0) < threshold["min_recall_at_k"]:
            failures.append(
                f"Recall@{threshold['recall_k']} "
                f"{metrics.get('recall_at_k', 0):.4f} < "
                f"threshold {threshold['min_recall_at_k']:.4f}"
            )

        # Validate QPS
        if metrics.get("qps", 0) < threshold["min_qps"]:
            failures.append(
                f"QPS {metrics.get('qps', 0):.0f} < "
                f"threshold {threshold['min_qps']:.0f}"
            )

        # Validate latency
        if metrics.get("p99_latency_ms", float("inf")) > threshold["max_p99_latency_ms"]:
            failures.append(
                f"P99 latency {metrics.get('p99_latency_ms', 0):.2f}ms > "
                f"threshold {threshold['max_p99_latency_ms']:.2f}ms"
            )

        # Validate precision (if available)
        if "precision_at_k" in metrics and "min_precision_at_k" in threshold:
            if metrics["precision_at_k"] < threshold["min_precision_at_k"]:
                failures.append(
                    f"Precision@{threshold['recall_k']} "
                    f"{metrics['precision_at_k']:.4f} < "
                    f"threshold {threshold['min_precision_at_k']:.4f}"
                )

        # Validate NDCG (if available)
        if "ndcg_at_k" in metrics and "min_ndcg_at_k" in threshold:
            if metrics["ndcg_at_k"] < threshold["min_ndcg_at_k"]:
                failures.append(
                    f"NDCG@{threshold['recall_k']} "
                    f"{metrics['ndcg_at_k']:.4f} < "
                    f"threshold {threshold['min_ndcg_at_k']:.4f}"
                )

        return {
            "passed": len(failures) == 0,
            "index_type": index_type,
            "failures": failures,
            "metrics": metrics,
        }

    def validate_batch(self, results: List[Dict]) -> List[Dict]:
        """Validate multiple benchmark results"""
        return [
            self.validate(result["index_type"], result["metrics"])
            for result in results
        ]


def print_validation_report(validation: Dict):
    """Print a validation report"""
    status = "✅ PASSED" if validation["passed"] else "❌ FAILED"

    print(f"\n=== Threshold Validation: {validation['index_type']} ===")
    print(f"Status: {status}")

    print("\nMetrics:")
    metrics = validation["metrics"]
    print(f"  Recall@10: {metrics.get('recall_at_k', 0):.4f}")
    print(f"  QPS: {metrics.get('qps', 0):.0f}")
    print(f"  P99 Latency: {metrics.get('p99_latency_ms', 0):.2f}ms")

    if "precision_at_k" in metrics:
        print(f"  Precision@10: {metrics['precision_at_k']:.4f}")

    if "ndcg_at_k" in metrics:
        print(f"  NDCG@10: {metrics['ndcg_at_k']:.4f}")

    if validation["failures"]:
        print("\nFailures:")
        for failure in validation["failures"]:
            print(f"  ❌ {failure}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_performance.py <benchmark_results.json>")
        sys.exit(2)

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(2)

    try:
        with open(input_file) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        sys.exit(2)

    # Load custom thresholds if specified
    thresholds = data.get("thresholds", DEFAULT_THRESHOLDS)
    validator = ThresholdValidator(thresholds)

    # Validate all results
    results = data.get("results", [])
    if not results:
        print("Error: No benchmark results found in input file")
        sys.exit(2)

    validations = validator.validate_batch(results)

    # Print reports
    print("\n" + "=" * 60)
    print("Performance Threshold Validation Report")
    print("=" * 60)

    for validation in validations:
        print_validation_report(validation)

    # Summary
    passed = sum(1 for v in validations if v["passed"])
    total = len(validations)

    print("\n" + "=" * 60)
    print(f"Summary: {passed}/{total} index types passed thresholds")
    print("=" * 60)

    # Exit with appropriate code
    if passed == total:
        print("\n✅ All performance thresholds passed!")
        sys.exit(0)
    else:
        print("\n❌ Some performance thresholds failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

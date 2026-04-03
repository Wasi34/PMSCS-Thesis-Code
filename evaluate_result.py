"""
Models:
  - gpt4o
  - gemini25pro
  - llama3

Usage:
    python evaluate_result.py
    python evaluate_result.py --model gpt4o
    python evaluate_result.py --model gemini25pro
    python evaluate_result.py --model llama3
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


MODELS = ["gpt4o", "gemini25pro", "llama3"]
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "outputs")


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word tokens."""
    return re.findall(r"\w+", normalize(text))


def exact_match(predicted: str, expected: str) -> bool:
    """Check if predicted answer exactly matches expected answer."""
    return normalize(predicted) == normalize(expected)


def partial_match(predicted: str, expected: str) -> bool:
    """Check if one normalized answer contains the other."""
    pred_norm = normalize(predicted)
    exp_norm = normalize(expected)
    return exp_norm in pred_norm or pred_norm in exp_norm


def precision_recall_f1(predicted: str, expected: str) -> Tuple[float, float, float]:
    """
    Compute token-level precision, recall, and F1.

    Precision = overlap / predicted_tokens
    Recall    = overlap / expected_tokens
    F1        = 2PR / (P + R)
    """
    pred_tokens = tokenize(predicted)
    exp_tokens = tokenize(expected)

    if not exp_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0
    if not exp_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    exp_counter = Counter(exp_tokens)

    overlap = pred_counter & exp_counter
    num_same = sum(overlap.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(exp_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate_model(model_name: str) -> Optional[Dict]:
    """Load results JSON and compute metrics for one model."""
    results_path = os.path.join(OUTPUT_BASE, model_name, "results.json")

    if not os.path.exists(results_path):
        print(f"  No results found for '{model_name}' at: {results_path}")
        return None

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)
    evaluable = [r for r in results if str(r.get("expected_answer", "")).strip()]
    skipped = total - len(evaluable)

    em_correct = 0
    partial_correct = 0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for row in evaluable:
        predicted = row.get("predicted_answer", "")
        expected = row.get("expected_answer", "")

        if exact_match(predicted, expected):
            em_correct += 1

        if partial_match(predicted, expected):
            partial_correct += 1

        precision, recall, f1 = precision_recall_f1(predicted, expected)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    n = len(evaluable)

    accuracy = (em_correct / n * 100) if n else 0.0
    partial_accuracy = (partial_correct / n * 100) if n else 0.0
    avg_precision = (precision_sum / n * 100) if n else 0.0
    avg_recall = (recall_sum / n * 100) if n else 0.0
    avg_f1 = (f1_sum / n * 100) if n else 0.0

    return {
        "model": model_name,
        "total_rows": total,
        "evaluable_rows": n,
        "skipped_rows": skipped,
        "exact_match_count": em_correct,
        "accuracy": round(accuracy, 2),
        "partial_match_count": partial_correct,
        "partial_match_accuracy": round(partial_accuracy, 2),
        "precision": round(avg_precision, 2),
        "recall": round(avg_recall, 2),
        "f1_score": round(avg_f1, 2),
    }


def print_report(metrics: Dict) -> None:
    """Pretty-print metrics for one model."""
    print(f"\n{'=' * 60}")
    print(f"Model: {metrics['model']}")
    print(f"{'=' * 60}")
    print(f"Total rows:              {metrics['total_rows']}")
    print(f"Evaluable rows:          {metrics['evaluable_rows']}")
    print(f"Skipped rows:            {metrics['skipped_rows']}")
    print(
        f"Accuracy (Exact Match):  "
        f"{metrics['exact_match_count']}/{metrics['evaluable_rows']} -> {metrics['accuracy']}%"
    )
    print(
        f"Partial Match Accuracy:  "
        f"{metrics['partial_match_count']}/{metrics['evaluable_rows']} -> {metrics['partial_match_accuracy']}%"
    )
    print(f"Precision:               {metrics['precision']}%")
    print(f"Recall:                  {metrics['recall']}%")
    print(f"F1 Score:                {metrics['f1_score']}%")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QA results using accuracy, precision, recall, and F1"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        default=None,
        help="Evaluate a specific model (default: evaluate all models)",
    )
    args = parser.parse_args()

    models_to_eval = [args.model] if args.model else MODELS

    print("\n=== QA Evaluation Report ===")

    all_metrics = []
    for model in models_to_eval:
        metrics = evaluate_model(model)
        if metrics:
            print_report(metrics)
            all_metrics.append(metrics)

    if len(all_metrics) > 1:
        print(f"\n{'=' * 90}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 90}")
        print(
            f"{'Model':<15} {'Accuracy':>10} {'Partial':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
        )
        print(
            f"{'-' * 15} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}"
        )
        for m in all_metrics:
            print(
                f"{m['model']:<15} "
                f"{m['accuracy']:>9}% "
                f"{m['partial_match_accuracy']:>9}% "
                f"{m['precision']:>9}% "
                f"{m['recall']:>9}% "
                f"{m['f1_score']:>9}%"
            )

    summary_path = os.path.join(OUTPUT_BASE, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nSummary saved to: {summary_path}\n")


if __name__ == "__main__":
    main()

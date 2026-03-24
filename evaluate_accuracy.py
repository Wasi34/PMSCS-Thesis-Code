"""
evaluate_accuracy.py
--------------------
Calculate accuracy metrics for each model's output JSON.

Supports:
  - Exact Match (EM): predicted answer exactly equals expected answer
  - Partial Match (F1-like): predicted answer contains expected answer or vice versa
  - Skip rows where expected_answer is empty

Usage:
    python evaluate_accuracy.py
    python evaluate_accuracy.py --model gpt4omini
    python evaluate_accuracy.py --model gemini25pro
    python evaluate_accuracy.py --model llama3_groq
"""

import argparse
import json
import os
from typing import Optional


MODELS = ["gpt4omini", "gemini25pro", "llama3_groq"]
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "outputs")


def normalize(text: str) -> str:
    """Normalize text for comparison: lowercase and strip whitespace."""
    return text.strip().lower()


def exact_match(predicted: str, expected: str) -> bool:
    """Check if predicted answer exactly matches expected answer."""
    return normalize(predicted) == normalize(expected)


def partial_match(predicted: str, expected: str) -> bool:
    """Check if expected answer is contained within the predicted answer."""
    pred_norm = normalize(predicted)
    exp_norm = normalize(expected)
    return exp_norm in pred_norm or pred_norm in exp_norm


def evaluate_model(model_name: str) -> Optional[dict]:
    """Load results JSON and compute accuracy metrics for a model."""
    results_path = os.path.join(OUTPUT_BASE, model_name, "results.json")

    if not os.path.exists(results_path):
        print(f"  ⚠️  No results found for '{model_name}' at: {results_path}")
        return None

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)
    # Only evaluate rows where an expected answer exists
    evaluable = [r for r in results if r.get("expected_answer", "").strip()]
    skipped = total - len(evaluable)

    em_correct = sum(
        1 for r in evaluable
        if exact_match(r.get("predicted_answer", ""), r["expected_answer"])
    )
    partial_correct = sum(
        1 for r in evaluable
        if partial_match(r.get("predicted_answer", ""), r["expected_answer"])
    )

    em_score = em_correct / len(evaluable) * 100 if evaluable else 0.0
    partial_score = partial_correct / len(evaluable) * 100 if evaluable else 0.0

    return {
        "model": model_name,
        "total_rows": total,
        "evaluable_rows": len(evaluable),
        "skipped_rows": skipped,
        "exact_match_count": em_correct,
        "exact_match_accuracy": round(em_score, 2),
        "partial_match_count": partial_correct,
        "partial_match_accuracy": round(partial_score, 2),
    }


def print_report(metrics: dict) -> None:
    """Pretty-print evaluation metrics for a model."""
    print(f"\n{'─'*50}")
    print(f"  Model: {metrics['model']}")
    print(f"{'─'*50}")
    print(f"  Total rows:            {metrics['total_rows']}")
    print(f"  Evaluable rows:        {metrics['evaluable_rows']}")
    print(f"  Skipped (no answer):   {metrics['skipped_rows']}")
    print(f"  Exact Match:           {metrics['exact_match_count']}/{metrics['evaluable_rows']}  → {metrics['exact_match_accuracy']}%")
    print(f"  Partial Match:         {metrics['partial_match_count']}/{metrics['evaluable_rows']}  → {metrics['partial_match_accuracy']}%")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy of chain-of-thought QA results"
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

    print("\n=== Chain-of-Thought QA — Accuracy Evaluation ===")

    all_metrics = []
    for model in models_to_eval:
        metrics = evaluate_model(model)
        if metrics:
            print_report(metrics)
            all_metrics.append(metrics)

    if len(all_metrics) > 1:
        print(f"\n{'═'*50}")
        print("  COMPARISON SUMMARY")
        print(f"{'═'*50}")
        print(f"  {'Model':<20} {'Exact Match':>12} {'Partial Match':>14}")
        print(f"  {'─'*20} {'─'*12} {'─'*14}")
        for m in all_metrics:
            print(
                f"  {m['model']:<20} {m['exact_match_accuracy']:>11}% {m['partial_match_accuracy']:>13}%"
            )
        print()

    # Save summary to JSON
    summary_path = os.path.join(OUTPUT_BASE, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"  📄 Summary saved to: {summary_path}\n")


if __name__ == "__main__":
    main()

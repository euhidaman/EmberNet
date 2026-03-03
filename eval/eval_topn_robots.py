#!/usr/bin/env python3
"""
eval/eval_topn_robots.py
========================
Evaluate EmberNet on the Top-N multi-robot selection benchmark.

Metrics:
  - Exact set accuracy (predicted robot set == gold robot set)
  - Per-robot precision, recall, F1
  - Macro-averaged F1 across all five robot types

Usage:
    python eval/eval_topn_robots.py --model checkpoints/stage2/final_model.pt
    python eval/eval_topn_robots.py --model checkpoints/stage2/final_model.pt --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROBOT_TYPES = [
    "Drone",
    "Underwater Robot",
    "Humanoid",
    "Robot with Wheels",
    "Robot with Legs",
]

# Lower-cased lookup for fuzzy matching
_ROBOT_LOWER = {r.lower(): r for r in ROBOT_TYPES}


def load_dataset(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: Dataset not found at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def gold_robots(entry: dict) -> set[str]:
    """Extract unique robot names from subtasks."""
    robots = set()
    for st in entry.get("subtasks", []):
        name = st.get("assigned_robot", "").strip()
        if name:
            robots.add(name)
    return robots


def build_prompt(entry: dict) -> str:
    instruction = entry.get("instruction", "").strip()
    task_input = entry.get("input", "").strip()
    return (
        f"{instruction}\n\n"
        f"{task_input}\n\n"
        f"List all suitable robot types for this task, separated by commas. "
        f"Choose from: {', '.join(ROBOT_TYPES)}."
    )


def parse_predicted_robots(text: str) -> set[str]:
    """Detect which of the five robot types appear in model output (case-insensitive)."""
    text_lower = text.lower()
    found = set()
    for canonical in ROBOT_TYPES:
        if canonical.lower() in text_lower:
            found.add(canonical)
    return found


def compute_metrics(predictions: list[tuple[set, set]]) -> dict:
    n = len(predictions)
    if n == 0:
        return {"num_examples": 0}

    exact_matches = sum(1 for pred, gold in predictions if pred == gold)

    # Per-robot TP/FP/FN
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred, gold in predictions:
        for r in ROBOT_TYPES:
            in_pred = r in pred
            in_gold = r in gold
            if in_pred and in_gold:
                tp[r] += 1
            elif in_pred and not in_gold:
                fp[r] += 1
            elif not in_pred and in_gold:
                fn[r] += 1

    per_robot = {}
    f1s = []
    for r in ROBOT_TYPES:
        p = tp[r] / (tp[r] + fp[r]) if (tp[r] + fp[r]) > 0 else 0.0
        rec = tp[r] / (tp[r] + fn[r]) if (tp[r] + fn[r]) > 0 else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        per_robot[r] = {"precision": round(p, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        f1s.append(f1)

    return {
        "num_examples": n,
        "exact_set_accuracy": round(exact_matches / n, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "per_robot": per_robot,
    }


def run(args):
    data = load_dataset(args.dataset)
    if args.limit:
        data = data[:args.limit]

    # Load model
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from inference import EmberVLM
    model = EmberVLM(model_path=args.model, device=args.device)

    predictions = []
    for i, entry in enumerate(data):
        prompt = build_prompt(entry)
        gold = gold_robots(entry)

        response = model.chat(prompt=prompt, reset=True)
        pred = parse_predicted_robots(response)
        predictions.append((pred, gold))

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(data)}] pred={sorted(pred)} gold={sorted(gold)}")

    metrics = compute_metrics(predictions)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "topn_robots.json"
    out_file.write_text(json.dumps(metrics, indent=2))

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Top-N Robot Selection Results")
    print(f"{'='*50}")
    print(f"  Examples:           {metrics['num_examples']}")
    print(f"  Exact Set Accuracy: {metrics['exact_set_accuracy']:.4f}")
    print(f"  Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"  Per-robot:")
    for r in ROBOT_TYPES:
        m = metrics["per_robot"][r]
        print(f"    {r:<22} P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
    print(f"{'='*50}")
    print(f"  Saved → {out_file}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate EmberNet on Top-N robot selection")
    parser.add_argument("--model", type=str, default="checkpoints/stage2/final_model.pt")
    parser.add_argument("--dataset", type=str,
                        default="benchmarks/robot-selection-dataset/multi_robot_selection_dataset.json")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

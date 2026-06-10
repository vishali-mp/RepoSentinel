"""
RepoSentinel · evals/ab_compare.py

A/B prompt comparison: runs two system prompt variants against the golden
dataset and prints a side-by-side precision/recall/F1 table.

This is how you turn prompt engineering from gut-feel into evidence.
Every time you change the system prompt, run this script. Keep the variant
that scores better. Log the result. Over time you build up a picture of
what actually works.

Usage
-----
  # Compare control vs a variant stored in files:
  python -m evals.ab_compare \\
    --control evals/prompts/v1.txt \\
    --variant evals/prompts/v2.txt \\
    --label-a "v1-baseline" \\
    --label-b "v2-stricter-confidence"

  # Compare against the built-in default prompt:
  python -m evals.ab_compare \\
    --variant evals/prompts/experiment.txt \\
    --label-b "experiment-added-cwe-context"

  # Run 3 times each and average (reduces LLM variance):
  python -m evals.ab_compare \\
    --variant evals/prompts/v2.txt \\
    --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.analyzer import SYSTEM_PROMPT
from evals.run_evals import run_evals, EvalReport


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v:.1%}"

def _delta(a: float, b: float) -> str:
    d = b - a
    if abs(d) < 0.001:
        return "  ≈ 0"
    sign = "+" if d > 0 else "−"
    return f"{sign}{abs(d):.1%}"

def _winner(a: float, b: float, metric: str) -> str:
    if abs(a - b) < 0.01:
        return "  tie"
    return "  B wins" if b > a else "  A wins"

def _bar(v: float, width: int = 20) -> str:
    filled = round(v * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Multi-run averaging
# ---------------------------------------------------------------------------

def run_multiple(
    prompt_label: str,
    prompt_text: str | None,
    dataset_path: str,
    n_runs: int,
) -> dict[str, float]:
    """Run evals n times and return averaged scores."""
    precisions, recalls, f1s = [], [], []
    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs} for '{prompt_label}' …")
        report = run_evals(
            dataset_path=dataset_path,
            prompt_label=f"{prompt_label}-run{i+1}",
            prompt_override=prompt_text,
            quiet=True,
        )
        precisions.append(report.precision)
        recalls.append(report.recall)
        f1s.append(report.f1)

    return {
        "precision": mean(precisions),
        "recall":    mean(recalls),
        "f1":        mean(f1s),
        "precision_std": stdev(precisions) if n_runs > 1 else 0.0,
        "recall_std":    stdev(recalls)    if n_runs > 1 else 0.0,
        "f1_std":        stdev(f1s)        if n_runs > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_comparison(
    label_a: str, scores_a: dict,
    label_b: str, scores_b: dict,
    n_runs: int,
):
    W = 68
    print(f"\n{'═' * W}")
    print(f"  A/B PROMPT COMPARISON  |  {n_runs} run(s) each")
    print(f"{'═' * W}")
    print(f"  {'Metric':<14} {'A: ' + label_a:<22} {'B: ' + label_b:<22} {'Δ':<10} {'Winner'}")
    print(f"  {'─'*14} {'─'*22} {'─'*22} {'─'*10} {'─'*8}")

    for metric in ("precision", "recall", "f1"):
        a = scores_a[metric]
        b = scores_b[metric]
        a_std = scores_a.get(f"{metric}_std", 0)
        b_std = scores_b.get(f"{metric}_std", 0)
        std_note = f" ±{a_std:.2f}" if n_runs > 1 else ""
        std_note_b = f" ±{b_std:.2f}" if n_runs > 1 else ""
        label = metric.replace("_", " ").title()
        print(f"  {label:<14} {_pct(a)}{std_note:<22} {_pct(b)}{std_note_b:<22} {_delta(a,b):<10} {_winner(a, b, metric)}")

    print(f"\n  Visual (F1):")
    print(f"  A  {_bar(scores_a['f1'])}  {_pct(scores_a['f1'])}")
    print(f"  B  {_bar(scores_b['f1'])}  {_pct(scores_b['f1'])}")

    # Overall recommendation
    f1_a, f1_b = scores_a["f1"], scores_b["f1"]
    print(f"\n{'─' * W}")
    if abs(f1_a - f1_b) < 0.02:
        print(f"  ◉  Results within noise margin (~2%). No clear winner.")
        print(f"     Consider: longer context, more diverse dataset, or more runs.")
    elif f1_b > f1_a:
        improvement = (f1_b - f1_a) / f1_a * 100 if f1_a > 0 else 0
        print(f"  ◉  Variant B wins by {improvement:.0f}% relative F1 improvement.")
        print(f"     Recommendation: adopt variant B as your new default prompt.")
    else:
        degradation = (f1_a - f1_b) / f1_a * 100 if f1_a > 0 else 0
        print(f"  ◉  Variant A (control) wins. B degrades F1 by {degradation:.0f}%.")
        print(f"     Recommendation: discard variant B, keep iterating.")
    print(f"{'═' * W}\n")

    # Save result to results.jsonl
    result_row = {
        "type": "ab_comparison",
        "label_a": label_a, "label_b": label_b,
        "n_runs": n_runs,
        "scores_a": scores_a, "scores_b": scores_b,
        "winner": "B" if f1_b > f1_a + 0.02 else ("A" if f1_a > f1_b + 0.02 else "tie"),
    }
    results_path = Path("evals/results.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a") as fh:
        fh.write(json.dumps(result_row) + "\n")
    print(f"  Result logged → {results_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="A/B prompt comparison for RepoSentinel")
    p.add_argument("--control", default=None,
                   help="Path to control (A) prompt file. Defaults to the built-in SYSTEM_PROMPT.")
    p.add_argument("--variant", required=True,
                   help="Path to variant (B) prompt file to test.")
    p.add_argument("--label-a", default="control",
                   help="Label for the A variant (default: 'control')")
    p.add_argument("--label-b", default="variant",
                   help="Label for the B variant")
    p.add_argument("--dataset", default="evals/golden_dataset.json",
                   help="Golden dataset to evaluate against")
    p.add_argument("--runs", type=int, default=1,
                   help="Number of runs to average over (reduces LLM variance, costs more API calls)")
    args = p.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Load prompts
    prompt_a = Path(args.control).read_text() if args.control else SYSTEM_PROMPT
    prompt_b = Path(args.variant).read_text()

    print(f"\nRunning A/B comparison: '{args.label_a}' vs '{args.label_b}'")
    print(f"Dataset: {args.dataset} | Runs: {args.runs}")
    print(f"\n[A] {args.label_a} — {len(prompt_a)} chars")
    print(f"[B] {args.label_b} — {len(prompt_b)} chars")

    print(f"\nScoring A: '{args.label_a}' …")
    scores_a = run_multiple(args.label_a, prompt_a, args.dataset, args.runs)

    print(f"\nScoring B: '{args.label_b}' …")
    scores_b = run_multiple(args.label_b, prompt_b, args.dataset, args.runs)

    print_comparison(args.label_a, scores_a, args.label_b, scores_b, args.runs)


if __name__ == "__main__":
    main()

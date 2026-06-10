"""
RepoSentinel · evals/run_evals.py

Eval harness: runs the analyzer against the golden dataset and computes
precision, recall, and F1 score.

The core challenge: a Finding from the agent doesn't map 1:1 to an
expected finding. We use fuzzy matching — a finding is a True Positive
if it's in the right file AND at least one expected keyword appears in
the combined title+description+suggestion text.

Usage
-----
  # Run against the default golden dataset:
  python -m evals.run_evals

  # Use a custom dataset:
  python -m evals.run_evals --dataset evals/my_dataset.json

  # Save results to JSONL for trend tracking:
  python -m evals.run_evals --out evals/results.jsonl

  # Quiet mode (just print the final score):
  python -m evals.run_evals --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

# ── Add project root to path so we can import agent.* ──────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.analyzer import Finding, query_llm


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    """One row from the golden dataset."""
    id: str
    description: str
    language: str
    filename: str
    code: str
    expected_findings: list[dict]
    notes: str


@dataclass
class MatchResult:
    """Outcome for one expected finding vs agent output."""
    case_id: str
    expected: dict
    matched_finding: dict | None     # the agent finding that matched, or None
    is_tp: bool                       # True Positive — expected and found
    is_fn: bool                       # False Negative — expected but missed


@dataclass
class EvalReport:
    run_id: str
    timestamp: str
    dataset_path: str
    prompt_label: str
    n_cases: int
    n_expected: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    match_results: list[dict]
    fp_details: list[dict]           # findings that didn't match anything expected


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def _finding_text(f: Finding) -> str:
    return f"{f.title} {f.description} {f.suggestion}".lower()


def _matches_expected(
    finding: Finding,
    expected: dict,
    line_proximity: int = 15,
) -> bool:
    """
    Fuzzy match: a finding counts as matching an expected finding if:
    1. At least one expected keyword appears in the finding text (required)
    2. The category matches (required when specified)
    3. The file matches by basename (required when specified)
    4. The reported line is within line_proximity lines of the expected
       line_range (soft check — skipped when line_range is absent)

    line_proximity=15 is intentionally generous. LLMs often report the
    call site rather than the definition, or flag the enclosing function
    rather than the exact line. We care that the agent found the class
    of issue in the right ballpark, not that it pinpointed the exact line.

    The line_range field in golden_dataset.json has the schema:
      line_range: [start_line, end_line]  (1-indexed, inclusive)
    """
    # 1. Keyword match (required — most discriminating signal)
    text = _finding_text(finding)
    keywords = [kw.lower() for kw in expected.get("keywords", [])]
    if keywords and not any(kw in text for kw in keywords):
        return False

    # 2. Category match (required when specified)
    expected_cat = expected.get("category", "")
    if expected_cat and finding.category != expected_cat:
        return False

    # 3. File match by basename (required when specified)
    if "file" in expected:
        if Path(finding.file).name != Path(expected["file"]).name:
            return False

    # 4. Line proximity (soft — only applied when line_range is present)
    line_range = expected.get("line_range")
    if line_range and len(line_range) == 2:
        expected_start, expected_end = line_range
        lo = expected_start - line_proximity
        hi = expected_end   + line_proximity
        finding_in_range = (
            lo <= finding.line_start <= hi or
            lo <= finding.line_end   <= hi
        )
        if not finding_in_range:
            return False

    return True


def match_findings(
    case: EvalCase,
    agent_findings: list[Finding],
) -> tuple[list[MatchResult], list[Finding]]:
    """
    Match agent findings to expected findings for a single case.

    Returns:
      - List of MatchResult (one per expected finding)
      - List of unmatched agent findings (False Positives)
    """
    matched_agent_indices: set[int] = set()
    results: list[MatchResult] = []

    for expected in case.expected_findings:
        matched = None
        for i, finding in enumerate(agent_findings):
            if i in matched_agent_indices:
                continue
            if _matches_expected(finding, expected):
                matched = finding
                matched_agent_indices.add(i)
                break

        results.append(MatchResult(
            case_id=case.id,
            expected=expected,
            matched_finding=asdict(matched) if matched else None,
            is_tp=matched is not None,
            is_fn=matched is None,
        ))

    # Unmatched agent findings = False Positives
    false_positives = [
        f for i, f in enumerate(agent_findings)
        if i not in matched_agent_indices
    ]
    return results, false_positives


# ---------------------------------------------------------------------------
# Agent runner — calls the LLM for a single code snippet
# ---------------------------------------------------------------------------

def run_agent_on_case(case: EvalCase, prompt_override: str | None = None) -> list[Finding]:
    """
    Run the analyzer on a single golden dataset case.

    We write the snippet to a temp file, create a minimal diff, and
    call query_llm directly (skipping git and static tools for speed).
    """
    # Build a minimal fake diff so the prompt context is similar to real usage
    lines = case.code.split("\n")
    diff_lines = [f"--- /dev/null", f"+++ b/{case.filename}"]
    diff_lines += [f"+{line}" for line in lines]
    fake_diff = "\n".join(diff_lines)

    file_contents = {case.filename: case.code}
    static_output = "{}"   # no static tools in eval — isolates LLM signal

    try:
        findings = query_llm(
            diff=fake_diff,
            file_contents=file_contents,
            static_output=static_output,
            confidence_threshold=0.0,    # collect everything, filter after
            system_prompt_override=prompt_override,
        )
    except Exception as exc:
        print(f"  [!] LLM call failed for {case.id}: {exc}")
        findings = []

    return findings


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_scores(
    all_match_results: list[MatchResult],
    all_fps: list[Finding],
) -> tuple[float, float, float]:
    """Return (precision, recall, f1)."""
    tp = sum(1 for r in all_match_results if r.is_tp)
    fn = sum(1 for r in all_match_results if r.is_fn)
    fp = len(all_fps)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

def run_evals(
    dataset_path: str = "evals/golden_dataset.json",
    out_path: str | None = None,
    prompt_label: str = "default",
    prompt_override: str | None = None,
    quiet: bool = False,
) -> EvalReport:
    import uuid
    run_id = uuid.uuid4().hex[:8]

    # Load dataset
    cases = [
        EvalCase(**row)
        for row in json.loads(Path(dataset_path).read_text())
    ]

    if not quiet:
        print(f"\n{'='*60}")
        print(f"RepoSentinel Eval Harness  |  run {run_id}")
        print(f"Dataset: {dataset_path}  ({len(cases)} cases)")
        print(f"Prompt:  {prompt_label}")
        print(f"{'='*60}\n")

    all_match_results: list[MatchResult] = []
    all_fps: list[Finding]               = []

    for i, case in enumerate(cases, 1):
        if not quiet:
            expected_count = len(case.expected_findings)
            label = f"✓ clean" if expected_count == 0 else f"{expected_count} bug(s)"
            print(f"[{i:02d}/{len(cases):02d}] {case.id:<15} {case.description[:45]:<45} ({label})")

        agent_findings = run_agent_on_case(case, prompt_override)

        # True negatives: clean cases that should have no findings
        if not case.expected_findings:
            if agent_findings:
                if not quiet:
                    print(f"         ⚠  False positives on clean case: "
                          f"{[f.title for f in agent_findings]}")
                all_fps.extend(agent_findings)
            else:
                if not quiet:
                    print(f"         ✓  Correctly returned no findings")
            continue

        match_results, fps = match_findings(case, agent_findings)
        all_match_results.extend(match_results)
        all_fps.extend(fps)

        if not quiet:
            for mr in match_results:
                icon = "✓ TP" if mr.is_tp else "✗ FN"
                kws  = mr.expected.get("keywords", [])[:2]
                matched_title = mr.matched_finding["title"] if mr.matched_finding else "—"
                print(f"         {icon}  expected={kws}  matched={matched_title!r:.50}")
            if fps:
                print(f"         ⚠  {len(fps)} FP(s): {[f.title for f in fps]}")

    # Compute scores
    precision, recall, f1 = compute_scores(all_match_results, all_fps)
    tp = sum(1 for r in all_match_results if r.is_tp)
    fn = sum(1 for r in all_match_results if r.is_fn)
    fp = len(all_fps)
    n_expected = len(all_match_results)

    report = EvalReport(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_path=dataset_path,
        prompt_label=prompt_label,
        n_cases=len(cases),
        n_expected=n_expected,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        match_results=[asdict(r) for r in all_match_results],
        fp_details=[asdict(f) for f in all_fps],
    )

    # Print summary
    if not quiet:
        print(f"\n{'─'*60}")
        print(f"Results for prompt: '{prompt_label}'")
        print(f"{'─'*60}")
        print(f"  Cases:          {len(cases)}")
        print(f"  Expected bugs:  {n_expected}")
        print(f"  True Positives: {tp}  (found the real bug)")
        print(f"  False Negatives:{fn}  (missed a real bug)")
        print(f"  False Positives:{fp}  (reported a non-bug)")
        print(f"")
        print(f"  Precision:      {precision:.1%}  (of what it reported, how much was real)")
        print(f"  Recall:         {recall:.1%}  (of real bugs, how many it found)")
        print(f"  F1 Score:       {f1:.1%}  (harmonic mean — your headline number)")
        print(f"{'─'*60}")

        if fn > 0:
            missed = [r for r in all_match_results if r.is_fn]
            print(f"\nMissed bugs (False Negatives):")
            for r in missed:
                print(f"  [{r.case_id}] {r.expected.get('keywords', [])[:2]}")

        if fp > 0:
            print(f"\nFalse Positives (hallucinated bugs):")
            for f in all_fps:
                print(f"  {f.title!r:.60}  ({f.file})")

    # Append to JSONL results file for trend tracking
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as fh:
            # Write compact summary row (not full match results, those are verbose)
            row = {k: v for k, v in asdict(report).items()
                   if k not in ("match_results", "fp_details")}
            fh.write(json.dumps(row) + "\n")
        if not quiet:
            print(f"\nResults appended → {out_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="RepoSentinel eval harness")
    p.add_argument("--dataset", default="evals/golden_dataset.json",
                   help="Path to golden dataset JSON")
    p.add_argument("--out", default="evals/results.jsonl",
                   help="JSONL file to append results to (for trend tracking)")
    p.add_argument("--prompt-label", default="default",
                   help="Label for this prompt variant (used in A/B comparison)")
    p.add_argument("--quiet", action="store_true",
                   help="Only print final scores")
    args = p.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    report = run_evals(
        dataset_path=args.dataset,
        out_path=args.out,
        prompt_label=args.prompt_label,
        quiet=args.quiet,
    )

    # Exit non-zero if recall below 50% (useful for CI)
    if report.recall < 0.5:
        print(f"\n⚠  Recall {report.recall:.0%} below 50% threshold.")
        sys.exit(1)


if __name__ == "__main__":
    main()

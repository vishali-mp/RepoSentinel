"""
RepoSentinel · main.py
CLI entrypoint. Invoked by GitHub Actions or locally.

Usage:
  python -m agent.main --mode pr --base-ref origin/main --head-ref HEAD \
         --pr-number 42 --commit-sha abc123

  python -m agent.main --mode scheduled --base-ref HEAD~10 --head-ref HEAD
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from agent.analyzer import analyze
from agent.reporter import report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RepoSentinel — AI code analysis agent")
    p.add_argument("--mode", choices=["pr", "scheduled"], default="scheduled")
    p.add_argument("--base-ref", default="HEAD~1")
    p.add_argument("--head-ref", default="HEAD")
    p.add_argument("--pr-number", type=int, default=None)
    p.add_argument("--commit-sha", default=None)
    p.add_argument("--ref", default="main", help="Branch name for scheduled runs")
    p.add_argument("--confidence", type=float, default=0.70,
                   help="Minimum confidence threshold 0.0–1.0 (default 0.70)")
    p.add_argument("--no-issues", action="store_true",
                   help="Don't open individual GitHub Issues for high/critical findings")
    p.add_argument("--json-output", action="store_true",
                   help="Print findings as JSON to stdout (useful for debugging)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[RepoSentinel] Starting {args.mode} analysis")
    print(f"  Base ref : {args.base_ref}")
    print(f"  Head ref : {args.head_ref}")
    print(f"  Confidence threshold: {args.confidence:.0%}")

    result = analyze(
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        confidence_threshold=args.confidence,
    )

    print(f"[RepoSentinel] Analysis complete")
    print(f"  Files analyzed : {result.files_analyzed}")
    print(f"  Lines analyzed : {result.lines_analyzed:,}")
    print(f"  Findings       : {len(result.findings)}")
    if result.error:
        print(f"  ⚠️  Error: {result.error}")

    if args.json_output:
        import dataclasses
        print(json.dumps(
            [dataclasses.asdict(f) for f in result.findings],
            indent=2,
        ))

    report(
        result=result,
        pr_number=args.pr_number,
        commit_sha=args.commit_sha,
        ref=args.ref,
        open_individual_issues=not args.no_issues,
    )

    # Exit non-zero if critical/high findings exist (useful for required status checks)
    critical_count = sum(
        1 for f in result.findings if f.severity in ("critical", "high")
    )
    if critical_count > 0:
        print(f"\n[RepoSentinel] ⚠️  {critical_count} critical/high finding(s). Exiting 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()

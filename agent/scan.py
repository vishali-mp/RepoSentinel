"""
RepoSentinel · scan.py

CLI entrypoint for cross-repo scanning.

Usage examples
--------------
# Discover and scan top Python AI repos:
python -m agent.scan \\
  --topics machine-learning ai \\
  --languages python \\
  --min-stars 500 \\
  --max-repos 10

# Scan an explicit watchlist:
python -m agent.scan \\
  --watchlist "huggingface/transformers" "openai/openai-python" \\
  --commits-window 30

# Full run with dashboard posting:
python -m agent.scan \\
  --topics "llm" "rag" \\
  --languages python typescript \\
  --max-repos 15 \\
  --dashboard-repo "yourname/sentinel-dashboard"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RepoSentinel cross-repo scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Discovery
    disc = p.add_argument_group("Discovery")
    disc.add_argument("--topics", nargs="*", default=[],
                      help="GitHub topics to search (e.g. machine-learning rag)")
    disc.add_argument("--languages", nargs="*", default=[],
                      help="Languages to search (e.g. python typescript)")
    disc.add_argument("--watchlist", nargs="*", default=[],
                      help="Explicit 'owner/repo' list to always include")
    disc.add_argument("--min-stars", type=int, default=200,
                      help="Minimum star count for discovered repos (default 200)")
    disc.add_argument("--exclude", nargs="*", default=[],
                      help="Repos to skip (owner/repo)")

    # Scanning
    scan = p.add_argument_group("Scanning")
    scan.add_argument("--max-repos", type=int, default=10,
                      help="Max repos to scan per run (default 10)")
    scan.add_argument("--commits-window", type=int, default=20,
                      help="Recent commits to analyze per repo (default 20)")
    scan.add_argument("--confidence", type=float, default=0.70,
                      help="Min LLM confidence threshold (default 0.70)")
    scan.add_argument("--workspace", type=str, default=".sentinel/workspace",
                      help="Directory for repo clones (default .sentinel/workspace)")
    scan.add_argument("--cleanup", action="store_true",
                      help="Delete repo clones after each scan (saves disk)")

    # Reporting
    rep = p.add_argument_group("Reporting")
    rep.add_argument("--output-dir", type=str, default=".",
                     help="Directory for JSON and markdown reports")
    rep.add_argument("--dashboard-repo", type=str, default=None,
                     help="'owner/repo' of the central dashboard repo for GitHub Issues")
    rep.add_argument("--no-github", action="store_true",
                     help="Skip GitHub Issue posting (local reports only)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.topics and not args.languages and not args.watchlist:
        print("Error: provide at least one of --topics, --languages, or --watchlist")
        sys.exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Discovery
    # ----------------------------------------------------------------
    from agent.discovery import GitHubDiscovery

    print("[Scan] Discovering repositories …")
    discovery = GitHubDiscovery()
    targets = discovery.discover(
        topics=args.topics or None,
        languages=args.languages or None,
        watchlist=args.watchlist or None,
        min_stars=args.min_stars,
        max_per_strategy=args.max_repos,
        max_total=args.max_repos,
        exclude=args.exclude,
    )

    if not targets:
        print("[Scan] No repositories discovered. Adjust --topics / --languages / --min-stars.")
        sys.exit(0)

    print(f"[Scan] Discovered {len(targets)} repos:")
    for r in targets:
        print(f"  {r.full_name:45s} ⭐{r.stars:>6,}  score={r.priority_score}")

    # ----------------------------------------------------------------
    # Scanning
    # ----------------------------------------------------------------
    from agent.scanner import CrossRepoScanner

    scanner = CrossRepoScanner(
        workspace_dir=Path(args.workspace),
        commits_window=args.commits_window,
        cleanup=args.cleanup,
        max_repos=args.max_repos,
    )
    report = scanner.scan(targets)

    # ----------------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------------
    from agent.multi_reporter import report_all

    if args.dashboard_repo:
        os.environ["SENTINEL_DASHBOARD_REPO"] = args.dashboard_repo

    outputs = report_all(
        report=report,
        output_dir=Path(args.output_dir),
        post_to_github=not args.no_github,
    )

    print("\n[Scan] Outputs:")
    for key, val in outputs.items():
        if val:
            print(f"  {key}: {val}")

    # Exit non-zero if any critical findings
    critical = sum(rr.critical_count for rr in report.repo_results)
    if critical:
        print(f"\n[Scan] ⚠️  {critical} critical finding(s) found.")
        sys.exit(1)


if __name__ == "__main__":
    main()

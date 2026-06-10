"""
RepoSentinel · multi_reporter.py

Aggregates ScanReport findings across all repos and outputs:
  1. A GitHub Issue in a central "dashboard" repo with a summary table
  2. A machine-readable findings.json artifact
  3. A human-readable scan_report.md

Designed to be the final step of the scheduled cross-repo workflow.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

from agent.reporter import SEVERITY_EMOJI, CATEGORY_EMOJI, SEVERITY_LABELS
from agent.scanner import ScanReport, RepoScanResult


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def write_json_report(report: ScanReport, output_path: Path = Path("findings.json")) -> Path:
    """Write all findings across all repos to a structured JSON file."""
    output: dict = {
        "scan_id":      report.scan_id,
        "started_at":   report.started_at,
        "finished_at":  report.finished_at,
        "summary": {
            "repos_scanned":  len(report.repo_results),
            "total_findings": report.total_findings,
            "total_files":    report.total_files,
            "total_lines":    report.total_lines,
        },
        "repos": [],
    }

    for rr in report.repo_results:
        repo_entry = {
            "repo":             rr.repo.full_name,
            "stars":            rr.repo.stars,
            "language":         rr.repo.language,
            "priority_score":   rr.repo.priority_score,
            "scanned_at":       rr.scanned_at,
            "duration_seconds": rr.duration_seconds,
            "files_analyzed":   rr.result.files_analyzed,
            "lines_analyzed":   rr.result.lines_analyzed,
            "error":            rr.error,
            "findings": [dataclasses.asdict(f) for f in rr.result.findings],
        }
        output["repos"].append(repo_entry)

    output_path.write_text(json.dumps(output, indent=2))
    print(f"[MultiReporter] JSON report → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown_report(report: ScanReport, output_path: Path = Path("scan_report.md")) -> Path:
    """Write a human-readable markdown report with per-repo finding details."""
    sections = [report.summarize(), ""]

    for rr in sorted(report.repo_results, key=lambda x: -x.finding_count):
        if not rr.result.findings and not rr.error:
            continue
        sections.append(f"---\n## [{rr.repo.full_name}](https://github.com/{rr.repo.full_name})")
        sections.append(
            f"⭐ {rr.repo.stars:,} · {rr.repo.language or 'unknown'} · "
            f"scanned in {rr.duration_seconds}s"
        )

        if rr.error:
            sections.append(f"\n⚠️ **Scan error:** `{rr.error}`\n")
            continue

        if not rr.result.findings:
            sections.append("\n✅ No findings above threshold.\n")
            continue

        for f in sorted(rr.result.findings,
                        key=lambda x: ["critical","high","medium","low"].index(x.severity)):
            sev  = SEVERITY_EMOJI[f.severity]
            cat  = CATEGORY_EMOJI[f.category]
            sections += [
                f"\n### {sev} {f.title}",
                f"`{f.file}:{f.line_start}` · {cat} {f.category} · confidence {f.confidence:.0%}",
                f"\n{f.description}",
                f"\n**Fix:** {f.suggestion}",
            ]

    output_path.write_text("\n".join(sections))
    print(f"[MultiReporter] Markdown report → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# GitHub dashboard issue
# ---------------------------------------------------------------------------

def post_dashboard_issue(
    report: ScanReport,
    token: str | None = None,
    dashboard_repo: str | None = None,
) -> dict | None:
    """
    Post a summary issue to a central dashboard repo.

    dashboard_repo should be a dedicated public or private repo like
    'yourname/sentinel-dashboard' where all cross-repo scan summaries land.
    """
    token = token or os.getenv("GITHUB_TOKEN")
    dashboard_repo = dashboard_repo or os.getenv("SENTINEL_DASHBOARD_REPO")

    if not token or not dashboard_repo:
        print("[MultiReporter] No SENTINEL_DASHBOARD_REPO set — skipping dashboard issue.")
        return None

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })

    # Severity breakdown across all repos
    all_findings = [
        f for rr in report.repo_results for f in rr.result.findings
    ]
    by_severity = {s: 0 for s in ["critical", "high", "medium", "low"]}
    by_category = {c: 0 for c in ["bug", "performance", "security", "style"]}
    for f in all_findings:
        by_severity[f.severity] += 1
        by_category[f.category] += 1

    sev_summary = " · ".join(
        f"{SEVERITY_EMOJI[s]} {n} {s}" for s, n in by_severity.items() if n
    )
    cat_summary = " · ".join(
        f"{CATEGORY_EMOJI[c]} {n} {c}" for c, n in by_category.items() if n
    )

    body = "\n".join([
        report.summarize(),
        "",
        "## Severity breakdown",
        sev_summary or "No findings.",
        "",
        "## Category breakdown",
        cat_summary or "No findings.",
        "",
        "## Top findings",
        "",
        "| Severity | Category | Repo | File | Title |",
        "|---|---|---|---|---|",
        *[
            f"| {SEVERITY_EMOJI[f.severity]} {f.severity} "
            f"| {CATEGORY_EMOJI[f.category]} {f.category} "
            f"| [{rr.repo.full_name}](https://github.com/{rr.repo.full_name}) "
            f"| `{f.file}:{f.line_start}` "
            f"| {f.title} |"
            for rr in report.repo_results
            for f in rr.result.findings
            if f.severity in ("critical", "high")
        ][:20],   # cap at 20 rows
        "",
        f"---",
        f"*Scan ID: `{report.scan_id}` · "
        f"[Full report artifact](https://github.com/{dashboard_repo}/actions)*",
    ])

    title = (
        f"[Sentinel] Cross-repo scan — {report.total_findings} findings "
        f"across {len(report.repo_results)} repos · {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    )

    resp = session.post(
        f"https://api.github.com/repos/{dashboard_repo}/issues",
        json={"title": title, "body": body, "labels": ["sentinel: scan"]},
    )
    if resp.ok:
        issue = resp.json()
        print(f"[MultiReporter] Dashboard issue → {issue['html_url']}")
        return issue
    else:
        print(f"[MultiReporter] Failed to post dashboard issue: {resp.text[:200]}")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def report_all(
    report: ScanReport,
    output_dir: Path = Path("."),
    post_to_github: bool = True,
) -> dict:
    """Run all reporting outputs. Returns paths and issue URL."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = write_json_report(report, output_dir / "findings.json")
    md_path   = write_markdown_report(report, output_dir / "scan_report.md")

    issue = None
    if post_to_github:
        issue = post_dashboard_issue(report)

    return {
        "json_report":     str(json_path),
        "markdown_report": str(md_path),
        "dashboard_issue": issue.get("html_url") if issue else None,
    }

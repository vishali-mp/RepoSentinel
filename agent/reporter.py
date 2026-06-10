"""
RepoSentinel · reporter.py
Posts analysis findings to GitHub as Issues and/or PR review comments.
Tracks already-reported issues to prevent duplicate noise.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import requests

from agent.analyzer import AnalysisResult, Finding, Severity

# ---------------------------------------------------------------------------
# Severity → GitHub label / emoji mapping
# ---------------------------------------------------------------------------

SEVERITY_EMOJI: dict[Severity, str] = {
    "critical": "🔴",
    "high":     "🟠",
    "medium":   "🟡",
    "low":      "🔵",
}

CATEGORY_EMOJI = {
    "bug":         "🐛",
    "performance": "⚡",
    "security":    "🔒",
    "style":       "🎨",
}

SEVERITY_LABELS: dict[Severity, str] = {
    "critical": "sentinel: critical",
    "high":     "sentinel: high",
    "medium":   "sentinel: medium",
    "low":      "sentinel: low",
}


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _finding_fingerprint(f: Finding) -> str:
    """Stable hash for a finding — used to avoid reopening duplicate issues."""
    key = f"{f.file}:{f.line_start}:{f.title}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _finding_to_markdown(f: Finding, include_header: bool = True) -> str:
    sev_icon = SEVERITY_EMOJI[f.severity]
    cat_icon = CATEGORY_EMOJI[f.category]
    lines = []
    if include_header:
        lines.append(f"## {sev_icon} {f.title}")
        lines.append("")
    lines += [
        f"| Field | Value |",
        f"|---|---|",
        f"| **Category** | {cat_icon} {f.category.capitalize()} |",
        f"| **Severity** | {sev_icon} {f.severity.capitalize()} |",
        f"| **File** | `{f.file}` |",
        f"| **Lines** | {f.line_start}–{f.line_end} |",
        f"| **Confidence** | {f.confidence:.0%} |",
        "",
        "### What's wrong",
        f.description,
        "",
        "### Suggested fix",
        f.suggestion,
        "",
        "---",
        f"*RepoSentinel · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
    ]
    return "\n".join(lines)


def _summary_table(findings: list[Finding]) -> str:
    if not findings:
        return "✅ No issues found above the confidence threshold."

    rows = ["| Severity | Category | File | Title |", "|---|---|---|---|"]
    for f in sorted(findings, key=lambda x: ["critical","high","medium","low"].index(x.severity)):
        sev = f"{SEVERITY_EMOJI[f.severity]} {f.severity}"
        cat = f"{CATEGORY_EMOJI[f.category]} {f.category}"
        rows.append(f"| {sev} | {cat} | `{f.file}:{f.line_start}` | {f.title} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# GitHub API client
# ---------------------------------------------------------------------------

class GitHubReporter:
    def __init__(self, token: str, repo: str):
        """
        token: GitHub token with issues:write and pull_requests:write
        repo:  owner/repo-name
        """
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self.repo = repo
        self.base = f"https://api.github.com/repos/{repo}"

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    def ensure_labels(self) -> None:
        """Create sentinel labels if they don't exist."""
        label_colors = {
            "sentinel: critical": "d73a4a",
            "sentinel: high":     "e4e669",
            "sentinel: medium":   "0075ca",
            "sentinel: low":      "cfd3d7",
            "sentinel: agent":    "7057ff",
        }
        existing = {l["name"] for l in self._get(f"{self.base}/labels").json()}
        for name, color in label_colors.items():
            if name not in existing:
                self.session.post(f"{self.base}/labels", json={"name": name, "color": color})

    # ------------------------------------------------------------------
    # Deduplication via issue search
    # ------------------------------------------------------------------

    def _fingerprint_in_open_issues(self, fingerprint: str) -> bool:
        resp = self._get(
            f"{self.base}/issues",
            params={"state": "open", "labels": "sentinel: agent", "per_page": 100},
        )
        for issue in resp.json():
            if fingerprint in (issue.get("body") or ""):
                return True
        return False

    # ------------------------------------------------------------------
    # Issue creation
    # ------------------------------------------------------------------

    def create_issue(self, f: Finding) -> dict | None:
        fingerprint = _finding_fingerprint(f)
        if self._fingerprint_in_open_issues(fingerprint):
            return None  # already reported

        body = _finding_to_markdown(f) + f"\n\n<!-- sentinel-fp:{fingerprint} -->"
        labels = ["sentinel: agent", SEVERITY_LABELS[f.severity]]

        resp = self.session.post(f"{self.base}/issues", json={
            "title": f"[Sentinel] {f.title}",
            "body": body,
            "labels": labels,
        })
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # PR summary comment
    # ------------------------------------------------------------------

    def post_pr_summary(self, pr_number: int, result: AnalysisResult) -> None:
        findings = result.findings
        summary = "\n".join([
            "# 🤖 RepoSentinel Analysis",
            "",
            f"Analyzed **{result.files_analyzed} files** / **{result.lines_analyzed:,} lines**.",
            "",
            _summary_table(findings),
            "",
            "---",
            f"*{len(findings)} finding(s) · confidence threshold 70% · "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        ])

        # Delete previous sentinel PR comment if exists
        comments = self._get(f"{self.base}/issues/{pr_number}/comments").json()
        for c in comments:
            if "🤖 RepoSentinel Analysis" in (c.get("body") or ""):
                self.session.delete(f"{self.base}/issues/comments/{c['id']}")

        self.session.post(
            f"{self.base}/issues/{pr_number}/comments",
            json={"body": summary},
        ).raise_for_status()

    # ------------------------------------------------------------------
    # PR inline review comments
    # ------------------------------------------------------------------

    def post_pr_review(self, pr_number: int, commit_sha: str, findings: list[Finding]) -> None:
        """Post inline review comments on the PR diff."""
        if not findings:
            return

        comments = []
        for f in findings:
            body = (
                f"{SEVERITY_EMOJI[f.severity]} **{f.title}**\n\n"
                f"{f.description}\n\n"
                f"**Suggestion:** {f.suggestion}"
            )
            comments.append({
                "path": f.file,
                "line": f.line_end,
                "side": "RIGHT",
                "body": body,
            })

        self.session.post(f"{self.base}/pulls/{pr_number}/reviews", json={
            "commit_id": commit_sha,
            "event": "COMMENT",
            "comments": comments[:10],  # GitHub caps at 10 per review
        })

    # ------------------------------------------------------------------
    # Scheduled run summary (posts as issue when no PR)
    # ------------------------------------------------------------------

    def post_scheduled_summary(self, result: AnalysisResult, ref: str) -> dict:
        findings = result.findings
        title = (
            f"[Sentinel] Scheduled scan — {len(findings)} issue(s) found"
            if findings else
            "[Sentinel] Scheduled scan — ✅ all clear"
        )
        body_parts = [
            f"Automated scan of `{ref}`.",
            f"Analyzed **{result.files_analyzed} files** / **{result.lines_analyzed:,} lines**.",
            "",
            _summary_table(findings),
        ]
        if result.error:
            body_parts.append(f"\n⚠️ Agent error: `{result.error}`")

        resp = self.session.post(f"{self.base}/issues", json={
            "title": title,
            "body": "\n".join(body_parts),
            "labels": ["sentinel: agent"],
        })
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, **kwargs) -> requests.Response:
        resp = self.session.get(url, **kwargs)
        resp.raise_for_status()
        return resp


def report(
    result: AnalysisResult,
    token: str | None = None,
    repo: str | None = None,
    pr_number: int | None = None,
    commit_sha: str | None = None,
    ref: str = "main",
    open_individual_issues: bool = True,
) -> None:
    """
    Main reporting entry point. Called from the CLI entrypoint.
    Falls back gracefully if GitHub credentials are not set.
    """
    token = token or os.getenv("GITHUB_TOKEN")
    repo = repo or os.getenv("GITHUB_REPOSITORY")

    if not token or not repo:
        # When --json-output is active, main.py already printed findings as JSON.
        # Only print the human-readable fallback when NOT in JSON mode.
        import sys
        json_mode = any(a in sys.argv for a in ("--json-output",))
        if not json_mode:
            print("No GitHub credentials — printing findings to stdout only.")
            for f in result.findings:
                print(f"\n{SEVERITY_EMOJI[f.severity]} [{f.severity.upper()}] {f.title}")
                print(f"  File: {f.file}:{f.line_start}")
                print(f"  {f.description}")
        else:
            print("No GitHub credentials — findings already written as JSON.", file=sys.stderr)
        return

    reporter = GitHubReporter(token=token, repo=repo)
    reporter.ensure_labels()

    if pr_number and commit_sha:
        reporter.post_pr_summary(pr_number, result)
        reporter.post_pr_review(pr_number, commit_sha, result.findings)
    else:
        reporter.post_scheduled_summary(result, ref)

    if open_individual_issues and result.findings:
        high_priority = [f for f in result.findings if f.severity in ("critical", "high")]
        for finding in high_priority:
            reporter.create_issue(finding)
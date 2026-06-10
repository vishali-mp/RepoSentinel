"""
RepoSentinel · scanner.py

Cross-repo scanning orchestrator.

For each discovered RepoTarget:
  1. Shallow-clone (or update) the repo into a temp workspace
  2. Compute the diff window (last N commits or since last scan)
  3. Run the analyzer pipeline (static tools + LLM + vector store)
  4. Collect findings into a ScanReport
  5. Clean up

The ScanReport is then handed to the multi_reporter for GitHub Issue
creation and JSON/markdown report generation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agent.analyzer import AnalysisResult, analyze
from agent.discovery import RepoTarget


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RepoScanResult:
    repo: RepoTarget
    result: AnalysisResult
    duration_seconds: float
    scanned_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: str | None = None

    @property
    def finding_count(self) -> int:
        return len(self.result.findings)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.result.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.result.findings if f.severity == "high")


@dataclass
class ScanReport:
    scan_id: str
    started_at: str
    finished_at: str = ""
    repo_results: list[RepoScanResult] = field(default_factory=list)
    total_findings: int = 0
    total_files: int = 0
    total_lines: int = 0

    def summarize(self) -> str:
        lines = [
            f"# RepoSentinel Scan Report",
            f"**Scan ID:** `{self.scan_id}`",
            f"**Started:** {self.started_at}",
            f"**Finished:** {self.finished_at}",
            f"**Repos scanned:** {len(self.repo_results)}",
            f"**Total findings:** {self.total_findings}",
            f"**Files analyzed:** {self.total_files}",
            f"**Lines analyzed:** {self.total_lines:,}",
            "",
            "## Results by repo",
            "",
            "| Repo | Stars | Findings | Critical | High | Files | Error |",
            "|------|-------|----------|----------|------|-------|-------|",
        ]
        for r in sorted(self.repo_results, key=lambda x: -x.finding_count):
            err = "⚠️" if r.error else "✅"
            lines.append(
                f"| [{r.repo.full_name}](https://github.com/{r.repo.full_name}) "
                f"| ⭐ {r.repo.stars:,} "
                f"| {r.finding_count} "
                f"| {r.critical_count} "
                f"| {r.high_count} "
                f"| {r.result.files_analyzed} "
                f"| {err} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _clone_or_update(repo: RepoTarget, workspace: Path) -> Path:
    """
    Shallow-clone the repo into workspace/<repo_name>.
    If it already exists (incremental run), fetch latest instead.
    Returns the path to the local clone.
    """
    repo_dir = workspace / repo.name
    if repo_dir.exists():
        subprocess.run(
            ["git", "-C", str(repo_dir), "fetch", "--depth=50", "origin"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "reset", "--hard",
             f"origin/{repo.default_branch}"],
            check=True, capture_output=True,
        )
    else:
        subprocess.run(
            ["git", "clone", "--depth=50", "--single-branch",
             "--branch", repo.default_branch,
             repo.clone_url, str(repo_dir)],
            check=True, capture_output=True, timeout=120,
        )
    return repo_dir


def _recent_commit_range(repo_dir: Path, n_commits: int = 20) -> tuple[str, str]:
    """Return (base_ref, head_ref) covering the last n commits."""
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-list", "--max-count", str(n_commits + 1), "HEAD"],
        capture_output=True, text=True, check=True,
    )
    commits = result.stdout.strip().splitlines()
    head = commits[0]
    base = commits[-1] if len(commits) > 1 else f"{head}~1"
    return base, head


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class CrossRepoScanner:
    """
    Scans a list of RepoTargets and returns a ScanReport.

    Parameters
    ----------
    workspace_dir   : persistent dir for repo clones (reused across runs)
    commits_window  : how many recent commits to analyze per repo
    cleanup         : if True, delete clones after each scan (saves disk)
    max_repos       : safety cap on number of repos per run
    """

    def __init__(
        self,
        workspace_dir: Path | None = None,
        commits_window: int = 20,
        cleanup: bool = False,
        max_repos: int = 10,
    ):
        self.workspace = workspace_dir or Path(tempfile.mkdtemp(prefix="sentinel-ws-"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.commits_window = commits_window
        self.cleanup = cleanup
        self.max_repos = max_repos

    def scan(self, targets: list[RepoTarget], scan_id: str = "") -> ScanReport:
        import uuid
        scan_id = scan_id or uuid.uuid4().hex[:8]
        started = datetime.now(timezone.utc).isoformat()
        report = ScanReport(scan_id=scan_id, started_at=started)

        targets = targets[: self.max_repos]
        print(f"\n[Scanner] Starting scan {scan_id} — {len(targets)} repos")

        for idx, repo in enumerate(targets, 1):
            print(f"\n[Scanner] ({idx}/{len(targets)}) {repo.full_name} ⭐{repo.stars:,}")
            repo_result = self._scan_one(repo)
            report.repo_results.append(repo_result)
            report.total_findings += repo_result.finding_count
            report.total_files    += repo_result.result.files_analyzed
            report.total_lines    += repo_result.result.lines_analyzed

            if self.cleanup:
                repo_dir = self.workspace / repo.name
                if repo_dir.exists():
                    shutil.rmtree(repo_dir)

            # Polite pacing between repos
            if idx < len(targets):
                time.sleep(2)

        report.finished_at = datetime.now(timezone.utc).isoformat()
        print(f"\n[Scanner] Done. {report.total_findings} findings across {len(targets)} repos.")
        return report

    def _scan_one(self, repo: RepoTarget) -> RepoScanResult:
        t0 = time.time()
        result = AnalysisResult()
        error = None

        try:
            repo_dir = _clone_or_update(repo, self.workspace)
            base_ref, head_ref = _recent_commit_range(repo_dir, self.commits_window)

            # Run analyzer inside the cloned repo directory
            orig_dir = Path.cwd()
            os.chdir(repo_dir)
            try:
                result = analyze(
                    base_ref=base_ref,
                    head_ref=head_ref,
                    run_id=f"{repo.full_name}-{head_ref[:7]}",
                    repo=repo.full_name,
                )
            finally:
                os.chdir(orig_dir)

        except subprocess.CalledProcessError as exc:
            error = f"git error: {exc.stderr.decode()[:200] if exc.stderr else str(exc)}"
            print(f"  ⚠️  {error}")
        except Exception as exc:
            error = str(exc)
            print(f"  ⚠️  {error}")

        return RepoScanResult(
            repo=repo,
            result=result,
            duration_seconds=round(time.time() - t0, 1),
            error=error,
        )

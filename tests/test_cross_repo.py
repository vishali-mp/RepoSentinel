"""
Tests for RepoSentinel cross-repo components:
  discovery.py, scanner.py, multi_reporter.py
No network calls — all GitHub API interactions are mocked.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from agent.analyzer import AnalysisResult, Finding
from agent.discovery import RepoTarget, GitHubDiscovery
from agent.scanner import RepoScanResult, ScanReport, CrossRepoScanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_target(**kwargs) -> RepoTarget:
    defaults = dict(
        full_name="owner/cool-repo",
        clone_url="https://github.com/owner/cool-repo.git",
        default_branch="main",
        stars=1200,
        language="Python",
        topics=["machine-learning"],
        open_issues=45,
        last_pushed="2026-05-01T10:00:00Z",
        priority_score=0.0,
    )
    defaults.update(kwargs)
    return RepoTarget(**defaults)


def make_finding(**kwargs) -> Finding:
    defaults = dict(
        category="security", severity="high",
        file="auth.py", line_start=20, line_end=25,
        title="SQL injection risk", description="Unsanitised input in query.",
        suggestion="Use parameterised queries.", confidence=0.91,
    )
    defaults.update(kwargs)
    return Finding(**defaults)


def make_scan_result(n_findings: int = 2, error: str | None = None) -> RepoScanResult:
    findings = [make_finding(title=f"Issue {i}") for i in range(n_findings)]
    result = AnalysisResult(findings=findings, files_analyzed=5, lines_analyzed=400)
    return RepoScanResult(
        repo=make_target(),
        result=result,
        duration_seconds=3.2,
        error=error,
    )


def make_report(n_repos: int = 2) -> ScanReport:
    report = ScanReport(scan_id="test-abc", started_at="2026-06-01T08:00:00Z",
                        finished_at="2026-06-01T08:30:00Z")
    for i in range(n_repos):
        rr = make_scan_result(n_findings=i + 1)
        rr.repo = make_target(full_name=f"owner/repo-{i}", stars=500 + i * 100)
        report.repo_results.append(rr)
        report.total_findings += rr.finding_count
        report.total_files    += rr.result.files_analyzed
        report.total_lines    += rr.result.lines_analyzed
    return report


# ---------------------------------------------------------------------------
# RepoTarget
# ---------------------------------------------------------------------------

class TestRepoTarget(unittest.TestCase):
    def test_owner_and_name(self):
        t = make_target(full_name="huggingface/transformers")
        self.assertEqual(t.owner, "huggingface")
        self.assertEqual(t.name, "transformers")

    def test_priority_score_assigned(self):
        t = make_target(stars=5000, last_pushed="2026-05-30T00:00:00Z", open_issues=100)
        discovery = GitHubDiscovery.__new__(GitHubDiscovery)
        score = GitHubDiscovery._priority_score(t)
        self.assertGreater(score, 0)

    def test_stale_repo_lower_score(self):
        fresh = make_target(stars=1000, last_pushed="2026-05-20T00:00:00Z")
        stale = make_target(stars=1000, last_pushed="2020-01-01T00:00:00Z")
        score_fresh = GitHubDiscovery._priority_score(fresh)
        score_stale = GitHubDiscovery._priority_score(stale)
        self.assertGreater(score_fresh, score_stale)

    def test_more_stars_higher_score(self):
        big   = make_target(stars=50000, last_pushed="2026-05-01T00:00:00Z")
        small = make_target(stars=100,   last_pushed="2026-05-01T00:00:00Z")
        self.assertGreater(
            GitHubDiscovery._priority_score(big),
            GitHubDiscovery._priority_score(small),
        )


# ---------------------------------------------------------------------------
# GitHubDiscovery
# ---------------------------------------------------------------------------

class TestGitHubDiscovery(unittest.TestCase):
    GITHUB_ITEM = {
        "full_name": "owner/repo",
        "clone_url": "https://github.com/owner/repo.git",
        "default_branch": "main",
        "stargazers_count": 800,
        "language": "Python",
        "topics": ["ai"],
        "open_issues_count": 30,
        "pushed_at": "2026-05-15T12:00:00Z",
    }

    def _mock_session(self, items: list[dict]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"items": items, "total_count": len(items)}
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        return mock_session

    def test_parse_repo(self):
        target = GitHubDiscovery._parse_repo(self.GITHUB_ITEM)
        self.assertEqual(target.full_name, "owner/repo")
        self.assertEqual(target.stars, 800)
        self.assertEqual(target.language, "Python")

    def test_by_topic_calls_search(self):
        disc = GitHubDiscovery.__new__(GitHubDiscovery)
        disc.session = self._mock_session([self.GITHUB_ITEM])
        results = disc.by_topic("machine-learning", max_results=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].full_name, "owner/repo")

    def test_by_language_calls_search(self):
        disc = GitHubDiscovery.__new__(GitHubDiscovery)
        disc.session = self._mock_session([self.GITHUB_ITEM])
        results = disc.by_language("python", max_results=5)
        self.assertEqual(len(results), 1)

    def test_by_watchlist_fetches_each(self):
        disc = GitHubDiscovery.__new__(GitHubDiscovery)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = self.GITHUB_ITEM
        disc.session = MagicMock()
        disc.session.get.return_value = mock_resp
        results = disc.by_watchlist(["owner/repo"])
        self.assertEqual(len(results), 1)

    def test_discover_deduplicates(self):
        disc = GitHubDiscovery.__new__(GitHubDiscovery)
        # Both topic and language return the same repo
        disc.session = self._mock_session([self.GITHUB_ITEM])
        with patch.object(disc, "by_topic", return_value=[GitHubDiscovery._parse_repo(self.GITHUB_ITEM)]), \
             patch.object(disc, "by_language", return_value=[GitHubDiscovery._parse_repo(self.GITHUB_ITEM)]), \
             patch.object(disc, "by_watchlist", return_value=[]):
            results = disc.discover(topics=["ai"], languages=["python"], max_total=10)
        self.assertEqual(len(results), 1)

    def test_discover_excludes_repos(self):
        disc = GitHubDiscovery.__new__(GitHubDiscovery)
        with patch.object(disc, "by_topic", return_value=[GitHubDiscovery._parse_repo(self.GITHUB_ITEM)]), \
             patch.object(disc, "by_language", return_value=[]), \
             patch.object(disc, "by_watchlist", return_value=[]):
            results = disc.discover(topics=["ai"], exclude=["owner/repo"])
        self.assertEqual(len(results), 0)

    def test_search_failure_returns_empty(self):
        disc = GitHubDiscovery.__new__(GitHubDiscovery)
        disc.session = MagicMock()
        disc.session.get.side_effect = Exception("network error")
        results = disc._search("topic:ai", 5)
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# ScanReport
# ---------------------------------------------------------------------------

class TestScanReport(unittest.TestCase):
    def test_summarize_contains_key_fields(self):
        report = make_report(n_repos=2)
        summary = report.summarize()
        self.assertIn("RepoSentinel Scan Report", summary)
        self.assertIn("owner/repo-0", summary)
        self.assertIn("owner/repo-1", summary)

    def test_repo_scan_result_counts(self):
        rr = make_scan_result(n_findings=3)
        self.assertEqual(rr.finding_count, 3)
        # All findings are "high" severity from make_finding
        self.assertEqual(rr.high_count, 3)
        self.assertEqual(rr.critical_count, 0)


# ---------------------------------------------------------------------------
# CrossRepoScanner
# ---------------------------------------------------------------------------

class TestCrossRepoScanner(unittest.TestCase):
    def _make_scanner(self, tmp: str) -> CrossRepoScanner:
        scanner = CrossRepoScanner.__new__(CrossRepoScanner)
        scanner.workspace = Path(tmp)
        scanner.commits_window = 10
        scanner.cleanup = False
        scanner.max_repos = 5
        return scanner

    def test_scan_returns_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            scanner = self._make_scanner(tmp)
            mock_result = make_scan_result(n_findings=1)
            with patch.object(scanner, "_scan_one", return_value=mock_result):
                report = scanner.scan([make_target()], scan_id="test-001")
        self.assertEqual(report.scan_id, "test-001")
        self.assertEqual(len(report.repo_results), 1)
        self.assertEqual(report.total_findings, 1)

    def test_scan_respects_max_repos(self):
        with tempfile.TemporaryDirectory() as tmp:
            scanner = self._make_scanner(tmp)
            scanner.max_repos = 2
            mock_result = make_scan_result(n_findings=0)
            targets = [make_target(full_name=f"owner/repo-{i}") for i in range(5)]
            with patch.object(scanner, "_scan_one", return_value=mock_result):
                report = scanner.scan(targets)
        self.assertEqual(len(report.repo_results), 2)

    def test_scan_one_handles_git_error(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            scanner = self._make_scanner(tmp)
            with patch("agent.scanner._clone_or_update",
                       side_effect=subprocess.CalledProcessError(1, "git", stderr=b"auth error")):
                result = scanner._scan_one(make_target())
        self.assertIsNotNone(result.error)
        self.assertIn("git error", result.error)

    def test_scan_one_handles_generic_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            scanner = self._make_scanner(tmp)
            with patch("agent.scanner._clone_or_update", side_effect=RuntimeError("disk full")):
                result = scanner._scan_one(make_target())
        self.assertIn("disk full", result.error)


# ---------------------------------------------------------------------------
# MultiReporter
# ---------------------------------------------------------------------------

class TestMultiReporter(unittest.TestCase):
    def test_write_json_report(self):
        from agent.multi_reporter import write_json_report
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "findings.json"
            report = make_report(n_repos=2)
            write_json_report(report, path)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text())
            self.assertEqual(data["scan_id"], "test-abc")
            self.assertEqual(len(data["repos"]), 2)
            self.assertIn("findings", data["repos"][0])

    def test_write_json_report_structure(self):
        from agent.multi_reporter import write_json_report
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "findings.json"
            write_json_report(make_report(1), path)
            data = json.loads(path.read_text())
            self.assertIn("summary", data)
            self.assertIn("total_findings", data["summary"])

    def test_write_markdown_report(self):
        from agent.multi_reporter import write_markdown_report
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            write_markdown_report(make_report(2), path)
            self.assertTrue(path.exists())
            content = path.read_text()
            self.assertIn("RepoSentinel Scan Report", content)
            self.assertIn("owner/repo-0", content)

    def test_markdown_includes_finding_details(self):
        from agent.multi_reporter import write_markdown_report
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            write_markdown_report(make_report(1), path)
            content = path.read_text()
            self.assertIn("Issue 0", content)

    def test_post_dashboard_issue_skipped_without_config(self):
        from agent.multi_reporter import post_dashboard_issue
        env = {k: v for k, v in os.environ.items()
               if k not in ("GITHUB_TOKEN", "SENTINEL_DASHBOARD_REPO")}
        with patch.dict(os.environ, env, clear=True):
            result = post_dashboard_issue(make_report())
        self.assertIsNone(result)

    def test_report_all_creates_files(self):
        from agent.multi_reporter import report_all
        with tempfile.TemporaryDirectory() as tmp:
            outputs = report_all(make_report(2), output_dir=Path(tmp), post_to_github=False)
            self.assertTrue(Path(outputs["json_report"]).exists())
            self.assertTrue(Path(outputs["markdown_report"]).exists())
            self.assertIsNone(outputs["dashboard_issue"])


if __name__ == "__main__":
    unittest.main()

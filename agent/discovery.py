"""
RepoSentinel · discovery.py

Discovers open-source GitHub repositories to scan using the GitHub Search API.

Three discovery strategies, composable and filterable:
  1. topic      — repos tagged with a specific topic (e.g. "machine-learning")
  2. language   — top repos in a language (e.g. "python")
  3. watchlist  — explicit owner/repo strings from config

Results are deduplicated, scored by a priority heuristic, and returned as
a ranked list of RepoTarget objects ready for the scanner.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Iterator

import requests


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RepoTarget:
    full_name: str          # "owner/repo"
    clone_url: str
    default_branch: str
    stars: int
    language: str | None
    topics: list[str]
    open_issues: int
    last_pushed: str        # ISO 8601
    priority_score: float = 0.0   # computed heuristic

    @property
    def owner(self) -> str:
        return self.full_name.split("/")[0]

    @property
    def name(self) -> str:
        return self.full_name.split("/")[1]


# ---------------------------------------------------------------------------
# GitHub Search client
# ---------------------------------------------------------------------------

class GitHubDiscovery:
    """
    Wraps GitHub's Search and Repository APIs for repo discovery.

    Rate limits (unauthenticated): 10 req/min search, 60 req/min rest
    Rate limits (authenticated):   30 req/min search, 5000 req/hr rest
    Always use a token for production.
    """

    SEARCH_URL = "https://api.github.com/search/repositories"
    REPOS_URL  = "https://api.github.com/repos"

    def __init__(self, token: str | None = None):
        self.session = requests.Session()
        token = token or os.getenv("GITHUB_TOKEN")
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.session.headers["Accept"] = "application/vnd.github+json"
        self.session.headers["X-GitHub-Api-Version"] = "2022-11-28"

    # ------------------------------------------------------------------
    # Strategy 1 — topic search
    # ------------------------------------------------------------------

    def by_topic(
        self,
        topic: str,
        language: str | None = None,
        min_stars: int = 100,
        max_results: int = 20,
    ) -> list[RepoTarget]:
        """Find repos tagged with a GitHub topic."""
        q = f"topic:{topic} stars:>={min_stars} fork:false archived:false"
        if language:
            q += f" language:{language}"
        return self._search(q, max_results)

    # ------------------------------------------------------------------
    # Strategy 2 — language search
    # ------------------------------------------------------------------

    def by_language(
        self,
        language: str,
        min_stars: int = 500,
        max_results: int = 20,
        pushed_within_days: int = 90,
    ) -> list[RepoTarget]:
        """Find actively maintained repos in a given language."""
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(days=pushed_within_days))
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        q = (
            f"language:{language} stars:>={min_stars} "
            f"pushed:>{cutoff_str} fork:false archived:false"
        )
        return self._search(q, max_results)

    # ------------------------------------------------------------------
    # Strategy 3 — explicit watchlist
    # ------------------------------------------------------------------

    def by_watchlist(self, repos: list[str]) -> list[RepoTarget]:
        """
        Fetch metadata for an explicit list of 'owner/repo' strings.
        Useful for repos you personally contribute to or monitor.
        """
        targets = []
        for full_name in repos:
            try:
                resp = self.session.get(f"{self.REPOS_URL}/{full_name}")
                resp.raise_for_status()
                targets.append(self._parse_repo(resp.json()))
                time.sleep(0.1)   # polite pacing
            except Exception as exc:
                print(f"[Discovery] Could not fetch {full_name}: {exc}")
        return targets

    # ------------------------------------------------------------------
    # Composite discovery with priority ranking
    # ------------------------------------------------------------------

    def discover(
        self,
        topics: list[str] | None = None,
        languages: list[str] | None = None,
        watchlist: list[str] | None = None,
        min_stars: int = 100,
        max_per_strategy: int = 15,
        max_total: int = 30,
        exclude: list[str] | None = None,
    ) -> list[RepoTarget]:
        """
        Run all enabled strategies, deduplicate, score, and rank results.

        Priority heuristic:
          score = log10(stars) * recency_factor * issue_density_factor

        Repos with many stars, recent activity, and open issues score highest
        — they're active, impactful, and likely to benefit from a scan.
        """
        seen: dict[str, RepoTarget] = {}
        exclude_set = set(exclude or [])

        for topic in (topics or []):
            for repo in self.by_topic(topic, min_stars=min_stars, max_results=max_per_strategy):
                if repo.full_name not in exclude_set:
                    seen[repo.full_name] = repo
            time.sleep(1.5)   # respect search rate limit

        for lang in (languages or []):
            for repo in self.by_language(lang, min_stars=min_stars, max_results=max_per_strategy):
                if repo.full_name not in exclude_set:
                    seen[repo.full_name] = repo
            time.sleep(1.5)

        for repo in self.by_watchlist(watchlist or []):
            if repo.full_name not in exclude_set:
                seen[repo.full_name] = repo

        ranked = sorted(seen.values(), key=self._priority_score, reverse=True)
        for r in ranked:
            r.priority_score = self._priority_score(r)

        return ranked[:max_total]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(self, query: str, max_results: int) -> list[RepoTarget]:
        targets = []
        per_page = min(max_results, 30)
        try:
            resp = self.session.get(
                self.SEARCH_URL,
                params={"q": query, "sort": "stars", "order": "desc", "per_page": per_page},
                timeout=15,
            )
            resp.raise_for_status()
            for item in resp.json().get("items", [])[:max_results]:
                targets.append(self._parse_repo(item))
        except Exception as exc:
            print(f"[Discovery] Search failed: {exc}")
        return targets

    @staticmethod
    def _parse_repo(item: dict) -> RepoTarget:
        return RepoTarget(
            full_name     = item["full_name"],
            clone_url     = item["clone_url"],
            default_branch= item.get("default_branch", "main"),
            stars         = item.get("stargazers_count", 0),
            language      = item.get("language"),
            topics        = item.get("topics", []),
            open_issues   = item.get("open_issues_count", 0),
            last_pushed   = item.get("pushed_at", ""),
        )

    @staticmethod
    def _priority_score(repo: RepoTarget) -> float:
        import math
        from datetime import datetime, timezone
        # Star weight
        star_score = math.log10(max(repo.stars, 1))
        # Recency (decay over 180 days)
        recency = 1.0
        if repo.last_pushed:
            try:
                pushed = datetime.fromisoformat(repo.last_pushed.replace("Z", "+00:00"))
                days_old = (datetime.now(timezone.utc) - pushed).days
                recency = max(0.1, 1.0 - days_old / 180)
            except Exception:
                pass
        # Open issue density (more issues = more opportunity)
        issue_factor = min(1.5, 1.0 + repo.open_issues / 500)
        return round(star_score * recency * issue_factor, 4)

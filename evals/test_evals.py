"""
Tests for the eval harness (run_evals.py).

These test the matching and scoring logic only — no real API calls.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.analyzer import Finding
from evals.run_evals import (
    EvalCase, MatchResult,
    _finding_text, _matches_expected,
    match_findings, compute_scores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_finding(**kwargs) -> Finding:
    defaults = dict(
        category="security", severity="critical",
        file="db.py", line_start=2, line_end=2,
        title="SQL injection via f-string interpolation",
        description="User input directly interpolated into SQL query string.",
        suggestion="Use parameterised queries instead of f-strings.",
        confidence=0.95,
    )
    defaults.update(kwargs)
    return Finding(**defaults)


def make_case(expected_findings=None, code="x = 1", id="test-001") -> EvalCase:
    return EvalCase(
        id=id,
        description="Test case",
        language="python",
        filename="db.py",
        code=code,
        expected_findings=expected_findings or [],
        notes="",
    )


EXPECTED_SQL = {
    "category": "security",
    "severity": "critical",
    "line_range": [2, 2],
    "keywords": ["sql injection", "f-string", "parameterised"],
}


# ---------------------------------------------------------------------------
# _matches_expected
# ---------------------------------------------------------------------------

class TestMatchesExpected(unittest.TestCase):
    def test_keyword_match(self):
        f = make_finding(title="SQL injection via f-string")
        self.assertTrue(_matches_expected(f, EXPECTED_SQL))

    def test_keyword_in_description(self):
        f = make_finding(
            title="Unsafe query",
            description="This is an sql injection vulnerability via parameterised query misuse.",
        )
        self.assertTrue(_matches_expected(f, EXPECTED_SQL))

    def test_no_keyword_match(self):
        f = make_finding(
            title="Missing null check",
            description="Variable may be null.",
            suggestion="Add a null guard.",
        )
        self.assertFalse(_matches_expected(f, EXPECTED_SQL))

    def test_wrong_category(self):
        f = make_finding(
            title="SQL injection risk",
            description="parameterised queries not used.",
            category="bug",   # expected is "security"
        )
        self.assertFalse(_matches_expected(f, EXPECTED_SQL))

    def test_case_insensitive(self):
        f = make_finding(title="SQL INJECTION VIA F-STRING")
        self.assertTrue(_matches_expected(f, EXPECTED_SQL))

    def test_no_category_in_expected_skips_category_check(self):
        expected = {"keywords": ["sql injection"], "line_range": [2, 2]}
        f = make_finding(title="sql injection found", category="bug")
        self.assertTrue(_matches_expected(f, expected))


# ---------------------------------------------------------------------------
# match_findings
# ---------------------------------------------------------------------------

class TestMatchFindings(unittest.TestCase):
    def test_true_positive(self):
        case = make_case(expected_findings=[EXPECTED_SQL])
        findings = [make_finding(title="SQL injection via f-string interpolation")]
        results, fps = match_findings(case, findings)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].is_tp)
        self.assertFalse(results[0].is_fn)
        self.assertEqual(len(fps), 0)

    def test_false_negative(self):
        case = make_case(expected_findings=[EXPECTED_SQL])
        findings = []   # agent found nothing
        results, fps = match_findings(case, findings)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].is_tp)
        self.assertTrue(results[0].is_fn)

    def test_false_positive(self):
        case = make_case(expected_findings=[EXPECTED_SQL])
        findings = [
            make_finding(title="SQL injection via f-string"),   # TP
            make_finding(title="Unrelated style issue",         # FP
                         category="style", description="x", suggestion="y"),
        ]
        results, fps = match_findings(case, findings)
        self.assertEqual(len([r for r in results if r.is_tp]), 1)
        self.assertEqual(len(fps), 1)

    def test_clean_case_no_false_positives(self):
        case = make_case(expected_findings=[])
        findings = []
        results, fps = match_findings(case, findings)
        self.assertEqual(results, [])
        self.assertEqual(fps, [])

    def test_each_finding_matched_once(self):
        """One finding should not match two expected items."""
        expected = [EXPECTED_SQL, EXPECTED_SQL.copy()]   # same bug listed twice
        case = make_case(expected_findings=expected)
        findings = [make_finding(title="SQL injection via f-string")]
        results, fps = match_findings(case, findings)
        tps = [r for r in results if r.is_tp]
        fns = [r for r in results if r.is_fn]
        self.assertEqual(len(tps), 1)
        self.assertEqual(len(fns), 1)

    def test_multiple_expected_bugs(self):
        expected_null = {
            "category": "bug",
            "keywords": ["null", "none", "not found"],
            "line_range": [3, 3],
        }
        case = make_case(expected_findings=[EXPECTED_SQL, expected_null])
        findings = [
            make_finding(title="SQL injection", description="parameterised fix needed."),
            make_finding(title="Null dereference", category="bug",
                         description="value may be None here.", suggestion="check for none"),
        ]
        results, fps = match_findings(case, findings)
        self.assertEqual(len([r for r in results if r.is_tp]), 2)


# ---------------------------------------------------------------------------
# compute_scores
# ---------------------------------------------------------------------------

class TestComputeScores(unittest.TestCase):
    def _mr(self, is_tp: bool) -> MatchResult:
        return MatchResult("x", {}, None, is_tp=is_tp, is_fn=not is_tp)

    def test_perfect_score(self):
        results = [self._mr(True)] * 5
        fps = []
        p, r, f1 = compute_scores(results, fps)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f1, 1.0)

    def test_all_missed(self):
        results = [self._mr(False)] * 5
        fps = []
        p, r, f1 = compute_scores(results, fps)
        self.assertEqual(r, 0.0)
        self.assertEqual(f1, 0.0)

    def test_precision_tradeoff(self):
        # 3 TPs, 0 FNs, 7 FPs → low precision, high recall
        results = [self._mr(True)] * 3
        fps = [make_finding() for _ in range(7)]
        p, r, f1 = compute_scores(results, fps)
        self.assertAlmostEqual(p, 0.3, places=1)
        self.assertAlmostEqual(r, 1.0)

    def test_recall_tradeoff(self):
        # 3 TPs, 7 FNs, 0 FPs → high precision, low recall
        results = [self._mr(True)] * 3 + [self._mr(False)] * 7
        fps = []
        p, r, f1 = compute_scores(results, fps)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 0.3, places=1)

    def test_empty_results(self):
        p, r, f1 = compute_scores([], [])
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f1, 0.0)

    def test_f1_harmonic_mean(self):
        # precision=0.5, recall=1.0 → f1 = 2*0.5*1.0/(0.5+1.0) = 0.667
        results = [self._mr(True)]
        fps = [make_finding()]
        p, r, f1 = compute_scores(results, fps)
        self.assertAlmostEqual(f1, 2/3, places=2)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

class TestDatasetLoading(unittest.TestCase):
    def test_golden_dataset_loads(self):
        dataset_path = Path("evals/golden_dataset.json")
        self.assertTrue(dataset_path.exists(), "golden_dataset.json missing")
        cases = [EvalCase(**row) for row in json.loads(dataset_path.read_text())]
        self.assertGreater(len(cases), 0)

    def test_all_cases_have_required_fields(self):
        cases = [EvalCase(**row) for row in
                 json.loads(Path("evals/golden_dataset.json").read_text())]
        for case in cases:
            self.assertTrue(case.id, f"Missing id")
            self.assertTrue(case.filename)
            self.assertTrue(case.code)
            self.assertIsInstance(case.expected_findings, list)

    def test_expected_findings_have_keywords(self):
        cases = [EvalCase(**row) for row in
                 json.loads(Path("evals/golden_dataset.json").read_text())]
        for case in cases:
            for ef in case.expected_findings:
                self.assertIn("keywords", ef,
                    f"Case {case.id} missing keywords in expected_findings")
                self.assertGreater(len(ef["keywords"]), 0,
                    f"Case {case.id} has empty keywords list")

    def test_clean_cases_have_empty_expected(self):
        cases = [EvalCase(**row) for row in
                 json.loads(Path("evals/golden_dataset.json").read_text())]
        clean = [c for c in cases if c.id.startswith("clean-")]
        self.assertGreater(len(clean), 0, "No clean cases found")
        for case in clean:
            self.assertEqual(case.expected_findings, [],
                f"Clean case {case.id} should have no expected findings")

    def test_buggy_cases_have_expected_findings(self):
        cases = [EvalCase(**row) for row in
                 json.loads(Path("evals/golden_dataset.json").read_text())]
        buggy = [c for c in cases if not c.id.startswith("clean-")]
        for case in buggy:
            self.assertGreater(len(case.expected_findings), 0,
                f"Bug case {case.id} should have at least one expected finding")


if __name__ == "__main__":
    unittest.main()

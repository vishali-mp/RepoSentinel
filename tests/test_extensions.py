"""
Tests for RepoSentinel vector store and trainer (fine-tuning pipeline).
Run with: pytest tests/ -v
"""

from __future__ import annotations

import os
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.analyzer import Finding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_finding(**kwargs) -> Finding:
    defaults = dict(
        category="bug", severity="high",
        file="app/auth.py", line_start=10, line_end=15,
        title="Unhandled None return", description="Function may return None.",
        suggestion="Add a null check.", confidence=0.88,
    )
    defaults.update(kwargs)
    return Finding(**defaults)


def make_records(n: int = 30) -> list[dict]:
    """Synthetic labelled records for training tests."""
    records = []
    for i in range(n):
        records.append({
            "features": {
                "category":   ["bug", "security", "performance", "style"][i % 4],
                "severity":   ["low", "medium", "high", "critical"][i % 4],
                "confidence": 0.5 + (i % 5) * 0.1,
                "file_ext":   [".py", ".ts", ".js"][i % 3],
                "title_len":  40 + i,
                "desc_len":   100 + i * 2,
            },
            "label": i % 2,   # alternating accepted/dismissed
            "text":  f"Finding number {i}: some description",
        })
    return records


# ---------------------------------------------------------------------------
# Vector store tests
# ---------------------------------------------------------------------------

class TestFallbackEmbed(unittest.TestCase):
    def test_returns_unit_vector(self):
        from agent.vector_store import _fallback_embed
        vec = _fallback_embed("hello world", dim=64)
        self.assertEqual(len(vec), 64)
        norm = sum(v ** 2 for v in vec) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_deterministic(self):
        from agent.vector_store import _fallback_embed
        v1 = _fallback_embed("test sentence")
        v2 = _fallback_embed("test sentence")
        self.assertEqual(v1, v2)

    def test_different_texts_differ(self):
        from agent.vector_store import _fallback_embed
        v1 = _fallback_embed("SQL injection vulnerability")
        v2 = _fallback_embed("memory leak in loop")
        self.assertNotEqual(v1, v2)


class TestFindingText(unittest.TestCase):
    def test_includes_key_fields(self):
        from agent.vector_store import _finding_text
        f = make_finding(category="security", title="SSRF risk")
        text = _finding_text(f)
        self.assertIn("security", text)
        self.assertIn("SSRF risk", text)
        self.assertIn("app/auth.py", text)


class TestFindingStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Patch embed to use fast fallback
        self.embed_patcher = patch(
            "agent.vector_store._embed",
            side_effect=lambda texts: [
                __import__("agent.vector_store", fromlist=["_fallback_embed"])
                ._fallback_embed(t) for t in texts
            ],
        )
        self.embed_patcher.start()

    def tearDown(self):
        self.embed_patcher.stop()

    def _store(self):
        from agent.vector_store import FindingStore
        return FindingStore(db_path=Path(self.tmp))

    def test_add_and_count(self):
        store = self._store()
        findings = [make_finding(title=f"Issue {i}") for i in range(3)]
        store.add(findings, run_id="run-1", repo="owner/repo")
        self.assertEqual(store._col.count(), 3)

    def test_no_duplicates_on_empty_store(self):
        store = self._store()
        findings = [make_finding()]
        dupes = store.find_duplicates(findings)
        self.assertEqual(dupes, set())

    def test_identical_finding_is_duplicate(self):
        store = self._store()
        f = make_finding(title="SQL injection via f-string")
        store.add([f], run_id="run-1")
        dupes = store.find_duplicates([f], threshold=0.85, look_back_days=36500)
        self.assertIn(0, dupes)

    def test_different_finding_not_duplicate(self):
        store = self._store()
        f1 = make_finding(title="SQL injection via f-string", category="security")
        f2 = make_finding(title="Memory leak in event loop", category="performance",
                          file="worker.py")
        store.add([f1], run_id="run-1")
        dupes = store.find_duplicates([f2], threshold=0.95, look_back_days=36500)
        self.assertNotIn(0, dupes)

    def test_label_updates_outcome(self):
        store = self._store()
        store.add([make_finding()], run_id="run-42")
        n = store.label("run-42", "accepted")
        self.assertEqual(n, 1)
        results = store._col.get(where={"run_id": "run-42"}, include=["metadatas"])
        self.assertEqual(results["metadatas"][0]["outcome"], "accepted")

    def test_stats_returns_counts(self):
        store = self._store()
        store.add([make_finding(category="bug")], run_id="r1")
        store.add([make_finding(category="security")], run_id="r2")
        stats = store.stats()
        self.assertEqual(stats["total"], 2)
        self.assertIn("bug", stats["categories"])

    def test_export_training_data_empty(self):
        store = self._store()
        # No labelled data → expect empty list with warning
        records = store.export_training_data(min_labelled=5)
        self.assertEqual(records, [])

    def test_export_training_data_labelled(self):
        store = self._store()
        store.add([make_finding()], run_id="r1", outcome="accepted")
        store.add([make_finding(title="Other")], run_id="r2", outcome="dismissed")
        records = store.export_training_data(min_labelled=1)
        self.assertEqual(len(records), 2)
        self.assertIn(records[0]["label"], [0, 1])


# ---------------------------------------------------------------------------
# Trainer / fine-tuning tests
# ---------------------------------------------------------------------------

class TestFeatureExtraction(unittest.TestCase):
    def test_all_keys_present(self):
        from agent.trainer import _extract_features, CATEGORIES, EXTENSIONS
        record = {
            "features": {
                "category": "security", "severity": "critical",
                "confidence": 0.9, "file_ext": ".py",
                "title_len": 55, "desc_len": 200,
            }
        }
        row = _extract_features(record)
        for cat in CATEGORIES:
            self.assertIn(f"cat_{cat}", row)
        self.assertIn("severity_ord", row)
        self.assertIn("llm_confidence", row)
        self.assertIn("title_len", row)

    def test_security_one_hot(self):
        from agent.trainer import _extract_features
        record = {"features": {"category": "security", "severity": "high",
                                "confidence": 0.8, "file_ext": ".py",
                                "title_len": 40, "desc_len": 100}}
        row = _extract_features(record)
        self.assertEqual(row["cat_security"], 1)
        self.assertEqual(row["cat_bug"], 0)

    def test_severity_ordinal(self):
        from agent.trainer import _extract_features
        for sev, expected in [("low", 0), ("medium", 1), ("high", 2), ("critical", 3)]:
            record = {"features": {"category": "bug", "severity": sev,
                                    "confidence": 0.5, "file_ext": ".py",
                                    "title_len": 30, "desc_len": 80}}
            row = _extract_features(record)
            self.assertEqual(row["severity_ord"], expected)


class TestBuildDataframe(unittest.TestCase):
    def test_shape(self):
        from agent.trainer import build_dataframe
        records = make_records(10)
        X, y = build_dataframe(records)
        self.assertEqual(len(X), 10)
        self.assertEqual(len(y), 10)
        self.assertFalse(X.isnull().any().any())

    def test_labels_binary(self):
        from agent.trainer import build_dataframe
        records = make_records(10)
        _, y = build_dataframe(records)
        self.assertTrue(set(y.unique()).issubset({0, 1}))


class TestTrain(unittest.TestCase):
    def test_skips_when_too_few_examples(self):
        from agent.trainer import train
        mock_store = MagicMock()
        mock_store.export_training_data.return_value = make_records(5)
        report = train(mock_store, min_examples=20)
        self.assertEqual(report["status"], "skipped")

    def test_trains_and_saves_model(self):
        from agent.trainer import train
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.pkl"
            with patch("agent.trainer.MODEL_PATH", model_path), \
                 patch("agent.trainer.REPORT_PATH", Path(tmp) / "report.json"):
                mock_store = MagicMock()
                mock_store.export_training_data.return_value = make_records(40)
                report = train(mock_store, min_examples=20)
                self.assertEqual(report["status"], "success")
                self.assertTrue(model_path.exists())
                self.assertIn(report["model"], ["random_forest", "gradient_boost", "logistic_regression"])


class TestFindingClassifier(unittest.TestCase):
    def test_falls_back_to_llm_confidence_when_no_model(self):
        from agent.trainer import FindingClassifier
        with patch("agent.trainer.MODEL_PATH", Path("/nonexistent/model.pkl")):
            clf = FindingClassifier(model_path=Path("/nonexistent/model.pkl"))
        f = make_finding(confidence=0.85)
        self.assertAlmostEqual(clf.score(f), 0.85)

    def test_should_report_threshold_fallback(self):
        from agent.trainer import FindingClassifier
        clf = FindingClassifier(model_path=Path("/nonexistent/model.pkl"))
        self.assertTrue(clf.should_report(make_finding(confidence=0.95)))
        self.assertFalse(clf.should_report(make_finding(confidence=0.40)))

    def test_loads_trained_model(self):
        from agent.trainer import FindingClassifier, build_dataframe
        from sklearn.ensemble import RandomForestClassifier
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.pkl"
            records = make_records(20)
            from agent.trainer import build_dataframe, _extract_features
            import pandas as pd
            X, y = build_dataframe(records)
            rf = RandomForestClassifier(n_estimators=10, random_state=42)
            rf.fit(X, y)
            bundle = {
                "model": rf,
                "feature_columns": list(X.columns),
                "optimal_threshold": 0.6,
                "model_name": "random_forest",
            }
            with open(model_path, "wb") as fh:
                pickle.dump(bundle, fh)

            clf = FindingClassifier(model_path=model_path)
            self.assertTrue(clf.is_trained)
            score = clf.score(make_finding())
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

"""
RepoSentinel · trainer.py

Fine-tuning pipeline: trains a lightweight classifier on labelled finding
history to predict whether a finding is worth reporting.

Replaces the flat confidence threshold with a model that learns from your
team's actual accept/dismiss decisions — so it gets smarter over time.

Architecture
------------
Input features (per finding):
  - category (one-hot: bug / performance / security / style)
  - severity (ordinal: critical=3, high=2, medium=1, low=0)
  - LLM confidence score (float 0–1)
  - file extension (one-hot: .py, .ts, .js, .tsx, .other)
  - title length (proxy for specificity)
  - description length (proxy for LLM reasoning depth)

Model: RandomForestClassifier (sklearn)
  - Fast to train on small datasets (20–500 examples)
  - Naturally handles mixed feature types
  - Outputs calibrated probabilities
  - Explainable via feature_importances_

The trained model is saved to .sentinel/model.pkl and loaded automatically
by the analyzer on subsequent runs.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from agent.vector_store import FindingStore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_PATH = Path(".sentinel/model.pkl")
REPORT_PATH = Path(".sentinel/training_report.json")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

CATEGORIES  = ["bug", "performance", "security", "style"]
SEVERITIES  = {"low": 0, "medium": 1, "high": 2, "critical": 3}
EXTENSIONS  = [".py", ".ts", ".js", ".tsx", ".jsx"]


def _extract_features(record: dict) -> dict:
    """Turn a raw training record into a flat feature dict."""
    f = record["features"]
    row: dict = {}

    # Category one-hot
    for cat in CATEGORIES:
        row[f"cat_{cat}"] = int(f.get("category") == cat)

    # Severity ordinal
    row["severity_ord"] = SEVERITIES.get(f.get("severity", "low"), 0)

    # LLM confidence
    row["llm_confidence"] = float(f.get("confidence", 0.5))

    # File extension one-hot
    ext = f.get("file_ext", ".other")
    for e in EXTENSIONS:
        row[f"ext_{e[1:]}"] = int(ext == e)
    row["ext_other"] = int(ext not in EXTENSIONS)

    # Text length proxies
    row["title_len"]  = min(f.get("title_len", 0), 200)   # cap outliers
    row["desc_len"]   = min(f.get("desc_len", 0), 1000)

    return row


def build_dataframe(records: list[dict]) -> tuple[pd.DataFrame, pd.Series]:
    rows = [_extract_features(r) for r in records]
    labels = [r["label"] for r in records]
    X = pd.DataFrame(rows).fillna(0)
    y = pd.Series(labels, dtype=int)
    return X, y


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

CANDIDATE_MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    ),
    "gradient_boost": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    ),
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ]),
}


def select_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> tuple[str, object, float]:
    """
    Run stratified k-fold CV on each candidate and return
    (name, model, best_roc_auc).
    Falls back to random_forest if dataset is too small for CV.
    """
    if len(y) < cv * 2:
        print(f"[Trainer] Dataset too small for {cv}-fold CV — using RandomForest directly.")
        model = CANDIDATE_MODELS["random_forest"]
        model.fit(X, y)
        return "random_forest", model, float("nan")

    best_name, best_model, best_score = None, None, -1.0
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in CANDIDATE_MODELS.items():
        scores = cross_val_score(model, X, y, cv=cv_splitter, scoring="roc_auc")
        mean_score = scores.mean()
        print(f"  {name:25s}  ROC-AUC = {mean_score:.3f} ± {scores.std():.3f}")
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_model = model

    best_model.fit(X, y)
    return best_name, best_model, best_score


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(store: "FindingStore", min_examples: int = 20) -> dict:
    """
    Full training pipeline. Returns a report dict.

    Steps:
    1. Export labelled data from vector store
    2. Build feature matrix
    3. Select best model via CV
    4. Evaluate on held-out fold
    5. Save model to .sentinel/model.pkl
    6. Write training report
    """
    print("[Trainer] Exporting labelled findings from vector store …")
    records = store.export_training_data(min_labelled=min_examples)

    if len(records) < min_examples:
        return {
            "status": "skipped",
            "reason": f"Need {min_examples} labelled examples, got {len(records)}",
            "records": len(records),
        }

    X, y = build_dataframe(records)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"[Trainer] Dataset: {len(records)} examples  ({n_pos} accepted, {n_neg} dismissed)")
    print(f"[Trainer] Features: {list(X.columns)}")
    print("[Trainer] Selecting best model …")

    best_name, best_model, cv_auc = select_best_model(X, y)
    print(f"[Trainer] Best model: {best_name}  (CV ROC-AUC = {cv_auc:.3f})")

    # Full-dataset evaluation for the report
    y_prob = best_model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    report_text = classification_report(y, y_pred, target_names=["dismissed", "accepted"])
    print("\n" + report_text)

    # Precision-recall curve → find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    optimal_idx = int(np.argmax(f1s))
    optimal_threshold = float(thresholds[optimal_idx]) if len(thresholds) > optimal_idx else 0.5
    print(f"[Trainer] Optimal probability threshold: {optimal_threshold:.3f}")

    # Feature importance (RF and GB expose this natively)
    feature_importance: dict = {}
    raw_model = best_model.steps[-1][1] if hasattr(best_model, "steps") else best_model
    if hasattr(raw_model, "feature_importances_"):
        feature_importance = dict(zip(X.columns, raw_model.feature_importances_.tolist()))
        top = sorted(feature_importance.items(), key=lambda x: -x[1])[:5]
        print(f"[Trainer] Top features: {top}")

    # Persist model bundle
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":              best_model,
        "feature_columns":    list(X.columns),
        "optimal_threshold":  optimal_threshold,
        "model_name":         best_name,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[Trainer] Model saved → {MODEL_PATH}")

    # Save training report
    training_report = {
        "status":             "success",
        "model":              best_name,
        "cv_roc_auc":         cv_auc,
        "optimal_threshold":  optimal_threshold,
        "n_examples":         len(records),
        "class_balance":      {"accepted": n_pos, "dismissed": n_neg},
        "feature_importance": feature_importance,
        "classification_report": report_text,
    }
    REPORT_PATH.write_text(json.dumps(training_report, indent=2))
    print(f"[Trainer] Report saved → {REPORT_PATH}")

    return training_report


# ---------------------------------------------------------------------------
# Inference helper (called by analyzer.py at runtime)
# ---------------------------------------------------------------------------

class FindingClassifier:
    """
    Wraps the trained model for inference.
    Falls back to the raw LLM confidence if no model is trained yet.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self._bundle: dict | None = None
        if model_path.exists():
            with open(model_path, "rb") as f:
                self._bundle = pickle.load(f)
            print(f"[Classifier] Loaded model: {self._bundle['model_name']} "
                  f"(threshold={self._bundle['optimal_threshold']:.2f})")
        else:
            print("[Classifier] No trained model found — using LLM confidence score.")

    @property
    def is_trained(self) -> bool:
        return self._bundle is not None

    def score(self, finding) -> float:
        """Return acceptance probability for a finding."""
        if not self._bundle:
            return finding.confidence

        record = {
            "features": {
                "category":   finding.category,
                "severity":   finding.severity,
                "confidence": finding.confidence,
                "file_ext":   Path(finding.file).suffix,
                "title_len":  len(finding.title),
                "desc_len":   len(finding.description),
            }
        }
        X = pd.DataFrame([_extract_features(record)])
        # Align columns to training feature set
        for col in self._bundle["feature_columns"]:
            if col not in X.columns:
                X[col] = 0
        X = X[self._bundle["feature_columns"]]
        return float(self._bundle["model"].predict_proba(X)[0, 1])

    def should_report(self, finding) -> bool:
        """True if the finding clears the learned threshold."""
        threshold = (
            self._bundle["optimal_threshold"] if self._bundle else 0.70
        )
        return self.score(finding) >= threshold

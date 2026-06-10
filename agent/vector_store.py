"""
RepoSentinel · vector_store.py

Semantic finding memory using ChromaDB + Anthropic embeddings.

Replaces the brittle SHA-256 fingerprint deduplication with true semantic
similarity search — so "SQL injection via f-string" and "unsanitised query
string concatenation" are correctly identified as the same class of issue,
even if the titles differ.

Also persists a full history of every finding ever surfaced, enabling:
  - Trend analysis (is bug density going up or down?)
  - Training data export for the fine-tuning pipeline
  - Recall of similar past issues during LLM analysis
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings

if TYPE_CHECKING:
    from agent.analyzer import Finding

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = Path(os.getenv("SENTINEL_DB_PATH", ".sentinel/chroma"))
SIMILARITY_THRESHOLD = float(os.getenv("SENTINEL_SIM_THRESHOLD", "0.92"))
COLLECTION_NAME = "findings"


# ---------------------------------------------------------------------------
# Embedding helper — uses Anthropic's voyage-3 via the embeddings endpoint
# ---------------------------------------------------------------------------

def _embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using Anthropic's voyage-3-lite model.
    Returns a list of float vectors (one per input text).
    Falls back to a simple TF-IDF-style hash vector when the API is
    unavailable (e.g. offline CI runs).
    """
    import anthropic

    import logging
    _log = logging.getLogger(__name__)

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.beta.messages.create(
            model="voyage-3-lite",
            max_tokens=1,
            messages=[{"role": "user", "content": t} for t in texts],
            betas=["embeddings-2025-03-05"],
        )
        return [e.embedding for e in response.embeddings]
    except KeyError:
        # ANTHROPIC_API_KEY not set — expected in offline/test environments
        _log.debug("ANTHROPIC_API_KEY not set; using fallback embeddings")
    except ImportError:
        _log.debug("anthropic not installed; using fallback embeddings")
    except Exception as exc:
        # Rate limit, network error, API change — log at WARNING so operators
        # can see it, but continue with fallback to keep the pipeline running.
        _log.warning(
            "Embedding API call failed (using fallback): %s: %s",
            type(exc).__name__, exc,
        )
    return [_fallback_embed(t) for t in texts]


def _fallback_embed(text: str, dim: int = 384) -> list[float]:
    """Deterministic pseudo-embedding for offline/test use."""
    import hashlib
    import struct
    seed = hashlib.sha256(text.encode()).digest()
    vec = []
    for i in range(0, dim * 4, 4):
        chunk = seed[(i % 32): (i % 32) + 4] or seed[:4]
        val = struct.unpack("f", chunk.ljust(4, b"\x00")[:4])[0]
        vec.append(val % 1.0)
    norm = sum(v ** 2 for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _finding_text(f: "Finding") -> str:
    """Canonical text representation of a finding for embedding."""
    return (
        f"{f.category} {f.severity} {f.title} "
        f"{f.description} {f.suggestion} file:{f.file}"
    )


def _finding_text_from_dict(payload: dict) -> str:
    """Same as _finding_text but works on a plain dict (e.g. from JSON storage).
    Replaces the fragile type("F", (), payload)() namespace hack.
    """
    return (
        "{category} {severity} {title} {description} {suggestion} file:{file}"
        .format_map(payload)
    )


# ---------------------------------------------------------------------------
# FindingStore
# ---------------------------------------------------------------------------

class FindingStore:
    """
    Persistent semantic store for RepoSentinel findings.

    Usage
    -----
    store = FindingStore()
    store.add(findings, run_id="pr-42", repo="owner/repo")
    dupes = store.find_duplicates(new_findings)
    fresh = [f for f in new_findings if f not in dupes]
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        findings: list["Finding"],
        run_id: str = "",
        repo: str = "",
        outcome: str = "pending",   # "accepted" | "dismissed" | "pending"
    ) -> None:
        """Embed and persist findings. outcome is updated later via label()."""
        if not findings:
            return

        texts = [_finding_text(f) for f in findings]
        embeddings = _embed(texts)
        timestamp = int(time.time())

        ids, docs, embs, metas = [], [], [], []
        for idx, (f, emb) in enumerate(zip(findings, embeddings)):
            uid = f"{run_id}-{idx}-{int(time.time())}"
            ids.append(uid)
            docs.append(_finding_text(f))
            embs.append(emb)
            metas.append({
                "run_id":   run_id,
                "repo":     repo,
                "outcome":  outcome,
                "timestamp": timestamp,
                "category": f.category,
                "severity": f.severity,
                "file":     f.file,
                "line_start": f.line_start,
                "confidence": f.confidence,
                "payload":  json.dumps(asdict(f)),
            })

        self._col.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    def label(self, run_id: str, outcome: str) -> int:
        """Update outcome for all findings from a given run. Returns count updated."""
        results = self._col.get(where={"run_id": run_id})
        if not results["ids"]:
            return 0
        for uid, meta in zip(results["ids"], results["metadatas"]):
            meta["outcome"] = outcome
            self._col.update(ids=[uid], metadatas=[meta])
        return len(results["ids"])

    # ------------------------------------------------------------------
    # Read / deduplication
    # ------------------------------------------------------------------

    def find_duplicates(
        self,
        findings: list["Finding"],
        threshold: float = SIMILARITY_THRESHOLD,
        look_back_days: int = 90,
    ) -> set[int]:
        """
        Return indices of findings that are semantically similar to
        already-stored findings (i.e. likely duplicates).
        """
        if not findings or self._col.count() == 0:
            return set()

        texts = [_finding_text(f) for f in findings]
        embeddings = _embed(texts)
        cutoff = int(time.time()) - look_back_days * 86_400

        duplicate_indices: set[int] = set()
        for idx, emb in enumerate(embeddings):
            results = self._col.query(
                query_embeddings=[emb],
                n_results=3,
                where={"timestamp": {"$gte": cutoff}},
                include=["distances", "metadatas"],
            )
            distances = results["distances"][0]
            # ChromaDB cosine distance: 0 = identical, 1 = orthogonal
            if distances and (1 - distances[0]) >= threshold:
                duplicate_indices.add(idx)

        return duplicate_indices

    def similar_context(
        self,
        query: str,
        n: int = 5,
        outcome_filter: str | None = None,
    ) -> list[dict]:
        """
        Retrieve n most similar past findings to a free-text query.
        Used to inject historical context into the LLM prompt.
        """
        if self._col.count() == 0:
            return []

        emb = _embed([query])[0]
        where = {"outcome": outcome_filter} if outcome_filter else None
        results = self._col.query(
            query_embeddings=[emb],
            n_results=min(n, self._col.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({
                "similarity": round(1 - dist, 3),
                "finding": json.loads(meta["payload"]),
                "outcome": meta["outcome"],
                "repo": meta["repo"],
            })
        return out

    # ------------------------------------------------------------------
    # Export for fine-tuning
    # ------------------------------------------------------------------

    def export_training_data(self, min_labelled: int = 20) -> list[dict]:
        """
        Export labelled findings as training examples.
        Only returns accepted/dismissed findings (not pending).
        Used by the fine-tuning pipeline in trainer.py.
        """
        results = self._col.get(
            where={"outcome": {"$in": ["accepted", "dismissed"]}},
            include=["metadatas"],
        )
        records = []
        for meta in results["metadatas"]:
            payload = json.loads(meta["payload"])
            records.append({
                "features": {
                    "category":   payload["category"],
                    "severity":   payload["severity"],
                    "confidence": payload["confidence"],
                    "file_ext":   Path(payload["file"]).suffix,
                    "title_len":  len(payload["title"]),
                    "desc_len":   len(payload["description"]),
                },
                "label": 1 if meta["outcome"] == "accepted" else 0,
                "text":  _finding_text_from_dict(payload),
            })

        if len(records) < min_labelled:
            print(
                f"[FindingStore] Only {len(records)} labelled examples "
                f"(need {min_labelled}). Collect more feedback before training."
            )
        return records

    def stats(self) -> dict:
        """Quick summary stats about the store."""
        total = self._col.count()
        if total == 0:
            return {"total": 0}
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        outcomes = {}
        categories = {}
        for m in all_meta:
            outcomes[m["outcome"]] = outcomes.get(m["outcome"], 0) + 1
            categories[m["category"]] = categories.get(m["category"], 0) + 1
        return {
            "total":      total,
            "outcomes":   outcomes,
            "categories": categories,
        }
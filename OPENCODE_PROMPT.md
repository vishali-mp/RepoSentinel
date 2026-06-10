# RepoSentinel — OpenCode Agent Context Prompt

Paste this into OpenCode (or Claude Code, Cursor, Aider) at the start of
any session. It reflects the verified file structure as of June 2026.

---

You are working on RepoSentinel — an autonomous AI code review agent.
Before writing any code, read and internalize the full project structure.

## Verified file structure

repo-sentinel/
├── agent/
│   ├── __init__.py
│   ├── analyzer.py       ← git diff → static tools → Claude → findings
│   ├── discovery.py      ← GitHub Search API, repo ranking by priority score
│   ├── scanner.py        ← cross-repo orchestration, cloning, per-repo scan
│   ├── vector_store.py   ← ChromaDB embeddings, semantic dedup, memory
│   ├── trainer.py        ← sklearn classifier, CV model selection, inference
│   ├── reporter.py       ← single-repo: GitHub Issues + PR inline comments
│   ├── multi_reporter.py ← cross-repo: findings.json + scan_report.md + dashboard
│   ├── main.py           ← CLI entrypoint (single-repo PR/scheduled runs)
│   └── scan.py           ← CLI entrypoint (cross-repo weekly scan)
├── evals/
│   ├── golden_dataset.json  ← 25 labelled code snippets with expected findings
│   ├── run_evals.py         ← precision/recall/F1 harness, results.jsonl tracking
│   ├── ab_compare.py        ← A/B prompt comparison CLI
│   └── prompts/
│       ├── v1_baseline.txt
│       └── v2_with_cwe_severity.txt
├── webapp/
│   ├── app.py            ← FastAPI routes (/, /health, /scan SSE), inline HTML UI
│   └── streaming.py      ← FindingStreamParser (partial JSON), stream_findings()
├── tests/
│   ├── test_analyzer.py   ← 13 tests: LLM parsing, confidence filtering, dedup
│   ├── test_extensions.py ← 22 tests: vector store, trainer, FindingClassifier
│   ├── test_cross_repo.py ← 23 tests: discovery, scanner, multi_reporter
│   ├── test_evals.py      ← 27 tests: matcher, scoring, line_range proximity
│   └── test_webapp.py     ← 22 tests: FindingStreamParser, SSE, FastAPI routes
├── .github/
│   └── workflows/
│       ├── sentinel.yml          ← PR-triggered single-repo analysis
│       └── cross_repo_scan.yml  ← weekly scheduled cross-repo scan
├── Dockerfile
├── README.md
└── requirements.txt      ← all deps: anthropic, fastapi, uvicorn, chromadb,
                             scikit-learn, pandas, numpy, ruff, bandit, pytest

Total: 107 passing tests across 5 test modules.

## Read these files in this order

1.  README.md
2.  agent/analyzer.py
3.  agent/vector_store.py
4.  agent/trainer.py
5.  agent/discovery.py
6.  agent/scanner.py
7.  agent/reporter.py
8.  agent/multi_reporter.py
9.  agent/main.py and agent/scan.py
10. webapp/streaming.py
11. webapp/app.py
12. evals/golden_dataset.json
13. evals/run_evals.py
14. evals/ab_compare.py
15. tests/ (all 5 files)
16. .github/workflows/sentinel.yml
17. .github/workflows/cross_repo_scan.yml

## Architecture — two execution paths

Path 1 — CLI / GitHub Actions:
  discovery.py → scanner.py → analyzer.py →
  vector_store.py (dedup + store) → trainer.py (classifier filter) →
  reporter.py / multi_reporter.py

Path 2 — Web UI:
  Browser EventSource → app.py /scan → scan_repo_stream() →
  stream_findings() [streaming.py] → FindingStreamParser →
  SSE events → DOM

## Key data models

Finding (dataclass, agent/analyzer.py):
  category: "bug" | "performance" | "security" | "style"
  severity: "critical" | "high" | "medium" | "low"
  file: str
  line_start: int
  line_end: int
  title: str
  description: str
  suggestion: str
  confidence: float  # 0.0–1.0

EvalCase expected_findings schema (golden_dataset.json):
  category: str
  severity: str
  line_range: [start, end]   ← used by _matches_expected() with proximity=15
  keywords: list[str]        ← fuzzy matched against title+description+suggestion

RepoTarget (dataclass, agent/discovery.py):
  full_name, clone_url, default_branch, stars, language,
  topics, open_issues, last_pushed, priority_score

## Critical design rules — do not violate

1. agent/ has zero knowledge of webapp/. webapp/ imports from agent/.
   Never reverse this dependency.

2. query_llm() in analyzer.py accepts system_prompt_override: str | None.
   This is how the eval harness swaps prompts. Never remove this parameter.

3. FindingStreamParser in streaming.py uses bracket-depth counting to emit
   complete Finding objects as Claude streams token by token. Never replace
   with a blocking collect-then-parse approach.

4. The vector store stores findings with outcome="pending". Outcomes update
   to "accepted" or "dismissed" externally (GitHub Issue close webhooks).
   trainer.py reads only labelled (non-pending) findings.

5. All GitHub API calls go through reporter.py or multi_reporter.py only.
   Never call the GitHub API from analyzer.py, scanner.py, or webapp/.

6. The vector store and classifier are optional in analyzer.py — wrapped in
   try/except at lines ~244-250. agent/ must remain importable without
   chromadb or scikit-learn installed.

7. webapp/ is optional. agent/ must work standalone as a pure CLI tool.

## Rules for every change

- Run pytest tests/ -v before AND after every change.
  All 107 tests must pass. Do not disable or skip tests.
- Add tests in tests/ for every new module or function you create.
- If you change the Finding dataclass, update golden_dataset.json schema.
- If you change the system prompt in analyzer.py, run:
    python -m evals.run_evals
  and report precision/recall/F1 before and after.
- Never hardcode secrets. All API keys from environment variables only.
- Keep all agent/ code synchronous. Async lives in webapp/ only.

## What you are about to build

[REPLACE THIS with your task, e.g.:
"Add a GitHub webhook handler that marks findings as accepted or dismissed
when their corresponding GitHub Issues are closed, feeding the trainer.py
feedback loop."]

## Before writing a single line of code

1. Confirm you have read all files listed above.
2. State which existing files you will modify and why.
3. State which new files you will create.
4. Identify any tests that will break and how you will fix them.
5. State what the test count will be after your change (currently 107).
6. Ask one clarifying question if anything is ambiguous.

Only then begin coding.

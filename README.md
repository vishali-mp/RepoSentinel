# 🤖 RepoSentinel

An autonomous AI code-review agent that scans any public GitHub repository — by URL or on PR — for bugs, performance issues, security vulnerabilities, and code quality problems. Powered by Claude, ChromaDB (RAG), and GitHub Actions.

## How it works

```
repo URL / git diff → ruff + bandit + eslint → Claude analysis → findings stream / GitHub Issues
```

RepoSentinel works in two modes:

**🔗 Any repo by URL** — Paste any public GitHub repo into the webapp. It shallow-clones, diffs the recent commits, and streams findings live via SSE. No setup needed.

**🔁 PRs + scheduled scans** — Install the GitHub Action in your repo. It triggers on every PR and runs weekly, posting findings as PR comments and GitHub Issues.

In both modes:

1. Computes a `git diff` of changed files
2. Runs static analysis tools (ruff, bandit)
3. Sends the diff + tool output to Claude for semantic reasoning
4. Queries past accepted findings from ChromaDB for context (RAG)
5. Filters findings by confidence threshold (default 70%)
6. Deduplicates semantically — never re-reports the same class of issue

## Quick start

### 1. Add to your repo

```bash
# Copy the .github/workflows/sentinel.yml into your repo
# Copy the agent/ directory and requirements.txt
```

### 2. Add secrets

In your GitHub repo → Settings → Secrets and variables → Actions:

| Secret | Value |
|--------|-------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |

`GITHUB_TOKEN` is provided automatically by GitHub Actions.

### 3. That's it

Open a PR - RepoSentinel will comment with its analysis within ~2 minutes.

---

## What it detects

| Category | Examples |
|----------|---------|
| 🐛 **Bugs** | Null dereference, off-by-one errors, wrong type assumptions, unhandled exceptions |
| ⚡ **Performance** | N+1 queries, blocking I/O in async code, unnecessary re-renders, memory leaks |
| 🔒 **Security** | SQL injection, hardcoded secrets, insecure defaults, SSRF, path traversal |
| 🎨 **Style** | Dead code, overly complex functions, missing error handling, unclear naming |

## Configuration

| CLI flag | Default | Description |
|----------|---------|-------------|
| `--confidence` | `0.70` | Minimum confidence to report a finding (0.0–1.0) |
| `--no-issues` | off | Don't open individual Issues for critical/high findings |
| `--json-output` | off | Print findings as JSON (useful for debugging) |

## Running locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# Analyze last 5 commits
python -m agent.main --mode scheduled --base-ref HEAD~5 --head-ref HEAD

# Debug: print findings as JSON, don't post to GitHub
python -m agent.main --base-ref HEAD~5 --json-output --no-issues
```

## Running tests

```bash
pytest tests/ -v
```

## Architecture

```
agent/
├── analyzer.py       # Git diff + static tools + LLM reasoning → findings
├── vector_store.py   # ChromaDB RAG: semantic dedup + context retrieval
├── reporter.py       # GitHub API: PR comments, Issues, deduplication
├── scanner.py        # Cross-repo scan orchestration
├── discovery.py      # GitHub API: discover repos by topic/language
├── trainer.py        # Fine-tune sklearn classifier from labelled findings
├── main.py           # CLI entrypoint (PR / local diffs)
└── scan.py           # CLI entrypoint (cross-repo scans)

webapp/
├── app.py            # FastAPI server with SSE streaming + inline UI
├── streaming.py      # Async finding streamer
└── Dockerfile        # Container image (Render / Railway / Fly.io ready)

evals/
├── golden_dataset.json   # 25 multi-category test cases
├── run_evals.py          # Precision / recall / F1 eval harness
└── ab_compare.py         # A/B prompt comparison tooling

agent/
└── chroma/               # Persistent vector store (auto-created)

.github/
└── workflows/
    └── cross_repo_scan.yml  # GitHub Actions: scheduled + manual cross-repo scans

tests/
└── test_analyzer.py         # Unit tests (no network calls)
```

## Secrets & privacy

RepoSentinel only reads code that changes in a diff. Your full codebase is never uploaded as whole. If you run a local model via Ollama and point the agent at it, no code leaves your infrastructure at all.

## Stack

- **Python 3.12**
- **Claude** / **Gemini** — semantic analysis
- **ChromaDB + voyage-3-lite** — vector store for RAG (semantic dedup + context retrieval)
- **ruff** — Python linting
- **bandit** — Python security scanning
- **eslint** — JS/TS linting
- **scikit-learn** — optional fine-tuning from labelled findings
- **FastAPI + uvicorn** — web server with SSE streaming
- **GitHub Actions** — scheduling and CI integration
- **GitHub API** — Issue and PR comment posting
- **Docker** — containerized deployment (Render / Railway / Fly.io)

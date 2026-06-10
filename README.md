# 🤖 RepoSentinel

An autonomous AI agent that periodically scans GitHub repositories for bugs, performance issues, security vulnerabilities, and code quality problems — powered by Claude and GitHub Actions.

## How it works

```
git diff → ruff + bandit + eslint → Claude analysis → GitHub Issues / PR comments
```

On every pull request and on a weekly schedule, RepoSentinel:

1. Computes a `git diff` of changed files
2. Runs static analysis tools (ruff, bandit, eslint)
3. Sends the diff + tool output to Claude for semantic reasoning
4. Filters findings by confidence threshold (default 70%)
5. Posts a summary table on the PR and opens individual Issues for critical/high findings
6. Deduplicates — never reopens an issue it already filed

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

Open a PR — RepoSentinel will comment with its analysis within ~2 minutes.

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
├── analyzer.py   # Git diff + static tools + LLM reasoning → findings
├── reporter.py   # GitHub API: PR comments, Issues, deduplication
└── main.py       # CLI entrypoint

.github/
└── workflows/
    └── sentinel.yml  # GitHub Actions: PR trigger + weekly schedule

tests/
└── test_analyzer.py  # Unit tests (no network calls)
```

## Secrets & privacy

RepoSentinel only reads code that changes in a diff. Your full codebase is never uploaded wholesale. If you run a local model via Ollama and point the agent at it, no code leaves your infrastructure at all.

## Stack

- **Python 3.12**
- **Claude** (claude-sonnet-4) — semantic analysis
- **ruff** — Python linting
- **bandit** — Python security scanning  
- **eslint** — JS/TS linting
- **GitHub Actions** — scheduling and CI integration
- **GitHub API** — Issue and PR comment posting

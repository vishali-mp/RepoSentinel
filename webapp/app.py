"""
RepoSentinel · webapp/app.py

FastAPI application with two key endpoints:

  GET /            → Serves the single-page UI (inline HTML)
  GET /scan        → SSE stream: clones repo, analyzes, streams findings
  GET /health      → Health check for deployment platforms

The /scan endpoint uses StreamingResponse with Server-Sent Events (SSE).
Each finding is emitted as:
  data: {"category": "security", "severity": "critical", ...}\n\n

Special events:
  data: {"event": "start",    "message": "..."}\n\n   — scan beginning
  data: {"event": "progress", "message": "..."}\n\n   — status update
  data: {"event": "done",     "stats": {...}}\n\n     — scan complete
  data: {"event": "error",    "message": "..."}\n\n   — something failed
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator


# ---------------------------------------------------------------------------
# Per-repo cooldown (in-memory, resets on server restart)
# ---------------------------------------------------------------------------

@dataclass
class RepoCache:
    cache: dict[str, float] = field(default_factory=dict)

    @property
    def cooldown_seconds(self) -> int:
        return int(os.getenv("REPO_COOLDOWN_SECONDS", "600"))

    def is_warm(self, repo: str) -> bool:
        ts = self.cache.get(repo)
        return ts is not None and (time.time() - ts) < self.cooldown_seconds

    def remaining(self, repo: str) -> int:
        ts = self.cache.get(repo)
        if ts is None:
            return 0
        remain = int(self.cooldown_seconds - (time.time() - ts))
        return max(remain, 0)

    def mark(self, repo: str) -> None:
        self.cache[repo] = time.time()


repo_cache = RepoCache()

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded


def _rate_key(request: Request) -> str:
    """Rate limit key: session + IP. Falls back to IP if no session param."""
    session = request.query_params.get("session", "")
    ip = request.client.host if request.client else "unknown"
    return f"{ip}:{session}" if session else ip

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.analyzer import run_ruff, run_bandit, read_file_with_lines
from webapp.streaming import stream_findings

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=_rate_key)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RepoSentinel",
    description="Autonomous AI code review agent — streaming findings via SSE",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def sse(data: dict) -> str:
    """Format a dict as an SSE event string."""
    return f"data: {json.dumps(data)}\n\n"


async def _keep_alive(interval: int = 15) -> AsyncIterator[str]:
    """Yields SSE comment lines to keep the connection alive."""
    while True:
        await asyncio.sleep(interval)
        yield ": keep-alive\n\n"


# ---------------------------------------------------------------------------
# Repo helpers (async wrappers around git operations)
# ---------------------------------------------------------------------------

async def _clone_repo(clone_url: str, dest: Path) -> None:
    """Shallow-clone a public GitHub repo."""
    proc = await asyncio.create_subprocess_exec(
        "git", "clone", "--depth=50", "--single-branch", clone_url, str(dest),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"git clone failed: {stderr.decode()[:200]}")


def _parse_github_url(url: str) -> tuple[str, str]:
    """
    Parse a GitHub URL or owner/repo string.
    Returns (clone_url, display_name).
    """
    url = url.strip().rstrip("/")

    # Handle plain "owner/repo" format
    if "/" in url and "github.com" not in url and not url.startswith("http"):
        owner_repo = url
        return f"https://github.com/{owner_repo}.git", owner_repo

    # Handle full GitHub URLs
    if "github.com" in url:
        # Extract owner/repo from URL
        parts = url.replace("https://", "").replace("http://", "").split("/")
        if len(parts) >= 3:
            owner_repo = f"{parts[1]}/{parts[2].replace('.git', '')}"
            return f"https://github.com/{owner_repo}.git", owner_repo

    raise ValueError(f"Could not parse GitHub URL: {url!r}")


# ---------------------------------------------------------------------------
# cwd-safe git helpers (no os.chdir — safe for concurrent async handlers)
# ---------------------------------------------------------------------------

def _get_changed_files_in(repo_dir: Path, base_ref: str, head_ref: str) -> list[Path]:
    """Return absolute Paths of changed Python/JS/TS files between two refs."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACM", base_ref, head_ref],
        capture_output=True, text=True, check=True, cwd=repo_dir,
    )
    extensions = {".py", ".js", ".ts", ".jsx", ".tsx"}
    return [
        repo_dir / f
        for f in result.stdout.splitlines()
        if Path(f).suffix in extensions and (repo_dir / f).exists()
    ]


def _get_diff_in(repo_dir: Path, base_ref: str, head_ref: str) -> str:
    """Return unified diff between two refs, truncated to ~50k chars."""
    result = subprocess.run(
        ["git", "diff", base_ref, head_ref, "--unified=5"],
        capture_output=True, text=True, check=True, cwd=repo_dir,
    )
    return result.stdout[:50_000]


# ---------------------------------------------------------------------------
# Core scan generator
# ---------------------------------------------------------------------------

async def scan_repo_stream(repo_input: str, commits: int = 20) -> AsyncIterator[str]:
    """
    Full scan pipeline as an async SSE generator.

    Yields SSE-formatted strings as the scan progresses:
      - progress events during setup
      - finding events as Claude generates them
      - done event with final stats
    """
    tmpdir = None
    start_time = time.time()
    n_findings = 0

    try:
        # ── Parse URL ─────────────────────────────────────────────
        try:
            clone_url, repo_name = _parse_github_url(repo_input)
        except ValueError as e:
            yield sse({"event": "error", "message": str(e)})
            return

        yield sse({"event": "start", "message": f"Starting scan of {repo_name}"})
        yield sse({"event": "progress", "message": "Cloning repository (shallow, last 50 commits)…"})

        # ── Clone ─────────────────────────────────────────────────
        tmpdir = Path(tempfile.mkdtemp(prefix="sentinel-web-"))
        repo_dir = tmpdir / "repo"
        await _clone_repo(clone_url, repo_dir)

        yield sse({"event": "progress", "message": "Identifying changed files…"})

        # ── Get diff window ───────────────────────────────────────
        # Pass cwd= to subprocess instead of os.chdir() — os.chdir() is
        # process-global and causes a race condition when concurrent
        # FastAPI handlers run in the same process.
        result = subprocess.run(
            ["git", "rev-list", f"--max-count={commits + 1}", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=repo_dir,
        )
        commit_list = result.stdout.strip().splitlines()
        base_ref = commit_list[-1] if len(commit_list) > 1 else "HEAD~1"
        head_ref = commit_list[0]

        changed_files = _get_changed_files_in(repo_dir, base_ref, head_ref)

        if not changed_files:
            yield sse({"event": "progress", "message": "No changed files found in the last 50 commits."})
            yield sse({"event": "done", "stats": {"repo": repo_name, "files": 0, "findings": 0, "elapsed": round(time.time() - start_time, 1)}})
            return

        n_files = len(changed_files)
        n_lines = sum(read_file_with_lines(f)[1] for f in changed_files)
        yield sse({"event": "progress", "message": f"Analyzing {n_files} changed files ({n_lines:,} lines)…"})

        # ── Static tools ──────────────────────────────────────
        yield sse({"event": "progress", "message": "Running static analysis (ruff, bandit)…"})

        ruff_out   = run_ruff(changed_files)
        bandit_out = run_bandit(changed_files)
        static_output = json.dumps({
            "ruff":   json.loads(ruff_out)   if ruff_out   and "{" in ruff_out   else [],
            "bandit": json.loads(bandit_out) if bandit_out and "{" in bandit_out else {},
        })

        # ── File contents (absolute paths — no chdir needed) ──────
        file_contents: dict[str, str] = {}
        for path in changed_files:
            content, _ = read_file_with_lines(path)
            file_contents[str(path.relative_to(repo_dir))] = content

        diff = _get_diff_in(repo_dir, base_ref, head_ref)

        # ── Stream findings from Claude ───────────────────────────
        yield sse({"event": "progress", "message": "Claude is analyzing the code — findings will appear as they're generated…"})

        async for finding in stream_findings(
            diff=diff,
            file_contents=file_contents,
            static_output=static_output,
        ):
            n_findings += 1
            # Emit each finding as it arrives
            yield sse({
                "event": "finding",
                "finding": {
                    "category":    finding.category,
                    "severity":    finding.severity,
                    "file":        finding.file,
                    "line_start":  finding.line_start,
                    "line_end":    finding.line_end,
                    "title":       finding.title,
                    "description": finding.description,
                    "suggestion":  finding.suggestion,
                    "confidence":  finding.confidence,
                }
            })

        # ── Done ──────────────────────────────────────────────────
        elapsed = round(time.time() - start_time, 1)
        yield sse({
            "event": "done",
            "stats": {
                "repo":     repo_name,
                "files":    n_files,
                "lines":    n_lines,
                "findings": n_findings,
                "elapsed":  elapsed,
            }
        })

    except asyncio.TimeoutError:
        yield sse({"event": "error", "message": "Clone timed out — repo may be too large or unavailable."})

    except Exception as exc:
        yield sse({"event": "error", "message": str(exc)[:300]})

    finally:
        if tmpdir and tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "RepoSentinel"}


@app.get("/scan")
@limiter.limit(os.getenv("RATE_LIMIT", "5/minute"))
async def scan(
    request: Request,
    repo: str = Query(..., description="GitHub repo URL or owner/repo"),
    commits: int = Query(20, ge=1, le=100, description="Number of recent commits to analyze"),
    model: str = Query("gemini-2.0-flash", description="LLM model name (e.g. claude-sonnet-4-20250514, gemini-2.0-flash)"),
    session: str = Query("", description="Client session ID for rate limiting"),
):
    """
    Stream findings for a GitHub repository as Server-Sent Events.

    Connect with: new EventSource('/scan?repo=owner/repo&model=gemini-2.0-flash')
    """
    # Normalize repo name for cooldown key
    repo_key = repo.strip().rstrip("/").lower()
    if repo_cache.is_warm(repo_key):
        remain = repo_cache.remaining(repo_key)
        async def _cooldown_stream():
            yield sse({"event": "error", "message": f"This repo was scanned recently. Please wait {remain}s before re-scanning."})
        return StreamingResponse(_cooldown_stream(), media_type="text/event-stream")

    # Derive provider from model name
    provider = "gemini" if model.startswith("gemini") else "anthropic"
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model

    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        async def error_stream():
            yield sse({"event": "error", "message": "GEMINI_API_KEY not configured on server."})
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        async def error_stream():
            yield sse({"event": "error", "message": "ANTHROPIC_API_KEY not configured on server."})
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    repo_cache.mark(repo_key)

    return StreamingResponse(
        scan_repo_stream(repo, commits),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering for SSE
            "Connection": "keep-alive",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page application."""
    return HTML_PAGE


# ---------------------------------------------------------------------------
# Single-page UI — inlined so the app is one file to deploy
# ---------------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RepoSentinel — AI Code Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0C0F1A; --surface: #141827; --border: #1E2438;
    --blue: #3A6FFF; --teal: #00D4B8; --amber: #F5A623;
    --red: #FF4D6A; --green: #22C98B;
    --text: #E2E8F0; --muted: #64748B;
    --font: 'Inter', sans-serif; --mono: 'DM Mono', monospace;
  }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         min-height: 100vh; display: flex; flex-direction: column; }

  header { padding: 1.25rem 2rem; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 12px; }
  .logo { font-family: var(--mono); font-size: 15px; font-weight: 500;
          color: var(--teal); letter-spacing: -0.3px; }
  .logo span { color: var(--muted); font-weight: 400; }
  .badge { font-family: var(--mono); font-size: 10px; padding: 2px 8px;
           border-radius: 3px; background: rgba(58,111,255,0.15);
           color: var(--blue); border: 1px solid rgba(58,111,255,0.2);
           letter-spacing: 0.05em; }

  main { flex: 1; max-width: 860px; width: 100%; margin: 0 auto;
         padding: 3rem 1.5rem; }

  .hero-title { font-size: 32px; font-weight: 600; letter-spacing: -0.5px;
                margin-bottom: 8px; line-height: 1.2; }
  .hero-title span { color: var(--teal); }
  .hero-sub { color: var(--muted); font-size: 15px; margin-bottom: 2.5rem; line-height: 1.6; }

  .input-row { display: flex; gap: 10px; margin-bottom: 1rem; }
  .repo-input { flex: 1; background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; padding: 0.75rem 1rem; font-family: var(--mono);
                font-size: 14px; color: var(--text); outline: none; transition: border-color 0.15s; }
  .repo-input:focus { border-color: var(--blue); }
  .repo-input::placeholder { color: var(--muted); }
  .scan-btn { background: var(--blue); color: #fff; border: none; border-radius: 8px;
              padding: 0.75rem 1.5rem; font-family: var(--font); font-size: 14px;
              font-weight: 600; cursor: pointer; transition: all 0.15s; white-space: nowrap; }
  .scan-btn:hover:not(:disabled) { background: #4D80FF; transform: translateY(-1px); }
  .scan-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  .provider-select { background: var(--surface); border: 1px solid var(--border);
                     border-radius: 8px; padding: 0.75rem 1rem; font-family: var(--mono);
                     font-size: 13px; color: var(--text); outline: none; cursor: pointer; }
  .provider-select:focus { border-color: var(--blue); }

  .examples { font-size: 12px; color: var(--muted); margin-bottom: 2.5rem; }
  .examples a { color: var(--blue); cursor: pointer; text-decoration: none; }
  .examples a:hover { text-decoration: underline; }

  /* Status bar */
  .status-bar { background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; padding: 0.875rem 1rem; margin-bottom: 1.5rem;
                display: none; align-items: center; gap: 10px; font-size: 13px; }
  .status-bar.show { display: flex; }
  .spinner { width: 14px; height: 14px; border: 2px solid var(--border);
             border-top-color: var(--blue); border-radius: 50%;
             animation: spin 0.8s linear infinite; flex-shrink: 0; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .status-text { color: var(--muted); }

  /* Stats row */
  .stats-row { display: flex; gap: 16px; margin-bottom: 1.5rem; display: none; }
  .stats-row.show { display: flex; }
  .stat-pill { background: var(--surface); border: 1px solid var(--border);
               border-radius: 6px; padding: 0.5rem 0.875rem; font-size: 12px; }
  .stat-pill strong { color: var(--text); font-weight: 600; }
  .stat-pill span { color: var(--muted); }

  /* Findings */
  .findings-header { font-size: 11px; font-weight: 500; letter-spacing: 0.08em;
                     text-transform: uppercase; color: var(--muted); margin-bottom: 0.875rem;
                     display: none; }
  .findings-header.show { display: block; }
  .findings-list { display: flex; flex-direction: column; gap: 10px; }

  .finding-card { background: var(--surface); border: 1px solid var(--border);
                  border-radius: 10px; padding: 1.125rem 1.25rem;
                  animation: slideIn 0.3s ease; }
  @keyframes slideIn { from { opacity: 0; transform: translateY(8px); }
                       to   { opacity: 1; transform: translateY(0); } }

  .finding-top { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px; }
  .sev-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }
  .sev-critical { background: var(--red); }
  .sev-high     { background: var(--amber); }
  .sev-medium   { background: #a78bfa; }
  .sev-low      { background: var(--muted); }

  .finding-title { font-size: 14px; font-weight: 600; line-height: 1.4; flex: 1; }
  .finding-meta { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
  .meta-tag { font-family: var(--mono); font-size: 11px; padding: 2px 8px;
              border-radius: 4px; color: var(--muted); border: 1px solid var(--border); }
  .meta-tag.cat-security    { color: #fb7185; border-color: rgba(251,113,133,0.25); background: rgba(251,113,133,0.08); }
  .meta-tag.cat-bug         { color: var(--amber); border-color: rgba(245,166,35,0.25); background: rgba(245,166,35,0.08); }
  .meta-tag.cat-performance { color: #34d399; border-color: rgba(52,211,153,0.25); background: rgba(52,211,153,0.08); }
  .meta-tag.cat-style       { color: #818cf8; border-color: rgba(129,140,248,0.25); background: rgba(129,140,248,0.08); }

  .finding-desc { font-size: 13px; color: var(--muted); line-height: 1.6; margin-bottom: 8px; }
  .finding-fix  { font-size: 13px; color: #94a3b8; line-height: 1.6; padding: 8px 10px;
                  background: rgba(58,111,255,0.06); border-left: 2px solid var(--blue);
                  border-radius: 0 4px 4px 0; }
  .finding-fix strong { color: var(--blue); font-weight: 500; }

  .confidence { font-family: var(--mono); font-size: 10px; color: var(--muted); margin-left: auto; flex-shrink: 0; }

  /* Empty/done states */
  .empty-state { text-align: center; padding: 4rem 2rem; color: var(--muted); display: none; }
  .empty-state.show { display: block; }
  .empty-icon { font-size: 40px; margin-bottom: 1rem; }
  .empty-title { font-size: 16px; font-weight: 500; margin-bottom: 6px; color: var(--text); }
  .empty-sub { font-size: 14px; }

  .error-msg { background: rgba(255,77,106,0.08); border: 1px solid rgba(255,77,106,0.2);
               border-radius: 8px; padding: 1rem 1.25rem; font-size: 13px;
               color: #ff8099; margin-bottom: 1.5rem; display: none; }
  .error-msg.show { display: block; }
</style>
</head>
<body>

<header>
  <div class="logo">Repo<span>Sentinel</span></div>
  <div class="badge">AI Code Review</div>
</header>

<main>
  <h1 class="hero-title">Find bugs before<br>they reach <span>production</span></h1>
  <p class="hero-sub">Paste any public GitHub repository — RepoSentinel analyzes recent commits using Claude AI and surfaces bugs, security vulnerabilities, and performance issues as they're found.</p>

  <div class="input-row">
    <input class="repo-input" id="repoInput" type="text"
           placeholder="owner/repo  or  https://github.com/owner/repo"
           onkeydown="if(event.key==='Enter') startScan()">
    <select id="modelSelect" class="provider-select">
      <optgroup label="Anthropic">
        <option value="claude-sonnet-4-20250514">Claude Sonnet 4</option>
        <option value="claude-haiku-3-20240307">Claude Haiku 3</option>
      </optgroup>
      <optgroup label="Google (free tier)">
        <option value="gemini-3.1-flash-lite" selected>Gemini 3.1 Flash Lite</option>
        <option value="gemini-3.5-flash">Gemini 3.5 Flash</option>
        <option value="gemini-2.5-flash-lite">Gemini 2.5 Flash Lite</option>
        <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
        <option value="gemini-2.0-flash">Gemini 2.0 Flash</option>
      </optgroup>
    </select>
    <button class="scan-btn" id="scanBtn" onclick="startScan()">Scan →</button>
  </div>

  <p class="examples">
    Try:
    <a onclick="fillAndScan('psf/requests')">psf/requests</a> ·
    <a onclick="fillAndScan('pallets/flask')">pallets/flask</a> ·
    <a onclick="fillAndScan('tiangolo/fastapi')">tiangolo/fastapi</a>
  </p>

  <div class="status-bar" id="statusBar">
    <div class="spinner"></div>
    <div class="status-text" id="statusText">Initialising…</div>
  </div>

  <div class="error-msg" id="errorMsg"></div>

  <div class="stats-row" id="statsRow"></div>
  <div class="findings-header" id="findingsHeader">Findings</div>
  <div class="findings-list" id="findingsList"></div>
  <div class="empty-state" id="emptyState">
    <div class="empty-icon">✅</div>
    <div class="empty-title">No issues found</div>
    <div class="empty-sub">No findings above the confidence threshold in recent commits.</div>
  </div>
</main>

<script>
// Session ID for per-user rate limiting (no login required)
function getSessionId() {
  let id = localStorage.getItem('sentinel_session');
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem('sentinel_session', id);
  }
  return id;
}

let currentSource = null;
let findingCount = 0;

const SEV_CLASS = { critical: 'sev-critical', high: 'sev-high', medium: 'sev-medium', low: 'sev-low' };
const CAT_CLASS  = { security: 'cat-security', bug: 'cat-bug', performance: 'cat-performance', style: 'cat-style' };
const SEV_LABEL  = { critical: '🔴 Critical', high: '🟠 High', medium: '🟡 Medium', low: '🔵 Low' };
const CAT_LABEL  = { security: '🔒 Security', bug: '🐛 Bug', performance: '⚡ Performance', style: '🎨 Style' };

function fillAndScan(repo) {
  document.getElementById('repoInput').value = repo;
  startScan();
}

function startScan() {
  const repo = document.getElementById('repoInput').value.trim();
  if (!repo) return;

  // Reset UI
  if (currentSource) currentSource.close();
  findingCount = 0;
  document.getElementById('findingsList').innerHTML = '';
  document.getElementById('statsRow').className = 'stats-row';
  document.getElementById('statsRow').innerHTML = '';
  document.getElementById('findingsHeader').className = 'findings-header';
  document.getElementById('emptyState').className = 'empty-state';
  document.getElementById('errorMsg').className = 'error-msg';
  document.getElementById('scanBtn').disabled = true;
  document.getElementById('statusBar').className = 'status-bar show';

  const model = document.getElementById('modelSelect').value;
  const session = getSessionId();
  const url = `/scan?repo=${encodeURIComponent(repo)}&commits=20&model=${encodeURIComponent(model)}&session=${encodeURIComponent(session)}`;
  currentSource = new EventSource(url);

  currentSource.onmessage = (e) => {
    const data = JSON.parse(e.data);

    if (data.event === 'start' || data.event === 'progress') {
      document.getElementById('statusText').textContent = data.message;
      return;
    }

    if (data.event === 'finding') {
      renderFinding(data.finding);
      return;
    }

    if (data.event === 'done') {
      currentSource.close();
      document.getElementById('statusBar').className = 'status-bar';
      document.getElementById('scanBtn').disabled = false;
      renderStats(data.stats);
      if (findingCount === 0) {
        document.getElementById('emptyState').className = 'empty-state show';
      }
      return;
    }

    if (data.event === 'error') {
      currentSource.close();
      document.getElementById('statusBar').className = 'status-bar';
      document.getElementById('scanBtn').disabled = false;
      const el = document.getElementById('errorMsg');
      el.textContent = '⚠ ' + data.message;
      el.className = 'error-msg show';
      return;
    }
  };

  currentSource.onerror = () => {
    currentSource.close();
    document.getElementById('statusBar').className = 'status-bar';
    document.getElementById('scanBtn').disabled = false;
    const el = document.getElementById('errorMsg');
    el.textContent = '⚠ Connection lost. Check that the server is running.';
    el.className = 'error-msg show';
  };
}

function renderFinding(f) {
  findingCount++;
  if (findingCount === 1) {
    document.getElementById('findingsHeader').className = 'findings-header show';
  }

  const card = document.createElement('div');
  card.className = 'finding-card';
  card.innerHTML = `
    <div class="finding-top">
      <div class="sev-dot ${SEV_CLASS[f.severity] || 'sev-low'}"></div>
      <div class="finding-title">${escHtml(f.title)}</div>
      <div class="confidence">${Math.round(f.confidence * 100)}%</div>
    </div>
    <div class="finding-meta">
      <span class="meta-tag ${CAT_CLASS[f.category] || ''}">${CAT_LABEL[f.category] || f.category}</span>
      <span class="meta-tag">${SEV_LABEL[f.severity] || f.severity}</span>
      <span class="meta-tag">${escHtml(f.file)}:${f.line_start}</span>
    </div>
    <div class="finding-desc">${escHtml(f.description)}</div>
    <div class="finding-fix"><strong>Fix:</strong> ${escHtml(f.suggestion)}</div>
  `;
  document.getElementById('findingsList').appendChild(card);
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderStats(stats) {
  const row = document.getElementById('statsRow');
  row.innerHTML = `
    <div class="stat-pill"><strong>${stats.findings}</strong> <span>findings</span></div>
    <div class="stat-pill"><strong>${stats.files}</strong> <span>files analyzed</span></div>
    <div class="stat-pill"><strong>${(stats.lines || 0).toLocaleString()}</strong> <span>lines</span></div>
    <div class="stat-pill"><strong>${stats.elapsed}s</strong> <span>elapsed</span></div>
  `;
  row.className = 'stats-row show';
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>"""
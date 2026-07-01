"""
RepoSentinel · analyzer.py
Orchestrates code analysis: git diff → static tools → LLM reasoning → structured findings.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# LLM provider selection
# ---------------------------------------------------------------------------

def _llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").strip().lower()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

Severity = Literal["critical", "high", "medium", "low"]
Category = Literal["bug", "performance", "security", "style"]


@dataclass
class Finding:
    category: Category
    severity: Severity
    file: str
    line_start: int
    line_end: int
    title: str
    description: str
    suggestion: str
    confidence: float  # 0.0 – 1.0


@dataclass
class AnalysisResult:
    findings: list[Finding] = field(default_factory=list)
    files_analyzed: int = 0
    lines_analyzed: int = 0
    static_tool_output: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def get_changed_files(base_ref: str = "HEAD~1", head_ref: str = "HEAD") -> list[Path]:
    """Return list of changed Python/JS/TS files between two refs."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACM", base_ref, head_ref],
        capture_output=True, text=True, check=True,
    )
    extensions = {".py", ".js", ".ts", ".jsx", ".tsx"}
    return [
        Path(f) for f in result.stdout.splitlines()
        if Path(f).suffix in extensions and Path(f).exists()
    ]


def get_diff(base_ref: str = "HEAD~1", head_ref: str = "HEAD") -> str:
    """Return unified diff between two refs, truncated to ~50k chars."""
    result = subprocess.run(
        ["git", "diff", base_ref, head_ref, "--unified=5"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout[:50_000]


def read_file_with_lines(path: Path) -> tuple[str, int]:
    """Read file content and return (content, line_count)."""
    content = path.read_text(errors="replace")
    return content, content.count("\n") + 1


# ---------------------------------------------------------------------------
# Static analysis
# ---------------------------------------------------------------------------

def run_ruff(files: list[Path]) -> str:
    """Run ruff linter on Python files, return JSON output."""
    py_files = [f for f in files if f.suffix == ".py"]
    if not py_files:
        return ""
    try:
        result = subprocess.run(
            ["ruff", "check", "--output-format=json", *[str(f) for f in py_files]],
            capture_output=True, text=True,
        )
        return result.stdout or result.stderr
    except FileNotFoundError:
        return '{"error": "ruff not installed"}'


def run_bandit(files: list[Path]) -> str:
    """Run bandit security scanner on Python files, return JSON output."""
    py_files = [f for f in files if f.suffix == ".py"]
    if not py_files:
        return ""
    try:
        result = subprocess.run(
            ["bandit", "-f", "json", "-q", *[str(f) for f in py_files]],
            capture_output=True, text=True,
        )
        return result.stdout or "{}"
    except FileNotFoundError:
        return '{"error": "bandit not installed"}'


def run_eslint(files: list[Path]) -> str:
    """Run eslint on JS/TS files, return JSON output."""
    js_files = [f for f in files if f.suffix in {".js", ".ts", ".jsx", ".tsx"}]
    if not js_files:
        return ""
    try:
        result = subprocess.run(
            ["npx", "eslint", "--format=json", *[str(f) for f in js_files]],
            capture_output=True, text=True,
        )
        return result.stdout or "[]"
    except FileNotFoundError:
        return '{"error": "eslint not available"}'


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are RepoSentinel, an expert code review agent specialising in:
    - Bugs and logic errors (off-by-one, null dereference, race conditions, wrong types)
    - Performance issues (N+1 queries, unnecessary re-renders, blocking I/O, memory leaks)
    - Security vulnerabilities (injection, hardcoded secrets, insecure defaults, SSRF)
    - Code style and maintainability (dead code, overly complex functions, missing error handling)

    You will receive:
    1. A git diff of recently changed files
    2. Static analysis tool output (ruff, bandit, eslint)
    3. The full content of changed files

    Your job: identify real, actionable issues. Be precise about file and line numbers.
    Avoid false positives — only report issues you are confident about.
    Do NOT report issues already clearly covered by the static tools (avoid duplication).
    Focus on semantic issues that static tools miss: logical bugs, architectural problems,
    subtle security flaws, and non-obvious performance traps.

    Respond ONLY with a valid JSON array of findings. No markdown, no preamble.
    Each finding must follow this exact schema:
    {
      "category": "bug" | "performance" | "security" | "style",
      "severity": "critical" | "high" | "medium" | "low",
      "file": "<relative file path>",
      "line_start": <integer>,
      "line_end": <integer>,
      "title": "<short title, max 80 chars>",
      "description": "<clear explanation of the issue, 1-3 sentences>",
      "suggestion": "<concrete fix or recommendation, 1-3 sentences>",
      "confidence": <float 0.0-1.0>
    }

    If you find no issues, return an empty array: []
""").strip()


def _build_prompt(
    diff: str,
    file_contents: dict[str, str],
    static_output: str,
) -> str:
    file_block = "\n\n".join(
        f"### {path}\n```\n{content[:8_000]}\n```"
        for path, content in file_contents.items()
    )
    return textwrap.dedent(f"""
        ## Git diff
        ```diff
        {diff[:20_000]}
        ```

        ## Static analysis output
        ```json
        {static_output[:5_000]}
        ```

        ## Full file contents
        {file_block}
    """).strip()


def _parse_findings(raw: str, confidence_threshold: float) -> list[Finding]:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    return [Finding(**item) for item in data if item.get("confidence", 0) >= confidence_threshold]


def _get_model(default_anthropic: str = "claude-sonnet-4-20250514", default_gemini: str = "gemini-2.0-flash") -> str:
    return os.getenv("LLM_MODEL") or (
        default_gemini if _llm_provider() == "gemini" else default_anthropic
    )


def _query_anthropic(system: str, user_message: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=_get_model(),
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text.strip()


def _query_gemini(system: str, user_message: str) -> str:
    from google import genai
    config = genai.types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=4096,
    )
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=_get_model(),
        contents=user_message,
        config=config,
    )
    return response.text.strip()


def query_llm(
    diff: str,
    file_contents: dict[str, str],
    static_output: str,
    confidence_threshold: float = 0.7,
    system_prompt_override: str | None = None,
) -> list[Finding]:
    """Send code context to LLM and parse structured findings.

    Provider is selected via the LLM_PROVIDER env var (anthropic | gemini).

    system_prompt_override: replaces SYSTEM_PROMPT entirely — used by the
    A/B eval harness to compare prompt variants without changing the codebase.
    """
    system = system_prompt_override if system_prompt_override else SYSTEM_PROMPT
    user_message = _build_prompt(diff, file_contents, static_output)

    provider = _llm_provider()
    if provider == "gemini":
        raw = _query_gemini(system, user_message)
    else:
        raw = _query_anthropic(system, user_message)

    return _parse_findings(raw, confidence_threshold)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze(
    base_ref: str = "HEAD~1",
    head_ref: str = "HEAD",
    confidence_threshold: float = 0.7,
    run_id: str = "",
    repo: str = "",
) -> AnalysisResult:
    """
    Full analysis pipeline (v2): git diff → static tools →
    historical context → LLM → classifier filter → semantic dedup.
    """
    # vector_store and trainer are optional — ImportError means not installed,
    # which is expected in lightweight deployments. Any other exception is a
    # real problem that should be logged, not silently swallowed.
    store = None
    classifier = None
    try:
        from agent.vector_store import FindingStore
        from agent.trainer import FindingClassifier
        store = FindingStore()
        classifier = FindingClassifier()
    except ImportError:
        pass  # chromadb / scikit-learn not installed — run without extensions
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "vector_store/classifier unavailable: %s", exc, exc_info=True
        )

    result = AnalysisResult()

    try:
        changed_files = get_changed_files(base_ref, head_ref)
        if not changed_files:
            return result

        result.files_analyzed = len(changed_files)

        file_contents: dict[str, str] = {}
        for path in changed_files:
            content, lines = read_file_with_lines(path)
            file_contents[str(path)] = content
            result.lines_analyzed += lines

        diff = get_diff(base_ref, head_ref)

        ruff_out = run_ruff(changed_files)
        bandit_out = run_bandit(changed_files)
        eslint_out = run_eslint(changed_files)
        combined_static = json.dumps({
            "ruff": json.loads(ruff_out) if ruff_out and not ruff_out.startswith('{"error') else [],
            "bandit": json.loads(bandit_out) if bandit_out else {},
            "eslint": json.loads(eslint_out) if eslint_out and not eslint_out.startswith('{"error') else [],
        }, indent=2)
        result.static_tool_output = combined_static

        # Inject historical context from vector store
        historical_context = ""
        if store:
            similar = store.similar_context(diff[:500], n=3, outcome_filter="accepted")
            if similar:
                historical_context = "\n## Previously accepted similar findings\n" + "\n".join(
                    f"- [{s['finding']['severity']}] {s['finding']['title']} (similarity={s['similarity']:.2f})"
                    for s in similar
                )

        raw_findings = query_llm(
            diff=diff,
            file_contents=file_contents,
            static_output=combined_static + historical_context,
            confidence_threshold=0.0,
        )

        # Classifier or threshold filter
        if classifier:
            filtered = [f for f in raw_findings if classifier.should_report(f)]
        else:
            filtered = [f for f in raw_findings if f.confidence >= confidence_threshold]

        # Semantic deduplication
        if store and filtered:
            dupe_indices = store.find_duplicates(filtered)
            filtered = [f for i, f in enumerate(filtered) if i not in dupe_indices]
            store.add(filtered, run_id=run_id, repo=repo, outcome="pending")

        result.findings = filtered

    except Exception as exc:
        result.error = str(exc)

    return result
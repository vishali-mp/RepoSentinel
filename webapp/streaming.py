"""
RepoSentinel · webapp/streaming.py

Converts the blocking query_llm() call into a true async streaming generator.

The core challenge: Claude's API streams raw text token-by-token.
Our system prompt tells Claude to return a JSON array of findings:
  [{"category": "security", ...}, {"category": "bug", ...}]

We can't parse this until the full string arrives — but that defeats the
point of streaming. Instead we use a bracket-depth scanner to detect when
a complete top-level JSON object {...} has closed, parse it immediately,
and yield it as a Finding. This way findings appear one at a time as Claude
generates them, not all at once at the end.

Key concept: partial JSON parsing via bracket counting
  - depth=0, we see '{' → depth=1 (start of a new finding)
  - depth=1+, we accumulate characters
  - depth=1, we see '}' → depth=0 → we have a complete object → parse it
  - We handle strings carefully (ignore brackets inside "...")
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.analyzer import Finding, SYSTEM_PROMPT


def _llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").strip().lower()


def _get_model() -> str:
    return os.getenv("LLM_MODEL") or (
        "gemini-2.0-flash" if _llm_provider() == "gemini" else "claude-sonnet-4-20250514"
    )


# ---------------------------------------------------------------------------
# Partial JSON parser — the tricky bit
# ---------------------------------------------------------------------------

class FindingStreamParser:
    """
    Incrementally scans a character stream for complete JSON objects.

    State machine:
      - Tracks { } depth to know when a top-level object is complete
      - Tracks whether we're inside a string literal (to ignore { } in strings)
      - Tracks escape sequences inside strings (to handle \\")
    """

    def __init__(self):
        self._buf   = ""      # accumulates raw text from the stream
        self._depth = 0       # current brace nesting depth
        self._in_string = False
        self._escaped   = False
        self._obj_start = -1  # index where the current top-level { began

    def feed(self, chunk: str) -> list[Finding]:
        """
        Feed a new chunk of text. Returns any complete Finding objects
        that were completed by this chunk.
        """
        completed: list[Finding] = []

        for i, ch in enumerate(chunk):
            abs_i = len(self._buf) + i   # position in the full accumulated stream

            # Track string boundaries and escape sequences
            if self._escaped:
                self._escaped = False
                continue

            if ch == "\\" and self._in_string:
                self._escaped = True
                continue

            if ch == '"' and not self._escaped:
                self._in_string = not self._in_string
                continue

            # Only count braces outside strings
            if not self._in_string:
                if ch == "{":
                    if self._depth == 0:
                        self._obj_start = len(self._buf) + i
                    self._depth += 1

                elif ch == "}" and self._depth > 0:
                    self._depth -= 1
                    if self._depth == 0 and self._obj_start >= 0:
                        # We have a complete top-level object
                        raw_obj = (self._buf + chunk)[self._obj_start : len(self._buf) + i + 1]
                        try:
                            data = json.loads(raw_obj)
                            finding = Finding(**data)
                            completed.append(finding)
                        except (json.JSONDecodeError, TypeError):
                            pass  # Malformed — skip this finding
                        self._obj_start = -1

        self._buf += chunk
        return completed


# ---------------------------------------------------------------------------
# Streaming analyzer
# ---------------------------------------------------------------------------

def _build_message(diff: str, file_contents: dict[str, str], static_output: str) -> str:
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
        {static_output[:3_000]}
        ```

        ## Full file contents
        {file_block}
    """).strip()


async def _stream_anthropic(
    system_prompt: str, user_message: str, parser: FindingStreamParser, confidence_threshold: float,
) -> AsyncIterator[Finding]:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    async with client.messages.stream(
        model=_get_model(),
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for text_chunk in stream.text_stream:
            findings = parser.feed(text_chunk)
            for f in findings:
                if f.confidence >= confidence_threshold:
                    yield f


async def _stream_gemini(
    system_prompt: str, user_message: str, parser: FindingStreamParser, confidence_threshold: float,
) -> AsyncIterator[Finding]:
    from google import genai
    config = genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=4096,
    )
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    stream = await client.aio.models.generate_content_stream(
        model=_get_model(),
        contents=user_message,
        config=config,
    )
    async for chunk in stream:
        if chunk.text:
            findings = parser.feed(chunk.text)
            for f in findings:
                if f.confidence >= confidence_threshold:
                    yield f


async def stream_findings(
    diff: str,
    file_contents: dict[str, str],
    static_output: str = "{}",
    confidence_threshold: float = 0.65,
    system_prompt: str = SYSTEM_PROMPT,
) -> AsyncIterator[Finding]:
    """
    Async generator that yields Finding objects one at a time as the LLM
    generates them, using streaming API.

    Provider is selected via the LLM_PROVIDER env var (anthropic | gemini).

    Usage:
        async for finding in stream_findings(diff, file_contents):
            yield finding
    """
    user_message = _build_message(diff, file_contents, static_output)
    parser = FindingStreamParser()

    provider = _llm_provider()
    if provider == "gemini":
        async for f in _stream_gemini(system_prompt, user_message, parser, confidence_threshold):
            yield f
    else:
        async for f in _stream_anthropic(system_prompt, user_message, parser, confidence_threshold):
            yield f

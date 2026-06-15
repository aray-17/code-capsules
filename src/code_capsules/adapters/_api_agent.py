"""
Shared tool loop for API-based ModelClient implementations.

Unlike ClaudeCLIClient which delegates to the Anthropic Claude CLI (a
turnkey coding agent), API-based providers expose a chat-completions /
responses API with function calling. To make them usable as coding agents,
we wrap the API in a multi-turn tool loop here.

Tool definitions match what claude provides natively: Read, Write, Edit, Bash.
A subclass implements `_call_api(messages, tools)` returning `_ProviderResponse`;
this base class drives the loop end-to-end.

Tool execution is sandboxed to `cwd` — no path can escape the worktree.
Bash runs in a subprocess with a wall-clock cap. Each tool call result is
fed back to the model as a tool_result message.

Session resume: invocations keyed by `session_id` cache their conversation
history in `_SESSIONS`. A `resume=` call replays the cached history before
the new prompt. This is an in-process cache; cross-process resume requires
a persistence backend (see _SESSIONS docstring).
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from code_capsules.api import InvocationResult


# Cross-call session cache (per-process). Keys: session_id. Values: list of
# provider-specific message dicts (the conversation history). Persistence
# across processes / harness restarts is the caller's job — for benchmarking
# runs that stay in one Python process this is sufficient.
_SESSIONS: dict[str, list[Any]] = {}


@dataclass
class ToolCall:
    """Compatible with code_capsules.runtime.stream_parser.ToolCallRecord shape.

    Read by signals (file_thrash, test_failure, traceback, etc.) and by the
    enriched stage-2 prompt builder. The same property names are required.
    """
    name: str
    file_path: Optional[str] = None
    input_size: int = 0
    output_size: int = 0
    turn_index: int = 0
    input_tokens_this_turn: int = 0
    output_tokens_this_turn: int = 0
    output_text: str = ""

    @property
    def is_read(self) -> bool:
        return self.name in ("Read", "ListDirectory")

    @property
    def is_write(self) -> bool:
        return self.name in ("Write", "Edit")

    @property
    def is_bash(self) -> bool:
        return self.name == "Bash"


@dataclass
class _ProviderResponse:
    """Provider-agnostic envelope returned by _call_api."""
    text: str
    tool_calls: list[dict]      # list of {name, args, call_id}
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0   # of input_tokens, how many were cache hits
                                    # (OpenAI auto-cache, Gemini context cache)
    stop_reason: str = "end_turn"   # "end_turn" / "tool_use" / "max_tokens" / "error"
    error: Optional[str] = None
    raw_message: Any = None     # opaque; subclass may need to append to history


# ── Tool definitions in vendor-neutral form ─────────────────────────────────

TOOL_DEFS = [
    {
        "name": "Read",
        "description": "Read the contents of a file in the working directory. "
                       "Returns the file contents as a string with line numbers "
                       "prefixed. For large files, use `offset` (1-indexed line "
                       "to start at) and `limit` (max lines to return) to "
                       "navigate without re-reading the same range.",
        "parameters": {
            "type": "object",
            "properties": {
                "path":   {"type": "string", "description": "Relative file path"},
                "offset": {"type": "integer", "description": "1-indexed start line (default 1)"},
                "limit":  {"type": "integer", "description": "max lines to return (default 2000)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "Write",
        "description": "Create a new file or replace an existing file with the "
                       "given content. The path argument is relative to the "
                       "working directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": "Replace exactly one occurrence of `old_string` with "
                       "`new_string` in the file at `path`. Fails if old_string "
                       "does not appear exactly once. Use Read first to verify "
                       "the exact text.",
        "parameters": {
            "type": "object",
            "properties": {
                "path":       {"type": "string", "description": "Relative file path"},
                "old_string": {"type": "string", "description": "Exact text to replace"},
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "Bash",
        "description": "Run a shell command in the working directory. Returns "
                       "stdout + stderr (first 4000 chars). Use for running "
                       "tests, listing files, exploring directory structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command"},
            },
            "required": ["command"],
        },
    },
]


# ── Tool execution ──────────────────────────────────────────────────────────

def _safe_join(cwd: Path, rel: str) -> Path:
    """Resolve `rel` against `cwd`; reject paths that escape the worktree."""
    p = (cwd / rel).resolve()
    cwd_resolved = cwd.resolve()
    try:
        p.relative_to(cwd_resolved)
    except ValueError:
        raise ValueError(f"path escapes working directory: {rel!r}")
    return p


def _execute_tool(name: str, args: dict, cwd: Path,
                  *, bash_timeout: int = 60) -> tuple[str, int]:
    """Execute one tool call. Returns (output_text, output_size_bytes).

    Errors are returned as text rather than raised — the model can recover.
    """
    try:
        if name == "Read":
            path = _safe_join(cwd, args["path"])
            if not path.exists():
                return f"Error: file not found: {args['path']}", 0
            offset = max(1, int(args.get("offset", 1)))
            limit = max(1, int(args.get("limit", 2000)))
            all_lines = path.read_text(errors="replace").splitlines()
            total = len(all_lines)
            start_idx = offset - 1
            end_idx = min(start_idx + limit, total)
            if start_idx >= total:
                return (f"Error: offset {offset} past end of file "
                        f"(file has {total} lines)"), 0
            # Prefix each line with its 1-indexed line number so the model
            # can navigate without ambiguity. Cap each line at 2000 chars.
            rendered = "\n".join(
                f"{i + 1:6d}\t{all_lines[i][:2000]}"
                for i in range(start_idx, end_idx)
            )
            footer = ""
            if end_idx < total:
                footer = (f"\n\n[truncated — file has {total} lines; "
                          f"read more with offset={end_idx + 1}]")
            elif offset > 1:
                footer = f"\n\n[end of file at line {total}]"
            out = rendered + footer
            # Hard ceiling at 24000 chars to bound prompt growth, but much
            # bigger than the old 8000-char cap.
            return out[:24000], len(out[:24000])

        if name == "Write":
            path = _safe_join(cwd, args["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            content = args["content"]
            path.write_text(content)
            return f"Wrote {len(content)} chars to {args['path']}", len(content)

        if name == "Edit":
            path = _safe_join(cwd, args["path"])
            if not path.exists():
                return f"Error: file not found: {args['path']}", 0
            old = args["old_string"]
            new = args["new_string"]
            content = path.read_text()
            n = content.count(old)
            if n == 0:
                return f"Error: old_string not found in {args['path']}", 0
            if n > 1:
                return (f"Error: old_string appears {n} times in {args['path']}; "
                        "must be unique. Add more context."), 0
            path.write_text(content.replace(old, new))
            return f"Edited {args['path']}", len(new)

        if name == "Bash":
            cmd = args["command"]
            try:
                proc = subprocess.run(
                    ["bash", "-c", cmd], cwd=cwd, capture_output=True,
                    text=True, timeout=bash_timeout,
                )
                out = (proc.stdout + proc.stderr)[:4000]
                return out, len(out)
            except subprocess.TimeoutExpired:
                return f"Error: command timed out after {bash_timeout}s", 0

        return f"Error: unknown tool {name!r}", 0
    except Exception as exc:
        return f"Error: {exc}", 0


# ── Base agent class ────────────────────────────────────────────────────────

class APIBasedAgent:
    """Base class for ModelClient impls driving an API-based provider.

    Subclasses must:
      - Set `name` and `default_model`
      - Implement `_initial_messages(prompt, system) -> list[dict]`
      - Implement `_append_tool_results(messages, tool_calls, results) -> list[dict]`
      - Implement `_call_api(model, messages, tools, max_tokens) -> _ProviderResponse`

    The base handles: tool loop, sandboxed tool execution, telemetry rollup,
    session caching, error containment.
    """

    name: str = "api_agent"
    default_model: str = ""

    # Subclass override if API uses a different system prompt key
    _system_prompt = (
        "You are a coding agent. Use the Read, Write, Edit, and Bash tools to "
        "explore the codebase, make minimal edits to source files, and run "
        "tests to verify. Do not modify files under tests/ — those are owned "
        "by the test harness. When the task is complete, respond with a brief "
        "summary (no tool calls)."
    )

    def invoke(
        self,
        prompt: str,
        *,
        cwd: Any,
        max_turns: int,
        timeout: int = 600,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        resume: Optional[str] = None,
    ) -> InvocationResult:
        cwd_path = Path(cwd)
        model = model or self.default_model
        t0 = time.monotonic()

        # Start fresh or resume from cached history
        if resume and resume in _SESSIONS:
            messages = list(_SESSIONS[resume])  # copy
            messages = self._append_user_message(messages, prompt)
        else:
            messages = self._initial_messages(prompt, self._system_prompt)

        tool_calls_collected: list[ToolCall] = []
        cum_input = 0
        cum_output = 0
        cum_cached_input = 0
        last_text = ""
        error: Optional[str] = None
        stop_reason = "end_turn"

        for turn_idx in range(1, max_turns + 1):
            if time.monotonic() - t0 > timeout:
                error = f"agent timeout after {timeout}s"
                break

            # Retry-on-degenerate-response: if the API returns no text, no
            # tool calls, AND no error, it's almost always a transient
            # capacity / throttling issue (observed on Gemini AI Studio
            # free tier under parallel load — silent truncation, no 503).
            # Up to 3 retries with exponential backoff before giving up.
            resp = None
            for retry_idx in range(3):
                try:
                    resp = self._call_api(model, messages, TOOL_DEFS,
                                           max_tokens=4096)
                except Exception as exc:
                    error = f"api error: {exc}"
                    resp = None
                    break
                degenerate = (
                    not resp.error
                    and not resp.tool_calls
                    and not (resp.text or "").strip()
                    and resp.output_tokens < 5
                )
                if not degenerate:
                    break
                # Backoff: 2s, 8s, 32s
                time.sleep(2 * (4 ** retry_idx))
            if resp is None:
                break

            cum_input += resp.input_tokens
            cum_output += resp.output_tokens
            cum_cached_input += resp.cached_input_tokens
            last_text = resp.text or last_text
            stop_reason = resp.stop_reason

            if resp.error:
                error = resp.error
                break

            # Append the assistant message so the next API call has the
            # full history (provider-specific shape)
            messages = self._append_assistant_message(
                messages, resp.text, resp.tool_calls, raw=resp.raw_message,
            )

            if not resp.tool_calls:
                # Empty tool-call response. Two cases to distinguish:
                #   (a) genuine completion — model made progress earlier
                #       (read/wrote/ran) and is now wrapping up
                #   (b) indecision — model emitted brief acknowledgement
                #       text without ever calling a tool ("I'll start now",
                #       "Sure, here's my plan: ...")
                # Without this guard, (b) silently exits the loop and the
                # task never gets done. Observed on Gemini in cross-vendor
                # HumanEval: 17/44 failures were this pattern with
                # turn=2/out=15 tokens. Fix: re-prompt only when no tool
                # call has been made yet.
                made_progress = any(
                    getattr(tc, "is_read", False)
                    or getattr(tc, "is_write", False)
                    or getattr(tc, "is_bash", False)
                    for tc in tool_calls_collected
                )
                if not made_progress and turn_idx < max_turns:
                    messages = self._append_user_message(
                        messages,
                        "You haven't taken any actions yet. Use the "
                        "available tools (Read, Write, Edit, Bash) to make "
                        "progress on the task. Don't describe what you'll "
                        "do — do it.",
                    )
                    continue
                break

            # Execute tools and feed results back
            results: list[tuple[dict, str, int]] = []
            for tc_raw in resp.tool_calls:
                tname = tc_raw["name"]
                targs = tc_raw.get("args", {}) or {}
                output_text, output_size = _execute_tool(tname, targs, cwd_path)
                tc_record = ToolCall(
                    name=tname,
                    file_path=targs.get("path"),
                    input_size=len(str(targs)),
                    output_size=output_size,
                    turn_index=turn_idx,
                    input_tokens_this_turn=resp.input_tokens,
                    output_tokens_this_turn=resp.output_tokens,
                    output_text=output_text[:1500],
                )
                tool_calls_collected.append(tc_record)
                results.append((tc_raw, output_text, output_size))

            messages = self._append_tool_results(messages, results)

        # Cache session for resume
        if session_id:
            _SESSIONS[session_id] = messages

        return InvocationResult(
            text=last_text,
            tool_calls=tool_calls_collected,
            num_turns=turn_idx if not error else max(1, turn_idx - 1),
            input_tokens=cum_input,
            output_tokens=cum_output,
            cache_read_tokens=0,
            cached_input_tokens=cum_cached_input,
            cost_usd=0.0,        # let the CostModel compute from tokens
            duration_ms=(time.monotonic() - t0) * 1000.0,
            session_id=session_id or "",
            model=model,
            error=error,
        )

    # ── Subclass hooks ──────────────────────────────────────────────────────

    def _initial_messages(self, prompt: str, system: str) -> list:
        raise NotImplementedError

    def _append_user_message(self, messages: list, prompt: str) -> list:
        raise NotImplementedError

    def _append_assistant_message(self, messages: list, text: str,
                                   tool_calls: list[dict],
                                   raw: Any = None) -> list:
        raise NotImplementedError

    def _append_tool_results(self, messages: list,
                              results: list[tuple[dict, str, int]]) -> list:
        raise NotImplementedError

    def _call_api(self, model: str, messages: list, tools: list[dict],
                  max_tokens: int) -> _ProviderResponse:
        raise NotImplementedError


__all__ = ["APIBasedAgent", "ToolCall", "TOOL_DEFS"]

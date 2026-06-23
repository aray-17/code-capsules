"""
Shared `claude -p` subprocess invocation with transient-error retry.

Lifted from tools/run_swebench_docker.py's _run_claude_with_500_retry +
_looks_like_transient_500. Centralized here so all variants share the same
proven retry behavior.

Variants don't import this directly; they call into _orchestration which uses
it. Kept as a separate module to make the retry policy easy to swap.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional


_API_ERR_PATTERNS = (
    "API Error: 500", "Internal server error",
    "API Error: 502", "API Error: 503", "API Error: 529",
    "service_unavailable",
    "API Error: 429", "rate_limit_error", "rate_limit_exceeded",
    "Rate limit exceeded", "rate limit reached", "Too Many Requests",
)
_RETRY_DELAYS_S = (15, 60, 180)


def _looks_like_transient(stdout: str) -> bool:
    if not stdout:
        return False
    head = stdout[:50_000]
    if not any(p in head for p in _API_ERR_PATTERNS):
        return False
    # If we got any tool_use, model produced *something* - don't retry.
    if '"type":"tool_use"' in head:
        return False
    return True


def run_claude(
    prompt: str,
    cwd: Path,
    *,
    max_turns: int,
    timeout: int = 600,
    model: Optional[str] = None,
    session_id: Optional[str] = None,
    resume: Optional[str] = None,
    max_retries: int = 3,
) -> subprocess.CompletedProcess:
    """Invoke `claude -p` with stream-json output and transient-error retry.

    Exactly one of session_id / resume should be passed (or neither for a
    fresh ephemeral session). Returns the CompletedProcess from subprocess.
    """
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "stream-json", "--verbose",
        "--max-turns", str(max_turns),
        "--permission-mode", "bypassPermissions",
    ]
    if session_id is not None:
        cmd.extend(["--session-id", session_id])
    if resume is not None:
        cmd.extend(["--resume", resume])
    if model is not None:
        cmd.extend(["--model", model])

    env = {**os.environ}
    attempt = 0
    last_proc = None
    while True:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=cwd, timeout=timeout, env=env,
        )
        last_proc = proc
        if not _looks_like_transient(proc.stdout):
            return proc
        if attempt >= max_retries:
            return last_proc
        delay = _RETRY_DELAYS_S[min(attempt, len(_RETRY_DELAYS_S) - 1)]
        print(f"    [retry] transient API error; sleeping {delay}s "
              f"(attempt {attempt + 1}/{max_retries})", flush=True)
        time.sleep(delay)
        attempt += 1


def capture_patch(worktree: Path) -> str:
    """Return `git diff HEAD` for the worktree (preserving trailing newline).

    Note: caller is responsible for any test-file stripping. This function
    returns the raw diff so domain-specific gates can decide their own policy.
    """
    try:
        r = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=worktree, capture_output=True, text=True, timeout=30,
        )
        return r.stdout
    except Exception:
        return ""


__all__ = ["run_claude", "capture_patch"]

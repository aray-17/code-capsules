"""
ClaudeCLIClient: drives the Anthropic Claude CLI as a coding agent.

Wraps the proven subprocess invocation from code_capsules.variants._runner +
stream-json parsing from code_capsules.runtime.stream_parser into a ModelClient.

The CLI handles the multi-turn tool-use loop end-to-end (file reads, edits,
bash, etc.) - we just give it the prompt + budget + worktree, and parse
back tokens/tool-calls/text/errors when it completes.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional

from code_capsules.api import InvocationResult


class ClaudeCLIClient:
    """ModelClient that shells out to `claude -p` and parses stream-json.

    Defaults match Anthropic's Claude Code CLI conventions. To target a
    specific model tier, pass `model=` (haiku/sonnet/opus or full ID); the
    CLI's default is used when None.
    """

    name = "claude_cli"

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
        from code_capsules.variants._runner import run_claude
        from code_capsules.runtime.stream_parser import parse_stream

        cwd_path = Path(cwd)
        try:
            proc = run_claude(
                prompt, cwd=cwd_path, max_turns=max_turns,
                timeout=timeout, model=model,
                session_id=session_id, resume=resume,
            )
        except subprocess.TimeoutExpired:
            return InvocationResult(
                error=f"claude timeout after {timeout}s",
                duration_ms=timeout * 1000.0,
            )

        session = parse_stream(proc.stdout)
        return InvocationResult(
            text=session.final_text,
            tool_calls=session.tool_calls,
            num_turns=session.num_turns,
            input_tokens=session.effective_input_tokens,
            output_tokens=session.total_output_tokens,
            cache_read_tokens=session.total_cache_read_tokens,
            cost_usd=session.cost_usd,
            duration_ms=session.duration_ms,
            session_id=session.session_id,
            model=session.model,
            error=session.error,
            raw_stdout=proc.stdout,
        )


__all__ = ["ClaudeCLIClient"]

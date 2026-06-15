"""
GenericAgentClient — skeleton + contract documentation for ModelClient extensions.

This class is NOT a working client; it's the reference impl users copy when
plugging in a non-Claude coding agent runtime. The framework's contract is
documented inline (`invoke` raises NotImplementedError with a checklist).

Common targets you might adapt:

  - **Aider** — pip install aider-chat; runs in a worktree with --message-file.
    Adapt: subprocess.run(["aider", "--model", model, "--message-file", ...])
    Parse: aider's chat history file for token counts; git diff for the patch.

  - **OpenAI Codex / Code Interpreter via Assistants API** — multi-turn loop.
    Adapt: openai.Client().beta.assistants.create + thread + runs.create.
    Parse: run.usage for tokens; run_step.steps for tool calls.

  - **OpenAI Responses API + custom tool loop** — you write the agent loop;
    framework only wraps it.
    Adapt: Loop {client.responses.create + your file/bash tools} until done.
    Parse: aggregate usage; map tool calls to {name, file_path, output_text}.

  - **Local llama.cpp / vLLM agent** — server-mode + your own tool loop.
    Adapt: same shape as the OpenAI custom loop, pointed at the local endpoint.

  - **MCP-server agent (any provider)** — Model Context Protocol gives you
    a uniform tool-call interface across providers. Highest portability path.

In all cases, the framework's contract is the same: invoke() returns an
InvocationResult with token / tool-call / text / cost / error fields. The
variant orchestration doesn't care which provider is behind the curtain.
"""
from __future__ import annotations

from typing import Any, Optional

from code_capsules.api import InvocationResult


class GenericAgentClient:
    """Skeleton showing what users must implement to plug in a new agent runtime.

    Constructor takes a `name` so multiple instances can co-exist in the
    registry (e.g. `openai_codex`, `aider`, `local_llama`).
    """

    def __init__(self, name: str):
        self.name = name

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
        """Override me. Required behavior:

        1. Run your agent runtime against `prompt` in directory `cwd`.
        2. Cap iterations at `max_turns`; cap wall-clock at `timeout` seconds.
        3. If `model` is set, pass it to your runtime (skip if your runtime
           is single-model).
        4. If `session_id` is set, start a new persistable session with that ID.
           If `resume` is set, continue from that session ID. The framework
           uses this for escalation + injection cycles — your runtime must
           support session resume to participate in those variants. If it
           doesn't, sequential-mode variants still work; raise here otherwise.
        5. Return InvocationResult with:
           - text:            final assistant text response
           - tool_calls:      list of objects with .name, .file_path, .output_text,
                              .is_read, .is_write, .is_bash properties
                              (the signal pipeline reads these)
           - num_turns:       actual turns used
           - input_tokens:    total billed input (including any cache reads)
           - output_tokens:   total output
           - cached_input_tokens: optional, default 0 — the portion of
                              input_tokens that hit the vendor's prompt cache.
                              The CostModel bills these at the discounted rate,
                              so reporting it is what earns the cache discount.
                              (cache_read_tokens is the legacy alias.)
           - cost_usd:        if your runtime reports cost; else leave 0 and
                              let the CostModel compute it from tokens
           - error:           string description if the run failed recoverably,
                              else None. Raise on truly unrecoverable errors.
        """
        raise NotImplementedError(
            f"{self.name!r}: implement invoke() — see GenericAgentClient docstring "
            f"for the contract. Copy this file and adapt to your runtime."
        )


__all__ = ["GenericAgentClient"]

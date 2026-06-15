"""
Parse `claude --print --output-format stream-json --verbose` output.

Each line of stdout is a JSON object (NDJSON). We extract:
- Tool calls (name, input summary, associated file paths)
- Per-turn token usage (input, output, cache)
- Final text response
- Session metadata (model, cost, duration, num_turns)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ToolCallRecord:
    """One tool invocation observed in the session."""
    name: str               # Write, Read, Bash, Edit, ...
    file_path: str | None   # path argument if Read/Write/Edit
    input_size: int         # len(str(input)) — proxy for input complexity
    output_size: int        # len(str(output)) — proxy for output size
    turn_index: int         # which assistant turn this came from
    input_tokens_this_turn: int   # input tokens for the turn containing this call
    output_tokens_this_turn: int
    # Truncated copy of the tool output (first ~1500 chars). Used by Phase 7C
    # runtime quality signals (traceback / test-failure detection) and any
    # downstream analysis that needs to inspect actual tool output, not just
    # its size. Default empty string keeps construction backwards-compatible.
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
class ParsedSession:
    """Result of parsing one claude stream-json session."""
    model: str
    tool_calls: list[ToolCallRecord]
    # Cumulative token counts across all turns
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_creation_tokens: int
    # Per-turn breakdown (index = turn number)
    turns_input_tokens: list[int] = field(default_factory=list)
    turns_output_tokens: list[int] = field(default_factory=list)
    final_text: str = ""
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    num_turns: int = 0
    session_id: str = ""
    error: str | None = None

    @property
    def total_billed_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def effective_input_tokens(self) -> int:
        """All tokens fed in (fresh + cache)."""
        return (
            self.total_input_tokens
            + self.total_cache_read_tokens
            + self.total_cache_creation_tokens
        )


def parse_stream(stdout: str) -> ParsedSession:
    """
    Parse NDJSON output from `claude --output-format stream-json --verbose`.
    Returns a ParsedSession with all tool calls and token counts.
    """
    model = "unknown"
    session_id = ""
    tool_calls: list[ToolCallRecord] = []
    turns_input: list[int] = []
    turns_output: list[int] = []
    total_input = total_output = total_cache_read = total_cache_create = 0
    final_text = ""
    cost_usd = 0.0
    duration_ms = 0.0
    num_turns = 0
    error: str | None = None

    # tool_use_id → ToolCallRecord (awaiting tool_result)
    pending: dict[str, ToolCallRecord] = {}
    # tool_use_id → turn_index
    pending_turn: dict[str, int] = {}

    turn_index = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "system" and event.get("subtype") == "init":
            model = event.get("model", "unknown")
            session_id = event.get("session_id", "")

        elif etype == "assistant":
            msg = event.get("message", {})
            usage = msg.get("usage", {})
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
            cr = usage.get("cache_read_input_tokens", 0)
            cc = usage.get("cache_creation_input_tokens", 0)

            turns_input.append(inp)
            turns_output.append(out)
            total_input += inp
            total_output += out
            total_cache_read += cr
            total_cache_create += cc

            for block in msg.get("content", []):
                btype = block.get("type", "")
                if btype == "tool_use":
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "")
                    inp_data = block.get("input", {})

                    file_path = (
                        inp_data.get("file_path")
                        or inp_data.get("path")
                        or inp_data.get("command", "")[:80]  # Bash command preview
                        or None
                    )
                    rec = ToolCallRecord(
                        name=tool_name,
                        file_path=file_path,
                        input_size=len(json.dumps(inp_data)),
                        output_size=0,
                        turn_index=turn_index,
                        input_tokens_this_turn=inp,
                        output_tokens_this_turn=out,
                    )
                    pending[tool_id] = rec
                    pending_turn[tool_id] = turn_index

                elif btype == "text":
                    final_text = block.get("text", "")

            turn_index += 1

        elif etype == "user":
            # Claude Code emits tool results as content blocks inside `user`
            # messages, not as top-level `tool_result` events. The previous
            # code looked for etype == "tool_result" at the top level and
            # never matched, leaving output_size = 0 on every tool call.
            # See stream-json schema:
            #   {"type": "user", "message": {"content": [
            #       {"type": "tool_result", "tool_use_id": "...", "content": "..."}
            #   ]}}
            msg = event.get("message", {})
            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue
                tool_id = block.get("tool_use_id", "")
                content = block.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                if tool_id in pending:
                    rec = pending.pop(tool_id)
                    content_str = str(content)
                    rec.output_size = len(content_str)
                    rec.output_text = content_str[:1500]
                    tool_calls.append(rec)

        elif etype == "tool_result":
            # Kept for backwards-compatibility in case some adapter emits
            # tool_result as a top-level event. Unused in current stream-json.
            tool_id = event.get("tool_use_id", "")
            content = event.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            if tool_id in pending:
                rec = pending.pop(tool_id)
                content_str = str(content)
                rec.output_size = len(content_str)
                rec.output_text = content_str[:1500]
                tool_calls.append(rec)

        elif etype == "result":
            cost_usd = event.get("total_cost_usd", 0.0)
            duration_ms = event.get("duration_ms", 0.0)
            num_turns = event.get("num_turns", turn_index)
            session_id = event.get("session_id", session_id)
            if event.get("is_error") or event.get("subtype") == "error":
                error = event.get("result", "unknown error")
            # The result event's aggregate usage is the authoritative token count.
            # Per-assistant-message output_tokens can miss tokens from Write tool
            # content (code generated inside tool_use inputs) — the result event
            # captures the true total across all turns.
            agg = event.get("usage", {})
            if agg:
                total_input = agg.get("input_tokens", total_input)
                total_output = agg.get("output_tokens", total_output)
                total_cache_read = agg.get("cache_read_input_tokens", total_cache_read)
                total_cache_create = agg.get("cache_creation_input_tokens", total_cache_create)

        elif etype == "rate_limit_event":
            pass  # ignore

    # Flush any tool calls that never got a result event (shouldn't happen)
    for rec in pending.values():
        tool_calls.append(rec)

    return ParsedSession(
        model=model,
        tool_calls=tool_calls,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cache_read_tokens=total_cache_read,
        total_cache_creation_tokens=total_cache_create,
        turns_input_tokens=turns_input,
        turns_output_tokens=turns_output,
        final_text=final_text,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        num_turns=num_turns,
        session_id=session_id,
        error=error,
    )

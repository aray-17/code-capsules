"""
Code-Capsules core data types.

The dataclasses every extension primitive exchanges: the task descriptor that
enters the framework, the variant config that parameterises a run, the session
state and finished attempt the signals/gates read, and the per-invocation and
per-run result records. These carry no behaviour, they are the typed payloads
that flow between the eight Protocols.

The Protocol interfaces themselves (the extension contract) and the catalogue of
shipped default implementations live in ``code_capsules.api.protocols``; the
name -> implementation registry lives in ``code_capsules.api.registry``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


# ── Supporting types ─────────────────────────────────────────────────────────

@dataclass
class TaskDescriptor:
    """A task to be solved by the framework. Domain-agnostic.

    Users provide the text (issue / prompt) and optional context (repo path,
    metadata). The framework's classifier receives this and produces a class
    label; the router picks a variant; the variant runs the task.
    """
    text: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantConfig:
    """Configuration describing how a Variant should run a task.

    Maps to harness flags (turn_budget, prompt_variant, preselect_top_n, etc.)
    but is harness-agnostic - variant implementations interpret it according
    to their orchestration mode.
    """
    name: str                                   # e.g. "plan_then_execute"
    mode: str                                   # sequential / escalating / injection_loop / cascade
    turn_budget: int = 20
    prompt_budget_hint: Optional[int] = None
    prompt_variant: Optional[str] = None        # plan_first / tool_alloc / phase_staged
    preselect_top_n: int = 0                    # 0 = no ranker; >0 = enable with top-N
    escalating_start_budget: Optional[int] = None
    escalating_target_budget: Optional[int] = None
    escalating_stage2_mode: Optional[str] = None
    always_escalate: bool = False
    injection_cycles: int = 0
    injection_per_cycle_turns: int = 0
    model: Optional[str] = None
    claude_timeout: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)  # for user-defined variants


@dataclass
class SessionState:
    """Snapshot of a session in progress or completed.

    Signals are computed from this. Signal implementations should not require
    any other context - pass everything in here.
    """
    tool_calls: list = field(default_factory=list)
    num_turns: int = 0
    effective_input_tokens: int = 0
    total_output_tokens: int = 0
    text_emitted: str = ""
    partial_patch: str = ""
    test_output: Optional[str] = None
    stage: int = 1                              # 1 = first attempt; 2+ = escalated
    elapsed_seconds: float = 0.0


@dataclass
class Attempt:
    """A finished attempt: a patch (or equivalent) and metadata about how
    it was produced. Quality gates evaluate Attempts.
    """
    patch: str
    metadata: dict[str, Any] = field(default_factory=dict)
    session_state: Optional[SessionState] = None


@dataclass
class InvocationResult:
    """One round-trip with a model agent runtime (one ModelClient.invoke call).

    Wraps the per-invocation telemetry the framework needs: tokens, tool
    activity, text, cost-input. Variants stitch one or more InvocationResults
    into a RunResult, computing cost via the active CostModel.

    The exact tool_calls shape is agent-runtime-specific; downstream code
    (signals, cascade triggers) treats them opaquely or accesses attributes
    via `getattr`. Where uniform tool-call telemetry matters,
    ModelClient implementations should return objects with the same shape as
    tools/stream_parser.ToolCallRecord (name, file_path, output_text,
    is_read/is_write/is_bash).
    """
    text: str = ""
    tool_calls: list = field(default_factory=list)
    num_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0           # legacy field (Anthropic prompt cache)
    cached_input_tokens: int = 0         # of `input_tokens`, how many were cache hits
                                          # (OpenAI auto-cache, Gemini context cache).
                                          # CostModel uses this to bill cached tokens
                                          # at the vendor's discount rate.
    cost_usd: float = 0.0          # if the runtime reports it directly; else 0.0
    duration_ms: float = 0.0
    session_id: str = ""
    model: str = ""
    error: Optional[str] = None
    raw_stdout: str = ""           # opaque; preserved for stream archiving


@dataclass
class RunResult:
    """Outcome of one Variant.run() call. Returned to the caller; logged.

    Resolved is the workload's success signal (gate-defined); cost_usd is
    derived from token usage via the active CostModel.
    """
    task_id: str
    resolved: Optional[bool]                    # True / False / None (unknown)
    patch: str
    num_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    stage_records: list = field(default_factory=list)   # for cascade / escalating modes
    signals_fired: list = field(default_factory=list)



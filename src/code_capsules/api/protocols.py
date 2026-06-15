"""
Code-Capsules extension API.

This module defines the eight Protocol interfaces that constitute Code-Capsules'
contribution surface. Users (developers, deployers, researchers) extend the
framework by implementing any of these Protocols against their own domain.
The framework ships concrete defaults for each, calibrated on SWE-bench Lite,
HumanEval, and MBPP — but the defaults are reference data, not the contribution.

## The eight extension primitives

WorkloadClassifier   — maps a task descriptor to a class label used for routing.
                       Registered: generic_prompt (text heuristics),
                       benchmark_saturated (trivial), yaml_mapping (ships the
                       SWE-bench repo→class table; pass a path for other domains).

RoutingStrategy      — picks a Variant config for a given task class.
                       Registered: rule_based (reads policy.yaml routes),
                       cascade (tiered try-cheap-then-expensive), default
                       (fixed fallback).

Variant              — runs a single task end-to-end and returns a result.
                       Registered (7): signaled_budget, plan_then_execute,
                       two_pass_critique, stuck_signal_injection,
                       per_tool_class_hint, phase_staged, relevance_ranker.
                       (implicit_budget / unbounded_budget are budget-knob
                       framings selected via CodeCapsulesPolicy, not variants.)

Signal               — computes a derived value from session state. Used by
                       cascade triggers and result analysis.
                       Registered: cap_pressure, file_thrash, test_failure,
                       traceback, patch_attempt_failed.

QualityGate          — validates an attempt; returns True iff it passes.
                       Registered: docker_eval (SWE-bench gold tests under
                       Docker; binds the evaluator lazily), binary_tests,
                       python_ast, always_pass.

CascadeTrigger       — decides whether to escalate/re-attempt after a gate
                       verdict.
                       Registered: heuristic (cap_pressure + any signal),
                       always_escalate (forces stage 2), never_escalate,
                       verifier_gate (execution-verifier-grounded),
                       diverse_agreement (the regression-gated hybrid governor;
                       back-compat alias cross_sample_agreement).

CostModel            — maps token usage to dollar cost. Used for $/task
                       reporting and cost-aware routing.
                       Registered: anthropic_public, openai_public,
                       gemini_public (date-stamped public pricing; 10% cache
                       discount via cached_input_tokens).

## Design principles

1. Protocols, not ABCs. Structural typing means a user can write a class that
   conforms without explicit inheritance. Reduces coupling.

2. Domain-agnostic. None of these Protocols reference specific benchmarks,
   repos, or domain keywords. Domain knowledge lives in the user-supplied
   classifier and the config files they pass in (e.g. the shipped
   code_capsules/config/swe_bench_repo_classes.yaml).

3. Registry pattern. Concrete implementations register by string name; the
   policy.yaml file references implementations by name. This lets users plug
   in custom implementations without modifying framework code.

4. Calibration evidence as data, not contribution. The SWE-bench / HE / MBPP
   results we ship in policy.yaml are reference defaults. Users deploying to
   different domains derive their own defaults using the same primitives.

See:
  - paper §3 (data-driven variant selection methodology)
  - paper §6 (per-axis findings — cross-tier validated)
  - paper §7 (deployment guidance — uses these primitives)
  - paper §10 (limitations — what's outside the calibration set)
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


# ── The eight extension Protocols ────────────────────────────────────────────

@runtime_checkable
class WorkloadClassifier(Protocol):
    """Maps a task to a class label used for routing.

    Class labels are user-defined strings (e.g. "visualization_workload",
    "find_symptom", "iteration_high"). The framework does not interpret them
    semantically; it just looks them up in the routing config.

    Implementations should be deterministic given the same input.
    """

    name: str

    def classify(self, task: TaskDescriptor) -> str:
        """Return one class label for this task. Empty string means 'default'."""
        ...


@runtime_checkable
class RoutingStrategy(Protocol):
    """Picks a VariantConfig for a given class label.

    Read routes from policy.yaml (or equivalent) and return the matching
    variant config. Strategies can be rule-based (first-match), cascade
    (try cheap → expensive), or learned.
    """

    name: str

    def route(self, class_label: str, available: dict[str, VariantConfig]) -> VariantConfig:
        """Return one variant config. `available` is the registry of all
        configs keyed by name (e.g. {"plan_then_execute": ..., "two_pass_critique": ...})."""
        ...


@runtime_checkable
class Variant(Protocol):
    """Runs one task end-to-end and returns a RunResult.

    Implementations encapsulate orchestration: prompt assembly, model invocation,
    optional escalation / injection / cascade. They should respect VariantConfig
    fields they understand and ignore the rest.

    Variants are the primary deployment surface — most users won't write one,
    but the option exists for custom orchestration modes (plan-vote-execute,
    role-specialization, etc.).
    """

    name: str

    def run(self, task: TaskDescriptor, config: VariantConfig) -> RunResult:
        """Run the task; return the result."""
        ...


@runtime_checkable
class Signal(Protocol):
    """Computes a derived value from session state.

    Signals are evaluated mid-session (for cascade triggers) and post-session
    (for analysis / per-variant tuning). Signal values are typically bool or
    float; the framework treats them opaquely.
    """

    name: str

    def compute(self, state: SessionState) -> Any:
        """Return the signal value (typically bool or float)."""
        ...


@runtime_checkable
class QualityGate(Protocol):
    """Validates an Attempt; returns True iff the attempt passes the gate.

    Gates determine the "resolved" outcome of a RunResult. Defaults: binary
    (test suite passes), python_ast (patch parses + executes), composable.

    User-defined gates: linter, type checker, security scanner, code-review
    bot, etc.
    """

    name: str

    def check(self, attempt: Attempt) -> bool:
        """True if the attempt passes the gate."""
        ...


@runtime_checkable
class CascadeTrigger(Protocol):
    """Decides whether to escalate within a variant cascade.

    Used by escalating / cascade-routing variants. Inputs: the signals that
    fired on the last stage's session, plus the current tier. Output: should
    we proceed to the next tier?

    Defaults: heuristic (cap_pressure ≥ threshold AND any signal fired),
    always_escalate (forces tier 2 unconditionally; used by the two-pass critique variant),
    signal_specific (only fires on a named signal).
    """

    name: str

    def should_escalate(self, signals: dict[str, Any], current_tier: int) -> bool:
        """True if the cascade should proceed to the next tier."""
        ...


@runtime_checkable
class ModelClient(Protocol):
    """Invokes one round-trip with a coding-agent model runtime.

    Abstracts away the specific CLI / API / harness used to drive a coding
    agent. The framework ships `claude_cli` (calls the Anthropic Claude CLI
    via subprocess + parses stream-json output), plus a `generic_agent`
    skeleton that documents the contract for users plugging in alternatives
    (Aider, custom OpenAI-API agent, local llama.cpp-based agent, etc.).

    Required behavior:
      - Run the agent in `cwd` against `prompt`, capped at `max_turns`
      - Honor `session_id` for fresh sessions and `resume` for continued ones
        (the framework uses this for escalation + injection cycles)
      - Return tokens / tool calls / text / errors in an InvocationResult
      - Be subprocess-safe (called concurrently from a thread pool)

    What the framework does NOT require:
      - Specific tool name conventions (file-path / bash detection is best-effort)
      - Cost reporting (CostModel computes from tokens when client doesn't)
      - Streaming output (return-everything-at-end is fine)
    """

    name: str

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
        """Run one round-trip; return telemetry. Should not raise on
        recoverable agent errors — surface them as InvocationResult.error."""
        ...


@runtime_checkable
class CostModel(Protocol):
    """Maps token usage to a dollar cost.

    Used for $/task reporting and cost-aware routing. The default
    anthropic_public model uses public API pricing as of a stamped date;
    users with prompt caching, batch API, or enterprise pricing override.

    Cost-aware routing strategies may consult this to pick variants under a
    per-task budget cap.
    """

    name: str
    pricing_date: str                          # ISO date stamp for the pricing reference

    def cost(self, input_tokens: int, output_tokens: int, model: str,
             cached_input_tokens: int = 0) -> float:
        """Return USD cost for one API call.

        `input_tokens` is the TOTAL input (including any cached subset).
        `cached_input_tokens` is the portion that hit the vendor's cache;
        implementations bill those at the discounted rate (typically 10-25%
        of the fresh-input rate). Default 0 preserves backward compatibility
        with callers that don't know about caching.
        """
        ...


# ── Public API surface ───────────────────────────────────────────────────────

__all__ = [
    # Supporting types
    "TaskDescriptor", "VariantConfig", "SessionState", "Attempt",
    "InvocationResult", "RunResult",
    # Protocols
    "WorkloadClassifier", "RoutingStrategy", "Variant",
    "Signal", "QualityGate", "CascadeTrigger", "CostModel", "ModelClient",
]

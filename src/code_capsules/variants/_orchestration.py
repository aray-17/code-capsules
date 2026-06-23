"""
Shared orchestration primitives used by concrete Variant implementations.

Three orchestration patterns:

  run_sequential        - one model invocation; gate at the end
  run_escalating        - stage 1 + conditional stage 2 (cascade trigger)
  run_injection_loop    - stage 1 + N feedback-injection cycles

Each primitive:
  - Builds the prompt via _prompt.build_user_prompt
  - Invokes the agent runtime via ModelClient.invoke (looked up from the
    registry - default 'claude_cli'; override via config.extra['model_client'])
  - Captures patch via _runner.capture_patch (git diff in the worktree)
  - Looks up the quality gate from the registry (config.extra['quality_gate'])
  - Looks up the cost model from the registry (config.extra['cost_model'])
  - Returns a RunResult

Domain-specific glue:
  - The gate is responsible for benchmark-specific evaluation (docker_eval,
    pytest, python AST). Variants only invoke gate.check(attempt).
  - The worktree is provided by the caller via task.context['worktree_path'].
    Benchmark harnesses set up the worktree; this module does not.
  - Pre-prompt blocks (FAIL_TO_PASS tests, etc.) come via
    task.context['pre_prompt_blocks'] - a list of strings prepended verbatim.
  - The agent runtime (Claude CLI, Aider, custom OpenAI agent, etc.) is
    abstracted behind ModelClient. Variant orchestration is model-agnostic.
"""
from __future__ import annotations

import time
import uuid as _uuid
from pathlib import Path
from typing import Optional

from code_capsules.api import (
    Attempt, InvocationResult, RunResult, SessionState,
    TaskDescriptor, VariantConfig,
)
from code_capsules.variants._prompt import build_user_prompt
from code_capsules.variants._runner import capture_patch


# ── Gate / cost-model resolution ─────────────────────────────────────────────

def _resolve_gate(config: VariantConfig):
    """Look up the QualityGate from the registry, defaulting to 'binary_tests'."""
    from code_capsules.api.registry import get
    name = config.extra.get("quality_gate", "binary_tests")
    return get("quality_gate", name)


def _resolve_cost_model(config: VariantConfig):
    """Look up the CostModel from the registry, defaulting to 'anthropic_public'."""
    from code_capsules.api.registry import get
    name = config.extra.get("cost_model", "anthropic_public")
    return get("cost_model", name)


def _resolve_trigger(config: VariantConfig):
    """Look up the CascadeTrigger from the registry; default 'heuristic'."""
    from code_capsules.api.registry import get
    name = config.extra.get("cascade_trigger", "heuristic")
    return get("cascade_trigger", name)


def _resolve_client(config: VariantConfig):
    """Look up the ModelClient from the registry; default 'claude_cli'."""
    from code_capsules.api.registry import get
    name = config.extra.get("model_client", "claude_cli")
    return get("model_client", name)


def _invoke_timeout(config: VariantConfig) -> int:
    """Per-invocation wall-clock cap. Honors legacy `claude_timeout` field."""
    return config.claude_timeout or int(config.extra.get("invoke_timeout", 600))


def _check_gate(gate, patch: str, task: TaskDescriptor, session_state: Optional[SessionState] = None) -> tuple[Optional[bool], dict]:
    """Run the gate on the captured patch; return (resolved, gate_extra).

    Empty patch short-circuits to False (no point invoking docker on nothing).
    Gate exceptions are caught and surfaced as resolved=None.
    """
    if not patch.strip():
        return False, {"note": "empty patch"}
    # Pass through the task context so user-defined gates can read
    # worktree_path, task_id, custom benchmark fields, etc.
    metadata = {
        "instance": task.context.get("instance", {}),
        "tests": task.context.get("tests", []),
        "worktree_path": task.context.get("worktree_path", ""),
        "task_id": task.context.get("task_id", ""),
    }
    attempt = Attempt(patch=patch, metadata=metadata, session_state=session_state)
    try:
        resolved = gate.check(attempt)
        return bool(resolved), {}
    except Exception as exc:
        return None, {"note": f"gate error: {exc}"}


# ── Sequential ───────────────────────────────────────────────────────────────

def run_sequential(task: TaskDescriptor, config: VariantConfig) -> RunResult:
    """One model invocation; gate at the end.

    Used by: signaled_budget, plan_then_execute, per_tool_class_hint,
    phase_staged.
    """
    task_id = task.context.get("task_id", "task")
    worktree = Path(task.context["worktree_path"])
    timeout = _invoke_timeout(config)

    prompt = build_user_prompt(
        task.text,
        prompt_budget_hint=config.prompt_budget_hint,
        prompt_variant=config.prompt_variant,
        preselect_block=task.context.get("preselect_block", ""),
        pre_blocks=task.context.get("pre_prompt_blocks"),
        instructions=task.context.get("instructions"),
    )

    t0 = time.monotonic()
    gate = _resolve_gate(config)
    client = _resolve_client(config)
    inv = client.invoke(
        prompt, cwd=worktree, max_turns=config.turn_budget,
        timeout=timeout, model=config.model,
    )
    patch = capture_patch(worktree)

    if inv.error and "timeout" in (inv.error or "").lower():
        return RunResult(
            task_id=task_id, resolved=None, patch=patch,
            num_turns=0, duration_seconds=time.monotonic() - t0,
            error=inv.error,
        )

    resolved, _ = _check_gate(gate, patch, task, _inv_to_state(inv, stage=1))
    return RunResult(
        task_id=task_id,
        resolved=resolved,
        patch=patch,
        num_turns=inv.num_turns,
        input_tokens=inv.input_tokens,
        output_tokens=inv.output_tokens,
        cost_usd=_compute_cost(inv, config),
        duration_seconds=time.monotonic() - t0,
        error=inv.error,
    )


# ── Escalating ───────────────────────────────────────────────────────────────

def run_escalating(task: TaskDescriptor, config: VariantConfig) -> RunResult:
    """Two-stage with cascade-trigger gating between stages.

    Used by: two_pass_critique (always_escalate=True), relevance-ranker +
    escalation (always_escalate=False; uses heuristic cascade trigger).

    Stage 1: invoke at start_budget. Gate. If resolved → return.
    Stage 2 (if trigger fires): resume invocation with extra turns; re-gate.

    Requires the configured ModelClient to support session resume.
    """
    task_id = task.context.get("task_id", "task")
    worktree = Path(task.context["worktree_path"])
    timeout = _invoke_timeout(config)

    start_budget = config.escalating_start_budget or 10
    target_budget = config.escalating_target_budget or config.turn_budget or 20
    stage2_mode = config.escalating_stage2_mode or "generic"

    session_id = str(_uuid.uuid4())
    t0 = time.monotonic()
    gate = _resolve_gate(config)
    client = _resolve_client(config)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    prompt1 = build_user_prompt(
        task.text,
        prompt_budget_hint=start_budget,
        prompt_variant=config.prompt_variant,
        preselect_block=task.context.get("preselect_block", ""),
        pre_blocks=task.context.get("pre_prompt_blocks"),
        instructions=task.context.get("instructions"),
    )
    inv1 = client.invoke(
        prompt1, cwd=worktree, max_turns=start_budget,
        timeout=timeout, model=config.model, session_id=session_id,
    )
    patch1 = capture_patch(worktree)

    if inv1.error and "timeout" in (inv1.error or "").lower():
        return RunResult(
            task_id=task_id, resolved=None, patch=patch1,
            num_turns=0, duration_seconds=time.monotonic() - t0,
            error=inv1.error,
            stage_records=[{"stage": 1, "result": "timeout"}],
        )

    state1 = _inv_to_state(inv1, stage=1)
    resolved1, _ = _check_gate(gate, patch1, task, state1)

    # Compute signals for trigger decision
    sigs_dict = _signals_dict(inv1, resolved=resolved1,
                              cap_pressure=inv1.num_turns / max(start_budget, 1))
    fired_names = [k for k, v in sigs_dict.items()
                   if v and k not in ("resolved", "cap_pressure")]

    if config.extra.get("force_stage2"):
        # Deployment-realistic two-pass: run stage 2 UNCONDITIONALLY, never consulting
        # the held-out resolved1 (gold) for the decision. resolved1 is still computed
        # above for logging/regression analysis but is NOT read here. Mirrors
        # controller.orchestrator.decide_escalation(force_stage2=True); keeps this
        # framework runner from drifting back into the leaky `escalate = not resolved1`.
        escalate = True
    elif config.always_escalate:
        # NB: reads resolved1 (gold docker_eval) -> oracle-optimistic; NOT deployable.
        # Use force_stage2 (above) or the heuristic trigger (below) for a leakage-free gate.
        escalate = not bool(resolved1)
    else:
        trigger = _resolve_trigger(config)
        escalate = trigger.should_escalate(sigs_dict, current_tier=1)

    inv1_cost = _compute_cost(inv1, config)
    if not escalate:
        return RunResult(
            task_id=task_id,
            resolved=resolved1,
            patch=patch1,
            num_turns=inv1.num_turns,
            input_tokens=inv1.input_tokens,
            output_tokens=inv1.output_tokens,
            cost_usd=inv1_cost,
            duration_seconds=time.monotonic() - t0,
            error=inv1.error,
            stage_records=[{"stage": 1, "resolved": resolved1,
                           "turns": inv1.num_turns}],
            signals_fired=fired_names,
        )

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    extra_turns = max(1, target_budget - start_budget)
    if stage2_mode == "enriched":
        stage2_prompt = _build_stage2_prompt_enriched(
            inv1, sigs_dict, inv1.num_turns,
            extra_turns, target_budget,
        )
    else:
        stage2_prompt = _build_stage2_prompt_generic(
            inv1.num_turns, target_budget,
        )

    inv2 = client.invoke(
        stage2_prompt, cwd=worktree, max_turns=extra_turns,
        timeout=timeout, model=config.model, resume=session_id,
    )
    if inv2.error and "timeout" in (inv2.error or "").lower():
        patch = capture_patch(worktree)
        return RunResult(
            task_id=task_id, resolved=None, patch=patch,
            num_turns=inv1.num_turns,
            input_tokens=inv1.input_tokens,
            output_tokens=inv1.output_tokens,
            cost_usd=inv1_cost,
            duration_seconds=time.monotonic() - t0,
            error=inv2.error,
            stage_records=[
                {"stage": 1, "resolved": resolved1, "turns": inv1.num_turns},
                {"stage": 2, "result": "timeout"},
            ],
            signals_fired=fired_names,
        )

    patch_final = capture_patch(worktree)
    resolved_final, _ = _check_gate(gate, patch_final, task, _inv_to_state(inv2, stage=2))
    inv2_cost = _compute_cost(inv2, config)

    return RunResult(
        task_id=task_id,
        resolved=resolved_final,
        patch=patch_final,
        num_turns=inv1.num_turns + inv2.num_turns,
        input_tokens=inv1.input_tokens + inv2.input_tokens,
        output_tokens=inv1.output_tokens + inv2.output_tokens,
        cost_usd=inv1_cost + inv2_cost,
        duration_seconds=time.monotonic() - t0,
        error=inv2.error or inv1.error,
        stage_records=[
            {"stage": 1, "resolved": resolved1, "turns": inv1.num_turns},
            {"stage": 2, "resolved": resolved_final, "turns": inv2.num_turns},
        ],
        signals_fired=fired_names,
    )


# ── Injection loop ───────────────────────────────────────────────────────────

def run_injection_loop(task: TaskDescriptor, config: VariantConfig) -> RunResult:
    """Stage 1 + N feedback-injection cycles.

    Used by: stuck_signal_injection. Each cycle parses failing-test info
    from the gate output and resumes the agent runtime with a targeted
    injection prompt.

    The feedback parser is plugged in via config.extra['feedback_parser']
    (callable: gate_stdout → dict). Default is the pytest/Django heuristic
    bundled with this module.

    Requires the configured ModelClient to support session resume.
    """
    task_id = task.context.get("task_id", "task")
    worktree = Path(task.context["worktree_path"])
    timeout = _invoke_timeout(config)

    start_budget = config.escalating_start_budget or config.turn_budget or 10
    per_cycle = config.injection_per_cycle_turns or 5
    max_cycles = config.injection_cycles or 3

    session_id = str(_uuid.uuid4())
    t0 = time.monotonic()
    gate = _resolve_gate(config)
    client = _resolve_client(config)
    parser = config.extra.get("feedback_parser") or _default_feedback_parser

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    prompt1 = build_user_prompt(
        task.text,
        prompt_budget_hint=start_budget,
        prompt_variant=config.prompt_variant,
        preselect_block=task.context.get("preselect_block", ""),
        pre_blocks=task.context.get("pre_prompt_blocks"),
        instructions=task.context.get("instructions"),
    )
    inv1 = client.invoke(
        prompt1, cwd=worktree, max_turns=start_budget,
        timeout=timeout, model=config.model, session_id=session_id,
    )
    patch = capture_patch(worktree)
    if inv1.error and "timeout" in (inv1.error or "").lower():
        return RunResult(
            task_id=task_id, resolved=None, patch=patch,
            num_turns=0, duration_seconds=time.monotonic() - t0,
            error=inv1.error,
        )

    resolved, gate_extra = _check_gate(gate, patch, task, _inv_to_state(inv1, stage=1))
    cum_turns = inv1.num_turns
    cum_in = inv1.input_tokens
    cum_out = inv1.output_tokens
    cum_cost = _compute_cost(inv1, config)
    cycles_run = 0
    stage_records = [{"stage": 1, "resolved": resolved, "turns": inv1.num_turns}]

    # ── Injection cycles ─────────────────────────────────────────────────────
    for cycle_idx in range(max_cycles):
        if resolved:
            break
        if not patch.strip():
            break
        feedback = parser(gate_extra.get("stdout", "") or "")
        inj_prompt = _build_injection_prompt(
            feedback, patch, cycle_idx, max_cycles, per_cycle,
        )
        inv_n = client.invoke(
            inj_prompt, cwd=worktree, max_turns=per_cycle,
            timeout=timeout, model=config.model, resume=session_id,
        )
        if inv_n.error and "timeout" in (inv_n.error or "").lower():
            stage_records.append({"stage": 2 + cycle_idx, "result": "timeout"})
            break
        cum_turns += inv_n.num_turns
        cum_in += inv_n.input_tokens
        cum_out += inv_n.output_tokens
        cum_cost += _compute_cost(inv_n, config)
        cycles_run += 1
        patch = capture_patch(worktree)
        resolved, gate_extra = _check_gate(gate, patch, task, _inv_to_state(inv_n, stage=2 + cycle_idx))
        stage_records.append({"stage": 2 + cycle_idx, "resolved": resolved,
                              "turns": inv_n.num_turns})

    return RunResult(
        task_id=task_id,
        resolved=resolved,
        patch=patch,
        num_turns=cum_turns,
        input_tokens=cum_in,
        output_tokens=cum_out,
        cost_usd=cum_cost,
        duration_seconds=time.monotonic() - t0,
        error=inv1.error,
        stage_records=stage_records,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _inv_to_state(inv: InvocationResult, *, stage: int) -> SessionState:
    """Map an InvocationResult into the SessionState that Signal impls consume."""
    return SessionState(
        tool_calls=inv.tool_calls,
        num_turns=inv.num_turns,
        effective_input_tokens=inv.input_tokens,
        total_output_tokens=inv.output_tokens,
        stage=stage,
    )


def _compute_cost(inv: InvocationResult, config: VariantConfig) -> float:
    """Return cost for one invocation.

    Prefer the client-reported cost when it set one (Anthropic CLI surfaces
    this). Otherwise compute via the active CostModel from tokens + model name,
    threading cached_input_tokens through so vendors with automatic prompt
    caching (OpenAI Responses, Gemini context cache) bill cached tokens at
    the discounted rate rather than full-input rate.
    """
    if inv.cost_usd > 0:
        return inv.cost_usd
    cost_model = _resolve_cost_model(config)
    model = inv.model or config.model or ""
    try:
        return cost_model.cost(
            inv.input_tokens, inv.output_tokens, model,
            cached_input_tokens=getattr(inv, "cached_input_tokens", 0),
        )
    except TypeError:
        # Older CostModel implementations don't accept cached_input_tokens;
        # fall back to the legacy 3-arg form (overstates cost slightly for
        # cache-aware vendors but preserves backward compatibility).
        return cost_model.cost(inv.input_tokens, inv.output_tokens, model)
    except Exception:
        return 0.0


def _signals_dict(inv: InvocationResult, *, resolved: Optional[bool],
                  cap_pressure: float) -> dict:
    """Compute the signal dict consumed by cascade triggers.

    Wraps controller.runtime_quality_signal.compute_signals for the boolean
    flags + adds cap_pressure + resolved.
    """
    from code_capsules.runtime.quality_signal import compute_signals
    sigs = compute_signals(inv.tool_calls) if inv.tool_calls else None
    out: dict = {
        "resolved": bool(resolved) if resolved is not None else False,
        "cap_pressure": cap_pressure,
    }
    if sigs is not None:
        out.update({
            "file_thrash": sigs.file_thrash,
            "test_failure": sigs.test_failure,
            "traceback": sigs.traceback,
            "patch_attempt_failed": sigs.patch_attempt_failed,
            "no_progress": getattr(sigs, "no_progress", False),
        })
    return out


def _build_stage2_prompt_generic(num_turns_used: int, total_budget: int) -> str:
    return (
        f"Your turn budget has been extended. You now have a total of "
        f"{total_budget} turns; you've used {num_turns_used} so far. "
        "Continue diagnosing the issue and complete the fix. Tests must pass."
    )


def _build_stage2_prompt_enriched(inv1: InvocationResult, signals: dict,
                                  num_turns_used: int,
                                  extra_turns: int, total_budget: int) -> str:
    """Two-pass critique enriched stage-2 prompt.

    Summarizes stage-1 tool activity + maps fired signals to a directive.
    """
    from collections import Counter
    reads = Counter(
        tc.file_path for tc in inv1.tool_calls
        if getattr(tc, "is_read", False) and getattr(tc, "file_path", None)
    )
    writes = Counter(
        tc.file_path for tc in inv1.tool_calls
        if getattr(tc, "is_write", False) and getattr(tc, "file_path", None)
    )
    bash_calls = [tc for tc in inv1.tool_calls if getattr(tc, "is_bash", False)]

    parts = []
    if reads:
        parts.append("**Files read:** " + ", ".join(
            f"{p} (×{c})" if c > 1 else p for p, c in reads.most_common(8)
        ))
    if writes:
        parts.append("**Files edited:** " + ", ".join(
            f"{p} (×{c})" if c > 1 else p for p, c in writes.most_common(8)
        ))
    if bash_calls:
        tail = (getattr(bash_calls[-1], "output_text", "") or "")[-300:].strip()
        parts.append(f"**Commands run:** {len(bash_calls)} total. "
                     f"Most recent output tail:\n```\n{tail}\n```")
    if not parts:
        parts.append("(no tool activity captured in stage 1)")
    summary = "\n\n".join(parts)

    if signals.get("patch_attempt_failed"):
        directive = ("Your most recent fix attempt was followed by a failing "
                     "test run — the patch you tried did not resolve the issue. "
                     "Re-examine whether you're targeting the right file or behavior.")
    elif signals.get("traceback"):
        directive = ("Your most recent attempt produced a Python traceback — your "
                     "fix may have broken imports or runtime behavior. Roll back "
                     "mentally and try a smaller, more surgical change.")
    elif signals.get("test_failure"):
        directive = ("Tests are still failing after your attempts. Read the failure "
                     "message carefully — what does it tell you about the actual bug?")
    elif signals.get("file_thrash"):
        directive = ("You re-read the same file multiple times without committing "
                     "to a fix. This suggests you may be focused on the wrong area, "
                     "or missing context elsewhere. Broaden your search.")
    else:
        directive = "Re-examine your approach with fresh eyes."

    return (
        f"Your turn budget has been extended. You used {num_turns_used} turns "
        f"in the initial attempt; you have {extra_turns} more turns now "
        f"(total budget: {total_budget}).\n\n"
        f"## What you did in the initial attempt\n\n{summary}\n\n"
        f"## Why I'm extending your budget\n\n{directive}\n\n"
        f"## What to do now\n\n"
        "Step back from your current approach. Consider files or aspects you "
        "haven't explored yet. Tests must pass. Do not repeat the same "
        "investigation pattern — if you've already inspected a file, trust "
        "your prior reading unless something new has come to light."
    )


# ── Injection-loop helpers (default feedback parser) ─────────────────────────

import re as _re
_PYTEST_FAIL_RE = _re.compile(r"FAILED\s+([^\s:]+::[^\s:]+(?:::[^\s:]+)?)")
_DJANGO_FAIL_RE = _re.compile(r"FAIL:\s+([\w.]+\s+\(\S+\))")
_TRACEBACK_FILE_RE = _re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)')


def _default_feedback_parser(gate_stdout: str) -> dict:
    """Default: extract failing-test name + most-recent traceback frame.

    Recognizes pytest ("FAILED foo.py::bar") and Django ("FAIL: x (mod.Cls)")
    formats. Domain-specific gates can override via config.extra['feedback_parser'].
    """
    info = {"failing_test": "", "traceback_file": "", "traceback_line": ""}
    if not gate_stdout:
        return info
    m = _PYTEST_FAIL_RE.search(gate_stdout)
    if m:
        info["failing_test"] = m.group(1)
    else:
        m = _DJANGO_FAIL_RE.search(gate_stdout)
        if m:
            info["failing_test"] = m.group(1)
    matches = list(_TRACEBACK_FILE_RE.finditer(gate_stdout))
    if matches:
        last = matches[-1]
        info["traceback_file"] = last.group(1)
        info["traceback_line"] = last.group(2)
    return info


def _build_injection_prompt(feedback: dict, last_patch: str,
                            cycle_idx: int, total_cycles: int,
                            extra_turns: int) -> str:
    test_line = ""
    if feedback.get("failing_test"):
        test_line = f"Failing test: `{feedback['failing_test']}`\n"
    if feedback.get("traceback_file"):
        tb = feedback["traceback_file"]
        ln = feedback.get("traceback_line", "?")
        test_line += f"Most-recent traceback frame: `{tb}:{ln}`\n"
    patch_excerpt = last_patch[:1500] if last_patch else "(no patch produced last cycle)"
    return (
        f"## Iteration {cycle_idx + 1} of up to {total_cycles}\n\n"
        f"Your previous patch did not pass the failing tests.\n\n"
        f"{test_line}\n"
        f"Your last patch (excerpt):\n```diff\n{patch_excerpt}\n```\n\n"
        f"You have {extra_turns} more turns. Refine the patch — focus narrowly on "
        f"the failing test. Re-read the implicated file at the traceback line, "
        f"identify what your last patch missed, and produce a corrected version. "
        f"Avoid restarting exploration; iterate on the patch you have.\n"
    )


__all__ = ["run_sequential", "run_escalating", "run_injection_loop"]

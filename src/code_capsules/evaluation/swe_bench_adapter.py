"""
SWE-bench → Code-Capsules framework adapter.

Bridges SWE-bench Lite's instance-dict shape into the framework's domain-agnostic
TaskDescriptor + DockerEvalGate, and back into the legacy DockerResult-shape
JSONL row that analyzers expect.

This module is intentionally narrow:
  - SWE-bench knowledge stays here, not in controller.runtime or
    controller.variants
  - The framework's extension surface (Protocols, registry, Runner) doesn't
    import this module
  - The new harness imports this; the legacy harness does not

Use:
    from code_capsules.evaluation.swe_bench_adapter import (
        make_task_descriptor,
        make_docker_eval_gate,
        run_result_to_legacy_jsonl,
    )

For each instance (one variant == a one-config policy on the single public runner):
    task = make_task_descriptor(instance, worktree_path=...)
    gate = make_docker_eval_gate()  # defaults to the runtime scorer
                                    # code_capsules.evaluation.docker_eval.docker_eval
    sampler, grade_fn, captured = make_variant_sampler(task, cfg, gate=gate)
    CodeCapsulesRunner(RunnerPolicy(configs=(cfg.name,), tiers=("default",))).run(sampler, grade_fn)
    row = run_result_to_legacy_jsonl(captured["result"], instance, mode=..., prompt_budget_hint=...)
    out_file.write(json.dumps(row) + "\\n")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from code_capsules.api import TaskDescriptor


# ── Task adapter ─────────────────────────────────────────────────────────────

def make_task_descriptor(instance: dict, worktree_path: Path) -> TaskDescriptor:
    """Build a TaskDescriptor from a SWE-bench Lite instance dict.

    The task text is the problem_statement. Context carries:
      - instance:           full instance dict (gate reads metadata['instance'])
      - worktree_path:      directory where claude runs (already set up by caller)
      - task_id:            instance_id for logging
      - pre_prompt_blocks:  list of pre-prompt content (FAIL_TO_PASS test list)
    """
    fail_tests: list[str] = instance.get("FAIL_TO_PASS", [])
    if isinstance(fail_tests, str):
        fail_tests = json.loads(fail_tests)

    tests_block = ""
    if fail_tests:
        ids = "\n".join(f"  - {t}" for t in fail_tests)
        tests_block = (
            f"## Tests that must pass after your fix\n\n{ids}\n\n"
            "These tests currently fail. Your fix must make them pass "
            "without breaking existing passing tests.\n\n"
        )

    return TaskDescriptor(
        text=instance["problem_statement"],
        context={
            "task_id": instance["instance_id"],
            "instance": instance,
            "worktree_path": str(worktree_path),
            "pre_prompt_blocks": [tests_block] if tests_block else [],
        },
    )


# ── Gate factory ─────────────────────────────────────────────────────────────

def make_docker_eval_gate(docker_eval=None, docker_timeout: int = 180):
    """Build a DockerEvalGate from an injected SWE-bench evaluator.

    docker_eval(patch, instance, timeout=...) -> dict applies the patch inside the
    SWE-bench container and reports whether every FAIL_TO_PASS test passes. A
    benchmark driver may inject its own implementation, keeping this adapter
    standalone. If omitted, the canonical scorer shipped in the runtime
    (code_capsules.evaluation.docker_eval.docker_eval) is used.
    """
    from code_capsules.evaluation.quality_gates_adaptors import DockerEvalGate
    if docker_eval is None:
        from code_capsules.evaluation.docker_eval import docker_eval as docker_eval  # noqa: PLW0127

    def _fn(patch: str, instance: dict, timeout: int) -> dict:
        return docker_eval(patch, instance, timeout=timeout)

    return DockerEvalGate(docker_eval_fn=_fn, timeout=docker_timeout)


# ── Result adapter ───────────────────────────────────────────────────────────

def run_result_to_legacy_jsonl(
    result,
    instance: dict,
    *,
    mode: str,
    prompt_budget_hint: Optional[int] = None,
    routing_decision: Optional[str] = None,
    turn_budget: Optional[int] = None,
) -> dict[str, Any]:
    """Map a RunResult into the legacy DockerResult-shape dict.

    Analyzers (tools/analyze_*.py) and the paper-bound JSONLs depend on the
    21-field schema. This function preserves it field-for-field so the new
    harness produces drop-in-compatible output.

    Fields not derivable from RunResult (eval_note, eval_stdout, has_patch
    semantics) are filled with conservative defaults so analyzers don't crash.
    """
    # Pull stage records (set by Runner.run as a routing meta entry, plus the
    # variant's own per-stage records)
    stage_records = [r for r in result.stage_records if not r.get("_meta")]
    routing_meta = next(
        (r for r in result.stage_records if r.get("_meta") == "routing"), {}
    )

    escalated: Optional[bool] = None
    stage1_turns: Optional[int] = None
    stage2_turns: Optional[int] = None
    escalation_reason: Optional[str] = None
    if stage_records:
        # Multi-stage variant (escalating or injection_loop)
        s1 = stage_records[0] if len(stage_records) >= 1 else {}
        s2 = stage_records[-1] if len(stage_records) >= 2 else None
        stage1_turns = s1.get("turns", 0)
        if s2 is not None and s2 is not s1:
            escalated = True
            stage2_turns = s2.get("turns", 0) or 0
            escalation_reason = s2.get("result") or "escalated"
        else:
            escalated = False
            stage2_turns = 0
            escalation_reason = "no escalation"

    eval_note = ""
    if not result.patch.strip():
        eval_note = "empty patch"
    elif result.error:
        eval_note = result.error[:200]

    return {
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "mode": mode,
        "routing_decision": routing_decision or routing_meta.get("variant", mode),
        "turn_budget": turn_budget if turn_budget is not None else result.num_turns,
        "actual_turns": result.num_turns,
        "effective_input_tokens": result.input_tokens,
        "total_output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "has_patch": bool(result.patch.strip()),
        # The captured git diff, persisted so a future scorer change can be
        # verified by a MODEL-FREE re-score of the stored patch
        # (benchmarks/swebench/rescore_from_patch.py) instead of an online
        # re-run. A row that archives its patch can be re-graded offline for $0;
        # every runtime-produced eval row carries its patch for this reason.
        "model_patch": result.patch or None,
        "resolved": result.resolved,
        "eval_note": eval_note,
        "eval_stdout": "",  # Runner doesn't surface raw gate stdout; analyzers
                            # use the stream archive for deep dives instead
        "duration_s": result.duration_seconds,
        "error": result.error,
        "prompt_budget_hint": prompt_budget_hint,
        "escalated": escalated,
        "stage1_turns": stage1_turns,
        "stage2_turns": stage2_turns,
        "escalation_reason": escalation_reason,
        "signals_fired": list(result.signals_fired),
    }


__all__ = [
    "make_task_descriptor",
    "make_docker_eval_gate",
    "run_result_to_legacy_jsonl",
]

"""
Plan-then-execute variant (Phase 10 Iter-C V4).

Sequential run with an explicit "## Approach" block instructing the model to
spend turn 1 on a written plan (no tools) before executing. Compresses the 29%
text-only-turn share measured in B-ext, freeing turns for productive tool use.

Cross-tier results (Sonnet n=150, Haiku n=150, Opus n=30):
  - Sonnet 44%/$0.287 — Pareto-frontier (promote)
  - Haiku 21% — dominates implicit-20 (same Pareto shape)
  - Opus 93%/$0.484 — dominates floor at 9% cheaper

Strongest cross-tier finding: dominates progressively more expensive baselines
as model capability rises.
"""
from __future__ import annotations

from dataclasses import replace

from code_capsules.api import RunResult, TaskDescriptor, VariantConfig
from code_capsules.variants._orchestration import run_sequential


class PlanThenExecuteVariant:
    name = "plan_then_execute"

    def run(self, task: TaskDescriptor, config: VariantConfig) -> RunResult:
        # Force prompt_variant=plan_first; preserve other config.
        adjusted = replace(
            config,
            prompt_variant="plan_first",
            prompt_budget_hint=config.prompt_budget_hint or config.turn_budget,
        )
        return run_sequential(task, adjusted)


__all__ = ["PlanThenExecuteVariant"]

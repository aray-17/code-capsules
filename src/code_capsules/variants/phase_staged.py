"""
Phase-staged variant.

Sequential run with an explicit 3-phase budget split (explore/first-patch/iterate)
matching the floor's implicit pattern. Hard structural constraint: the prompt
declares exact turn ranges per phase.

Sonnet n=150: 41%/$0.291, dropped (over-constrained at 17.8 of 35 turns;
underperforms plan-then-execute, which uses softer framing).
"""
from __future__ import annotations

from dataclasses import replace

from code_capsules.api import RunResult, TaskDescriptor, VariantConfig
from code_capsules.variants._orchestration import run_sequential


class PhaseStagedVariant:
    name = "phase_staged"

    def run(self, task: TaskDescriptor, config: VariantConfig) -> RunResult:
        adjusted = replace(
            config,
            prompt_variant="phase_staged",
            prompt_budget_hint=config.prompt_budget_hint or config.turn_budget,
        )
        return run_sequential(task, adjusted)


__all__ = ["PhaseStagedVariant"]

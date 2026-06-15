"""
Signaled N-turn budget variant — single-shot sequential run.

The Phase 7B-Supp baseline that the rest of the Pareto frontier is measured
against. Tells the model its budget up front (AC M-1 pattern) and runs to
completion. No escalation, no injection.

Calibrated cross-tier as the Pareto-frontier reference at budgets {5,10,20,40}.
"""
from __future__ import annotations

from code_capsules.api import RunResult, TaskDescriptor, VariantConfig
from code_capsules.variants._orchestration import run_sequential


class SignaledBudgetVariant:
    """Single-shot run with explicit turn-budget hint in the prompt."""

    name = "signaled_budget"

    def run(self, task: TaskDescriptor, config: VariantConfig) -> RunResult:
        # If the caller didn't set a prompt_budget_hint, default to the turn budget.
        if config.prompt_budget_hint is None:
            config = _with_budget_hint(config)
        return run_sequential(task, config)


def _with_budget_hint(config: VariantConfig) -> VariantConfig:
    from dataclasses import replace
    return replace(config, prompt_budget_hint=config.turn_budget)


__all__ = ["SignaledBudgetVariant"]

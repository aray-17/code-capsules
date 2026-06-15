"""
Per-tool-class hint variant (Phase 10 Iter-C V5).

Sequential run with an "## Approach" block suggesting a tool-class budget
(~5 reads / ~3 patches / ~5 test-iterate cycles) matching the B-ext observed
distribution. Soft framing — no hard structural constraint.

Sonnet n=150: 42%/$0.283 — on Pareto frontier (no promote vs V1/V4).
"""
from __future__ import annotations

from dataclasses import replace

from code_capsules.api import RunResult, TaskDescriptor, VariantConfig
from code_capsules.variants._orchestration import run_sequential


class PerToolClassHintVariant:
    name = "per_tool_class_hint"

    def run(self, task: TaskDescriptor, config: VariantConfig) -> RunResult:
        adjusted = replace(
            config,
            prompt_variant="tool_alloc",
            prompt_budget_hint=config.prompt_budget_hint or config.turn_budget,
        )
        return run_sequential(task, adjusted)


__all__ = ["PerToolClassHintVariant"]

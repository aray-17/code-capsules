"""
Two-pass critique variant (Phase 10 Iter-C V1).

Escalating mode with `always_escalate=True`: stage 1 always followed by stage 2
(unless stage 1 already resolved). Stage 2 receives an enriched prompt
summarizing stage-1 activity + a signal-specific directive.

The "two-pass critique" framing — first attempt, then forced critique with
context — is the highest-RESOLVE iteration-axis treatment in cross-tier evaluation.
It is a QUALITY knob, not a cost knee: its earlier cost numbers (e.g. "$0.308")
were oracle-discounted by an in-loop GOLD stage-2 skip and are being re-priced
leakage-free via force_stage2 (honest cost is ~floor-class, dominated by
plan-then-execute on the cost axis). Resolve rates below are unaffected by the leak.

  NB: `always_escalate=True` means "escalate unless stage 1 already resolved",
  which reads the GOLD gate -> NOT deployable as-is. A deployable two-pass uses
  force_stage2 (always run both) or the execution-verifier cascade trigger.

Cross-tier RESOLVE rates (leak-free; cost re-priced separately):
  - Sonnet n=150: ~47% (highest-resolve variant)
  - Haiku n=150: ~32%
  - Opus n=30: ~90% (dominated by plan-then-execute, which is enough on Opus)

Defaults: start_budget=10 (Pareto knee), target_budget=20, stage2_mode=enriched.
Use stage2_mode='generic' for the V1-baseline behavior; 'enriched' is the
default for the variant that ships in policy.yaml.
"""
from __future__ import annotations

from dataclasses import replace

from code_capsules.api import RunResult, TaskDescriptor, VariantConfig
from code_capsules.variants._orchestration import run_escalating


class TwoPassCritiqueVariant:
    name = "two_pass_critique"

    def run(self, task: TaskDescriptor, config: VariantConfig) -> RunResult:
        adjusted = replace(
            config,
            always_escalate=True,
            escalating_start_budget=config.escalating_start_budget or 10,
            escalating_target_budget=config.escalating_target_budget
                                      or config.turn_budget or 20,
            escalating_stage2_mode=config.escalating_stage2_mode or "enriched",
        )
        return run_escalating(task, adjusted)


__all__ = ["TwoPassCritiqueVariant"]

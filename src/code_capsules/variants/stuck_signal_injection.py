"""
Stuck-signal injection variant (Phase 10 Iter-C V2).

Injection loop: stage 1 at start_budget, then up to N cycles of (parse failing
test info from gate → resume claude with targeted injection prompt → re-gate).
The injection prompt includes the failing-test name, most-recent traceback frame,
and an excerpt of the last patch.

The feedback parser is pluggable via config.extra['feedback_parser'] — the
default handles pytest + Django output. For non-Python gates, supply a
domain-specific parser.

Cross-tier results:
  - Sonnet n=150: 41%/$0.325 — dropped at aggregate (only 18% injection-fire rate)
    BUT: V2 signal on scientific-class workloads REPLICATES on held-out
    (astropy +33pp n=6 → scikit-learn +9pp n=23 on first-150)
  - Haiku n=150: 5% — COLLAPSES (Haiku at b=10 can't produce reasonable stage-1
    patches for the injection loop to refine)
  - Opus n=30: 73%/$0.404 — dominated by P8-C

Recommendation: deploy on per-class basis for Sonnet scientific workloads;
not generally on Haiku.

Defaults: start_budget=10, per_cycle_turns=5, max_cycles=3.
"""
from __future__ import annotations

from dataclasses import replace

from code_capsules.api import RunResult, TaskDescriptor, VariantConfig
from code_capsules.variants._orchestration import run_injection_loop


class StuckSignalInjectionVariant:
    name = "stuck_signal_injection"

    def run(self, task: TaskDescriptor, config: VariantConfig) -> RunResult:
        adjusted = replace(
            config,
            escalating_start_budget=config.escalating_start_budget or 10,
            injection_cycles=config.injection_cycles or 3,
            injection_per_cycle_turns=config.injection_per_cycle_turns or 5,
        )
        return run_injection_loop(task, adjusted)


__all__ = ["StuckSignalInjectionVariant"]

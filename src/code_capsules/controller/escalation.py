"""
Phase 7C: escalation controller for adaptive turn-budget management.

After a Claude Code session completes, this module decides whether to grant
additional turns (escalate) based on:
  - Did the session resolve?
  - Did it actually press up against the budget?
  - Did mid-session quality signals fire (`controller.runtime_quality_signal`)?

Coding-side analog of Agentic-Capsules' C-14 (adaptive controller) and
E-1 (escalation ladder triggered by rolling-mean quality breach). Where AC
escalates compound → two_phase → sequential, Code-Capsules escalates within
a single session's turn budget: signaled-10 → signaled-20.

Anchored to Phase 7B/7B-Supp data:
  - Phase 7B: budget=20 is the implicit-regime knee (16.7% pass, $0.69/task).
  - Phase 7B-Supp: signaled-10 = 16.7% pass at $0.39/task — new Pareto knee.
  - Above 20 turns: pass-rate plateaus at 16.7%. No quality benefit.
So escalation_target_budget caps at 20.

Design choices:
  - Stateful decision is a pure function of session outcome (no I/O).
  - Returns a frozen `EscalationDecision` dataclass — caller dispatches.
  - Three "don't escalate" branches: resolved, low-effort (didn't push cap),
    or clean-give-up (cap-hit but no progress signals). Only the last branch
    is non-obvious — it says "extra turns won't help if the model has been
    spinning."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from code_capsules.runtime.quality_signal import QualitySignals


@dataclass(frozen=True)
class EscalationDecision:
    escalate: bool
    reason: str
    # When escalating, the new --max-turns to grant and the budget value
    # to use in the prompt-budget-hint. Both None when not escalating.
    next_budget: Optional[int] = None
    next_prompt_budget_hint: Optional[int] = None


def should_escalate(
    *,
    resolved: bool,
    actual_turns: int,
    current_budget: int,
    signals: QualitySignals,
    target_budget: int = 20,
    cap_pressure_threshold: float = 0.7,
) -> EscalationDecision:
    """
    Decide whether to escalate after a session completes.

    Inputs:
      resolved              — did the FAIL_TO_PASS tests pass post-session?
      actual_turns          — turns the model actually used
      current_budget        — the --max-turns cap on the just-completed session
      signals               — QualitySignals computed from the session's tool calls
      target_budget         — budget to escalate to (default 20 per Phase 7B knee)
      cap_pressure_threshold — model must have used ≥ this fraction of the budget
                                to be eligible for escalation (default 0.7).
                                If the model self-terminated well below the cap,
                                it didn't need more turns — extra turns won't help.

    Decision tree:
      1. If resolved          → don't escalate (no point)
      2. If current ≥ target  → can't escalate (already at the ceiling)
      3. If turns/budget < cap_pressure_threshold → don't escalate
         (the model self-terminated below the cap; if it stopped early without
         signals, it gave up, not "ran out of time")
      4. If signals.any       → escalate (active stalling — more turns may help)
      5. Else                 → don't escalate (cap-hit but no progress signals
         is a "clean give-up" — model was spinning, more time unlikely to fix)
    """
    # 1. Already done
    if resolved:
        return EscalationDecision(
            escalate=False,
            reason=f"already resolved at budget={current_budget}",
        )

    # 2. At the ceiling
    if current_budget >= target_budget:
        return EscalationDecision(
            escalate=False,
            reason=f"already at target budget ({current_budget} >= {target_budget})",
        )

    # 3. Didn't press the cap → escalation won't help
    cap_pressure = (actual_turns / current_budget) if current_budget else 0.0
    if cap_pressure < cap_pressure_threshold:
        return EscalationDecision(
            escalate=False,
            reason=(
                f"low cap pressure ({actual_turns}/{current_budget} = "
                f"{cap_pressure:.2f} < {cap_pressure_threshold:.2f}); "
                "model self-terminated early, extra turns unlikely to help"
            ),
        )

    # 4. Quality signals fired → escalate
    if signals.any:
        fired = ",".join(signals.names_fired)
        return EscalationDecision(
            escalate=True,
            reason=(
                f"cap pressure {cap_pressure:.2f} + signals fired [{fired}] → "
                f"escalate {current_budget} → {target_budget}"
            ),
            next_budget=target_budget,
            next_prompt_budget_hint=target_budget,
        )

    # 5. Cap-hit but clean — model gave up productively
    return EscalationDecision(
        escalate=False,
        reason=(
            f"cap pressure {cap_pressure:.2f} but no quality signals fired; "
            "clean give-up — extra turns unlikely to help"
        ),
    )

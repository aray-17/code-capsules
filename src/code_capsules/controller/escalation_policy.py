"""Escalation ladder policy: per-stage configs + per-stage escalation triggers.

An EscalationStage names ONE rung of the model ladder: which tier to run, which
diverse configs to sample THERE (per-stage — the validated destination shapes
run a 2-config ensemble at the cheap tier and a single config at the expensive
one), and the TRIGGER that must fire for the controller to escalate INTO this
stage after the previous stage's agreement-failure. The first stage's trigger
is never consulted (it always runs).

Validated destination shapes the triggers encode (EXP-4 n=300 deep-eval,
2026-06-10, evals/pareto_upgrade_implementation_plan.md §1):

  regok_true — GOV-V2 escalate-regok=1-to-Opus = 84/150 @ $1.60/res. Opus
      recovered 20/21 of its recoveries inside the regok=1 band vs 3/17 in
      regok=0 — the selected candidate keeping the existing suite green is the
      gate-positive escalation signal. Caveat (G4): regok=0 is NOT doom
      off-django (P(gold|regok=0)=0.37 held-out) — a non-fired regok_true
      trigger means "no positive evidence", never "confirmed doomed".
  has_patch — cascade signaled10 -> opus_floor on NO-PATCH = 86 @ $0.805
      (django-concentrated caveat). Fires when a candidate produced NO patch
      ("no patch yet" = needs more compute, 43% winnable per EXP-1), mirroring
      the validated signaled10->opus cascade. NB: the trigger is NAMED for the
      signal it reads; it fires on the no-patch case.
  no_signal — fires only when the round's failure_kind is "no_signal" (every
      non-RESOLVED grade UNKNOWN/NOPATCH: no usable execution reading). Note
      the ladder ALREADY escalates no_signal failures by default regardless of
      trigger (see orchestrator.run_controller); use this trigger to build a
      stage that handles ONLY the no-reading band.
  always — unconditional escalation on agreement-failure (the legacy
      escalate_on_agreement=True behavior).
  voi — value-of-information break-even (value_of_information.voi_escalate):
      escalate iff p_resolve * V > C with the stage-configured cost C
      (voi_cost) and value V (voi_value). p_resolve comes from
      signals["p_resolve"] when a calibrated belief is wired, else from an
      optional pluggable PSource (plan §4 item 5 -- e.g. the resolvability
      adapter, opt-in via policy `p_source`, ships disabled), else from the
      EXP-1 per-grade belief table keyed on the selected candidate's grade.
  never — internal/legacy value (the escalate_on_agreement=False conversion);
      the trigger never fires, so agreement-failure declines at this rung.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from code_capsules.controller.verifier import NOPATCH

TRIGGERS = ("regok_true", "has_patch", "no_signal", "always", "voi", "never")


@dataclass(frozen=True)
class EscalationStage:
    """One rung of the escalation ladder.

    tier      : the model-tier id the stage runs at (e.g. "sonnet", "opus")
    configs   : the diverse configs to sample AT THIS STAGE (per-stage — e.g.
                ("floor", "siginject") at the cheap tier, ("floor",) at Opus)
    trigger   : the condition (one of TRIGGERS) under which the PREVIOUS
                stage's agreement-failure escalates into this stage; ignored
                on the first stage
    voi_cost  : C, the cost of one escalation (required iff trigger == "voi")
    voi_value : V, the worth of one resolve  (required iff trigger == "voi")
    """

    tier: str
    configs: tuple
    trigger: str = "always"
    voi_cost: Optional[float] = None
    voi_value: Optional[float] = None

    def __post_init__(self):
        if self.trigger not in TRIGGERS:
            raise ValueError(
                f"trigger must be one of {TRIGGERS}, got {self.trigger!r}")
        if self.trigger == "voi" and (self.voi_cost is None or self.voi_value is None):
            raise ValueError(
                "trigger='voi' requires stage-configured voi_cost (C) and voi_value (V)")
        if not self.configs:
            raise ValueError("an EscalationStage needs at least one config")


def _no_patch(signals: dict, selection: Any) -> bool:
    """A candidate produced no patch. Prefers the round's n_nopatch count
    (agreement_reading threads it); falls back to the selection's grade when
    the signals were hand-built (selected NOPATCH = every candidate empty,
    since NOPATCH is the lowest selector rank)."""
    n = signals.get("n_nopatch")
    if n is not None:
        return n > 0
    return selection is None or getattr(selection, "grade", None) == NOPATCH


def stage_trigger_fires(stage: EscalationStage, signals: dict, selection: Any = None,
                        p_source: Any = None) -> bool:
    """Evaluate `stage.trigger` against the previous round's signals/selection:
    True = the agreement-failure escalates INTO this stage; False = it does not
    (the ladder then lets the governor have its terminal say — rescue or
    decline with a recorded reason; see orchestrator.run_controller).

    p_source: optional value_of_information.PSource consulted by the `voi`
    trigger when signals carry no precomputed `p_resolve` (evidence = the
    round's signals dict + the selected grade); None -> the EXP-1 per-grade
    belief table, exactly the prior behavior."""
    t = stage.trigger
    if t == "always":
        return True
    if t == "never":
        return False
    if t == "regok_true":
        # GOV-V2 gate-positive escalation: 84/150 @ $1.60/res; Opus recoveries
        # 20/21 in regok=1 vs 3/17 in regok=0 (see module docstring + G4 caveat).
        return signals.get("selected_regression_ok") is True
    if t == "has_patch":
        # The validated signaled10->opus_floor cascade: escalate the NO-patch
        # case (86 @ $0.805, django-concentrated caveat).
        return _no_patch(signals, selection)
    if t == "no_signal":
        return signals.get("failure_kind") == "no_signal"
    if t == "voi":
        from code_capsules.controller.value_of_information import (
            voi_escalate, GRADE_P_RESOLVE_EXP1,
        )
        p = signals.get("p_resolve")
        if p is None and p_source is not None:
            evidence = dict(signals)
            evidence.setdefault("grade", getattr(selection, "grade", None))
            p = p_source.p_resolve(evidence)
        if p is None:
            grade = getattr(selection, "grade", None)
            p = GRADE_P_RESOLVE_EXP1.get(grade, 0.0)
        return voi_escalate(p, stage.voi_value, stage.voi_cost).escalate
    raise ValueError(f"unknown trigger {t!r}")  # unreachable: __post_init__ validates


__all__ = ["TRIGGERS", "EscalationStage", "stage_trigger_fires"]

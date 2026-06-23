"""
The controller's core formula: value-of-information escalation (design doc §1).

    Expected marginal value of buying one more action a:
        EMV(a) = ΔP(success | a) · V  −  cost(a)
    Take the action with the highest EMV; STOP when max_a EMV(a) < 0.

For the single escalate-or-stop decision this reduces to a break-even on the
belief p = P(resolve | escalate, evidence):

        escalate  iff  p · V > C_escalate        (equivalently  p > C/V)

The two pieces:
  1. BELIEF p  -- calibrated from EXECUTION evidence, not self-report (a
     controlled study proved self-report doesn't separate). v0 belief = the
     empirical per-grade conversion rate measured in that study (n=50 django).
     This is crude and SHOULD sharpen as (grade, outcome) pairs accrue -- the
     mechanism-mining loop (design §2d). Swap in a logistic/GBM on richer
     execution features later;
     the interface is just grade/features -> p.
  2. DECISION  -- the EMV break-even above. The C/V ratio is the operator's
     single tuning knob (cost tolerance): lower C/V escalates more grades.

VerifierGate is the special case of this formula where the abandon set =
{grades whose p falls below the break-even}. This module is the general form.

Belief sources are pluggable via the `PSource` protocol (plan §4 item 5): any
object with `p_resolve(evidence) -> float`. `GradeTableP(GRADE_P_RESOLVE_EXP1)`
is the shipped default and reproduces `voi_escalate_for_grade` exactly; richer
sources (e.g. tools/resolvability_adapter.ResolvabilityPSource, a persisted
sklearn pipeline over repo + issue-size features) plug in without touching the
decision arithmetic. Richer sources ship DISABLED (`p_source: null`) until the
promotion criteria are validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

from code_capsules.controller.verifier import (
    PARTIAL, FLAT, BROKEN, NOPATCH, UNKNOWN,
)

# v0 BELIEF: P(resolve after escalation | stage-1 grade), empirical from the
# calibration study (n=50 django, 2026-06-01). RESOLVED is omitted -- already
# resolved => stop,
# not an escalation candidate. These are crude (small per-grade n) and are the
# calibration the controller refines as more runs land.
GRADE_P_RESOLVE_EXP1 = {
    NOPATCH: 10 / 23,   # 0.435 -- "no patch yet" = needs more turns, often winnable
    BROKEN:  1 / 7,     # 0.143
    PARTIAL: 0 / 3,     # 0.000 (n=3, noisy)
    FLAT:    0 / 5,     # 0.000 -- ran, nothing moved (confident doomed)
    UNKNOWN: 0 / 6,     # 0.000
}


@runtime_checkable
class PSource(Protocol):
    """Pluggable VOI belief source: evidence -> P(resolve | escalate, evidence).

    `evidence` is a mapping. Recognized keys (all optional; a source uses what
    it understands and must tolerate missing keys):
      grade        -- stage-1 verifier grade (NOPATCH/BROKEN/PARTIAL/FLAT/UNKNOWN)
      repo         -- true repo name (derive from instance_id, NOT the JSONL
                      `repo` field, which is actually repo_kind -- plan §3 G8)
      issue_chars  -- turn-0 problem-statement length (chars)
      issue_words  -- turn-0 problem-statement length (words)
      tier         -- model tier of the stage-1 probe ("sonnet"/"haiku"/"opus")
      trace features -- optional early-dynamics features from the live stage-1
                      stream (see tools/resolvability_adapter.TraceFeatureExtractor)

    NEVER feed `dur_s` or any wall-time span that includes gold evaluation as
    evidence -- it is scoring-contaminated (plan §3 G6).
    """

    def p_resolve(self, evidence: Mapping) -> float:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class GradeTableP:
    """The default PSource: a grade -> p lookup table.

    `GradeTableP(GRADE_P_RESOLVE_EXP1)` makes `voi_escalate_for_evidence` the
    exact degenerate case of `voi_escalate_for_grade` (equivalence is pinned by
    tests/test_p_source.py). Unknown / missing grades map to p = 0.0, matching
    `belief.get(grade, 0.0)` in `voi_escalate_for_grade`.
    """

    table: Mapping[str, float] = field(
        default_factory=lambda: GRADE_P_RESOLVE_EXP1)

    def p_resolve(self, evidence: Mapping) -> float:
        return self.table.get(evidence.get("grade"), 0.0)


#: Shipped default belief source -- the current grade-table behavior. Richer
#: sources (resolvability adapter) are opt-in via policy `p_source`, which
#: ships as null/None (disabled) until tracking §5D promotion criteria are met.
DEFAULT_P_SOURCE = GradeTableP(GRADE_P_RESOLVE_EXP1)


@dataclass(frozen=True)
class VOIDecision:
    escalate: bool
    emv: float          # expected marginal value = p*V - C
    p_resolve: float    # the belief that drove it
    breakeven_p: float  # C / V  (escalate iff p_resolve > this)


def expected_marginal_value(p_resolve: float, value: float, cost: float) -> float:
    """EMV of one more escalation: ΔP·V − cost  (design §1)."""
    return p_resolve * value - cost


def voi_escalate(p_resolve: float, value: float, cost: float) -> VOIDecision:
    """Escalate iff expected marginal value is positive (the §1 stop rule)."""
    emv = expected_marginal_value(p_resolve, value, cost)
    breakeven = (cost / value) if value > 0 else float("inf")
    return VOIDecision(escalate=emv > 0.0, emv=emv,
                       p_resolve=p_resolve, breakeven_p=breakeven)


def voi_escalate_for_grade(grade: str, value: float, cost: float,
                           belief: dict[str, float] = GRADE_P_RESOLVE_EXP1) -> VOIDecision:
    """VOI escalate decision using the calibrated grade->p belief.

    value : V, the worth of one resolve (operator units -- e.g. $/resolve)
    cost  : C, the cost of one escalation (same units)
    """
    p = belief.get(grade, 0.0)
    return voi_escalate(p, value, cost)


def voi_escalate_for_evidence(evidence: Mapping, value: float, cost: float,
                              p_source: PSource | None = None) -> VOIDecision:
    """VOI escalate decision from an evidence dict via a pluggable PSource.

    With the default `p_source` (None -> `DEFAULT_P_SOURCE`, i.e.
    `GradeTableP(GRADE_P_RESOLVE_EXP1)`) this is exactly
    `voi_escalate_for_grade(evidence["grade"], value, cost)` -- the degenerate
    case. Pass a richer PSource (e.g. ResolvabilityPSource) to use more of the
    evidence; the EMV break-even decision rule is unchanged.

    evidence : mapping with the keys documented on `PSource`
    value    : V, the worth of one resolve (operator units)
    cost     : C, the cost of one escalation (same units)
    """
    src = p_source if p_source is not None else DEFAULT_P_SOURCE
    return voi_escalate(src.p_resolve(evidence), value, cost)


__all__ = [
    "VOIDecision", "GRADE_P_RESOLVE_EXP1",
    "PSource", "GradeTableP", "DEFAULT_P_SOURCE",
    "expected_marginal_value", "voi_escalate", "voi_escalate_for_grade",
    "voi_escalate_for_evidence",
]

"""
Verifier-selector: choose the best candidate patch from K diverse attempts by
an execution verifier -- the QUALITY lever of the adaptive controller.

Diversity across configs gives coverage the single best config lacks (EXP-2:
Sonnet first-150, union of 9 configs = 86/150 vs best-single floor 75/150 =
+7pp; floor+siginject alone reach 85/86). Same-config resampling does NOT help
(+2.7pp -- failures are correlated). The verifier *realizes* the coverage by
picking the candidate its execution reading ranks highest -- the Agentless
verify-select principle applied to diverse Claude Code configs.

Generic by construction (neutrality mandate): select_by_verifier takes
candidates + a grade function (patch -> VerifierGrade). SWE-bench shims supply
the grade function:
  - gold-test grading (oracle) -> the +7pp coverage CEILING,
  - generated-reproduction-test grading -> the deployable number (lower, by the
    repro verifier's false-negative rate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from code_capsules.controller.verifier import (
    RESOLVED, PARTIAL, BROKEN, FLAT, NOPATCH, UNKNOWN,
)

# Selection preference: most-likely-resolved first. RESOLVED wins outright;
# PARTIAL (some movement) beats no-movement; the doomed tail is last. A patch
# that didn't apply (BROKEN) is preferred over an empty one (NOPATCH) only as a
# last resort -- both are non-resolving, but neither should beat PARTIAL.
# UNKNOWN=2 is deliberate: no-evidence beats negative-evidence. A real patch
# with no readable execution signal (UNKNOWN) outranks one the verifier
# affirmatively read as non-moving or non-applying (FLAT/BROKEN) and the empty
# patch (NOPATCH), but stays below any positive movement (PARTIAL).
_RANK = {RESOLVED: 5, PARTIAL: 3, UNKNOWN: 2, BROKEN: 1, FLAT: 1, NOPATCH: 0}


@dataclass(frozen=True)
class Candidate:
    config: str       # which diverse config produced this patch
    patch: str


@dataclass(frozen=True)
class Selection:
    config: str       # winning config
    patch: str        # winning patch
    grade: str        # its verifier grade
    n_candidates: int
    resolved_signal: bool  # True iff the selected grade is RESOLVED
    discriminated: bool = False  # True iff the winning rank STRICTLY exceeds the
                                 # best rank among the OTHER candidates (a tie ->
                                 # False: the verifier didn't distinguish the winner;
                                 # a sole candidate wins vacuously -> True)


def select_by_verifier(
    candidates: Sequence[Candidate],
    grade_fn: Callable[[str], str],
) -> Optional[Selection]:
    """Pick the candidate whose execution grade ranks highest.

    grade_fn(patch) -> VerifierGrade (one of controller.verifier.GRADES).
    Empty patches are graded NOPATCH without calling grade_fn. Ties are broken
    by candidate order (stable -- pass the cheapest/most-trusted config first).
    Returns None for an empty candidate list.
    """
    best: Optional[Candidate] = None
    best_grade = UNKNOWN
    best_rank = -1
    second_rank = -1          # best rank among the non-winning candidates
    for c in candidates:
        g = NOPATCH if not (c.patch or "").strip() else grade_fn(c.patch)
        r = _RANK.get(g, 0)
        if r > best_rank:
            best, best_grade, best_rank, second_rank = c, g, r, best_rank
        elif r > second_rank:
            second_rank = r
    if best is None:
        return None
    return Selection(
        config=best.config, patch=best.patch, grade=best_grade,
        n_candidates=len(candidates), resolved_signal=(best_grade == RESOLVED),
        discriminated=(best_rank > second_rank),
    )


def agreement_reading(
    candidates: Sequence[Candidate],
    grade_fn: Callable[[str], str],
) -> dict:
    """Cross-sample diverse-agreement signal for the COST lever (the read of the
    SAME K diverse candidates that select_by_verifier uses for the quality lever).

    Returns the signals dict CrossSampleAgreementCascade consumes:
      n_samples    -- K diverse candidates graded
      n_resolved   -- how many the verifier grades RESOLVED
      n_nopatch    -- how many produced NO patch (the has_patch escalation
                      trigger reads this -- the validated signaled10->opus
                      cascade escalates the no-patch case)
      failure_kind -- None (some sample resolved) | "all_failed" | "no_signal":
                      the APPROVED zero-resolved split (sign-off 2026-06-10).
                      "all_failed" needs >=1 REAL grade (FLAT/PARTIAL/BROKEN)
                      affirmatively confirming doom; "no_signal" means every
                      non-RESOLVED grade is UNKNOWN/NOPATCH -- NO usable
                      execution reading, so it is NOT a doom call (the cascade
                      escalates it if a stage remains, else SHIPS the fallback,
                      mirroring the EXP-4 harness NO_REPRO_FALLBACK -- this
                      removes the framework/harness divergence where the
                      framework abandoned the no-repro band the harness ships)
      all_failed   -- back-compat alias: failure_kind == "all_failed". NB this
                      is NARROWER than the old key (which was True for ANY
                      zero-resolved round, including the no-signal band).

    Empty patches grade NOPATCH without calling grade_fn (mirrors select_by_verifier).
    At deployment the grade_fn is the repro/regression verifier (no gold); the
    'agreement on failure' across diverse configs is the validated doomed signal
    (~98% in-family precision at K=2), so the cost lever needs no gold tests."""
    n = len(candidates)
    n_resolved = 0
    n_nopatch = 0
    n_real_fail = 0          # FLAT / PARTIAL / BROKEN: an actual execution reading
    for c in candidates:
        g = NOPATCH if not (c.patch or "").strip() else grade_fn(c.patch)
        if g == RESOLVED:
            n_resolved += 1
        elif g == NOPATCH:
            n_nopatch += 1
        elif g != UNKNOWN:
            n_real_fail += 1
    failure_kind = None
    if n > 0 and n_resolved == 0:
        failure_kind = "all_failed" if n_real_fail > 0 else "no_signal"
    return {"n_samples": n, "n_resolved": n_resolved, "n_nopatch": n_nopatch,
            "failure_kind": failure_kind,
            "all_failed": failure_kind == "all_failed"}


__all__ = ["Candidate", "Selection", "select_by_verifier", "agreement_reading"]

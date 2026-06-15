"""Unit tests for the verifier-selector (quality lever). Fast, no Docker."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from code_capsules.controller.selector import (  # noqa: E402
    Candidate, select_by_verifier, _RANK,
)
from code_capsules.controller.verifier import (  # noqa: E402
    RESOLVED, PARTIAL, FLAT, BROKEN, NOPATCH, UNKNOWN,
)


def test_empty_candidates():
    assert select_by_verifier([], lambda p: RESOLVED) is None


def test_picks_resolved_over_partial():
    cands = [Candidate("floor", "patchA"), Candidate("siginject", "patchB")]
    grades = {"patchA": PARTIAL, "patchB": RESOLVED}
    sel = select_by_verifier(cands, lambda p: grades[p])
    assert sel.config == "siginject" and sel.grade == RESOLVED and sel.resolved_signal


def test_partial_beats_flat():
    cands = [Candidate("floor", "flat"), Candidate("siginject", "partial")]
    grades = {"flat": FLAT, "partial": PARTIAL}
    sel = select_by_verifier(cands, lambda p: grades[p])
    assert sel.config == "siginject" and sel.grade == PARTIAL and not sel.resolved_signal


def test_empty_patch_graded_nopatch_without_calling_fn():
    calls = []
    def gfn(p):
        calls.append(p); return RESOLVED
    cands = [Candidate("floor", ""), Candidate("siginject", "real")]
    sel = select_by_verifier(cands, gfn)
    assert sel.config == "siginject"           # real patch wins over empty
    assert "" not in calls                     # empty patch never graded


def test_stable_tie_break_prefers_first():
    # equal grade -> first (cheapest/most-trusted) config wins
    cands = [Candidate("floor", "a"), Candidate("siginject", "b")]
    sel = select_by_verifier(cands, lambda p: PARTIAL)
    assert sel.config == "floor"


def test_rank_table_ordering():
    # NOPATCH < BROKEN/FLAT < UNKNOWN < PARTIAL < RESOLVED. UNKNOWN=2 is the
    # deliberate no-evidence-beats-negative-evidence point: a real patch with no
    # readable signal outranks affirmatively-negative readings but never PARTIAL.
    assert _RANK[NOPATCH] < _RANK[BROKEN] == _RANK[FLAT] < _RANK[UNKNOWN] \
        < _RANK[PARTIAL] < _RANK[RESOLVED]


def test_unknown_real_patch_beats_flat_and_nopatch():
    # no-evidence (UNKNOWN) beats negative-evidence (FLAT) and the empty patch
    grades = {"flat": FLAT, "mystery": UNKNOWN}
    sel = select_by_verifier(
        [Candidate("floor", "flat"), Candidate("siginject", "mystery"),
         Candidate("third", "")],
        lambda p: grades[p],
    )
    assert sel.config == "siginject" and sel.grade == UNKNOWN


def test_partial_still_beats_unknown():
    grades = {"mystery": UNKNOWN, "moved": PARTIAL}
    sel = select_by_verifier(
        [Candidate("floor", "mystery"), Candidate("siginject", "moved")],
        lambda p: grades[p],
    )
    assert sel.config == "siginject" and sel.grade == PARTIAL


def test_discriminated_false_on_tie():
    # equal ranks -> the verifier did NOT distinguish the winner
    sel = select_by_verifier(
        [Candidate("floor", "a"), Candidate("siginject", "b")], lambda p: PARTIAL)
    assert sel.discriminated is False
    # BROKEN and FLAT share a rank -> also a tie
    grades = {"a": BROKEN, "b": FLAT}
    sel = select_by_verifier(
        [Candidate("floor", "a"), Candidate("siginject", "b")],
        lambda p: grades[p])
    assert sel.discriminated is False


def test_discriminated_true_on_strict_win():
    grades = {"a": RESOLVED, "b": FLAT}
    sel = select_by_verifier(
        [Candidate("floor", "a"), Candidate("siginject", "b")],
        lambda p: grades[p])
    assert sel.grade == RESOLVED and sel.discriminated is True


def test_discriminated_vacuous_true_for_sole_candidate():
    # no other candidates -> the strict-exceeds condition holds vacuously
    sel = select_by_verifier([Candidate("floor", "a")], lambda p: FLAT)
    assert sel.discriminated is True


def test_coverage_recovery_simulation():
    """The lever in miniature: floor misses, siginject resolves, selector recovers.

    Reproduces the EXP-2 mechanism on a tiny cohort: where floor FLATs but
    siginject RESOLVES, verifier-select flips the instance to resolved.
    """
    # (floor_grade, siginject_grade, expect_recovered)
    cohort = [
        (FLAT, RESOLVED, True),     # floor misses, siginject gets it -> recover
        (RESOLVED, FLAT, True),     # floor already resolves -> stays resolved
        (FLAT, FLAT, False),        # neither -> not recovered
        (PARTIAL, RESOLVED, True),  # siginject strictly better -> recover
    ]
    recovered = 0
    for fg, sg, _ in cohort:
        gmap = {"f": fg, "s": sg}
        sel = select_by_verifier(
            [Candidate("floor", "f"), Candidate("siginject", "s")],
            lambda p: gmap[p],
        )
        if sel.resolved_signal:
            recovered += 1
    assert recovered == 3  # the three with a RESOLVED candidate


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} selector tests passed")

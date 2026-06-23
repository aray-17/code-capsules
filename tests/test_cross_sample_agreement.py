"""Tests for the CrossSampleAgreementCascade primitive + agreement_reading helper
(the validated VOI cost lever: diverse-agreement-on-failure -> escalate tier / abandon).
Runs under pytest or `python3` directly (env pytest is broken)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from code_capsules.controller.cascade_triggers import CrossSampleAgreementCascade  # noqa: E402
from code_capsules.controller.selector import Candidate, agreement_reading  # noqa: E402
from code_capsules.controller.verifier import RESOLVED, FLAT, NOPATCH  # noqa: E402

C = CrossSampleAgreementCascade(max_tier=2, min_samples=2)             # default: abandon
E = CrossSampleAgreementCascade(max_tier=2, min_samples=2, escalate_on_agreement=True)


def test_canonical_name_is_diverse_agreement_with_backcompat_alias():
    # The paper (Table 2, Listing 1) calls this "diverse_agreement"; the old
    # internal name stays as a back-compat alias the registry also resolves.
    assert C.name == "diverse_agreement"
    assert "cross_sample_agreement" in C.aliases


def test_escalation_kind_is_tier():
    # if escalation is enabled it is the MODEL TIER, not the turn budget
    assert C.escalation_kind == "tier"


def test_some_sample_resolved_stops():
    sig = {"n_samples": 2, "n_resolved": 1, "all_failed": False}
    assert C.decision(sig, current_tier=1) == C.STOP
    assert C.should_escalate(sig, current_tier=1) is False


def test_agreement_fail_abandons_by_default():
    # K>=2 diverse samples all failed -> validated economical action = ABANDON
    sig = {"n_samples": 2, "n_resolved": 0, "all_failed": True}
    assert C.decision(sig, current_tier=1) == C.ABANDON
    assert C.should_escalate(sig, current_tier=1) is False   # more same-tier compute is wasted


def test_agreement_fail_escalates_tier_only_when_opted_in():
    sig = {"n_samples": 2, "n_resolved": 0, "all_failed": True}
    assert E.decision(sig, current_tier=1) == E.ESCALATE_TIER   # opt-in + tier available
    assert E.should_escalate(sig, current_tier=1) is True
    assert E.decision(sig, current_tier=2) == E.ABANDON         # opt-in but at top tier


def test_low_k_gathers_more_samples():
    # K=1 (< min_samples) failing is only ~85% precise -> gather another sample,
    # do NOT abandon on weak evidence (and do NOT tier-escalate prematurely)
    sig = {"n_samples": 1, "n_resolved": 0, "all_failed": True}
    assert C.decision(sig, current_tier=1) == C.GATHER_SAMPLES
    assert E.decision(sig, current_tier=1) == E.GATHER_SAMPLES


def test_safe_fallback_resolved_flag():
    # no n_resolved/n_samples, but an explicit resolved flag -> STOP
    assert C.decision({"resolved": True}, current_tier=1) == C.STOP


def test_governor_default_and_validation():
    # hybrid_regok is the shipped default (the no-silent-abandon fix); modes closed.
    assert C.governor == "hybrid_regok" and C.ship_gate == "repro_or_regok"
    try:
        CrossSampleAgreementCascade(governor="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        CrossSampleAgreementCascade(ship_gate="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_hybrid_regok_rescues_would_be_abandon():
    # agreement-failure + selected regression channel True -> STOP (ship the
    # carried patch); False/None/absent -> ABANDON exactly as before.
    base = {"n_samples": 2, "n_resolved": 0, "all_failed": True}
    assert C.decision({**base, "selected_regression_ok": True}, current_tier=1) == C.STOP
    assert C.decision({**base, "selected_regression_ok": False}, current_tier=1) == C.ABANDON
    assert C.decision({**base, "selected_regression_ok": None}, current_tier=1) == C.ABANDON
    assert C.decision(base, current_tier=1) == C.ABANDON


def test_hybrid_rescue_does_not_preempt_gather_or_escalate():
    # weak evidence still gathers; opted-in escalation still takes the tier
    low_k = {"n_samples": 1, "n_resolved": 0, "selected_regression_ok": True}
    assert C.decision(low_k, current_tier=1) == C.GATHER_SAMPLES
    agree = {"n_samples": 2, "n_resolved": 0, "selected_regression_ok": True}
    assert E.decision(agree, current_tier=1) == E.ESCALATE_TIER
    assert E.decision(agree, current_tier=2) == E.STOP    # top tier -> rescue, not abandon


def test_pure_regok_keys_on_regression_channel_alone():
    P = CrossSampleAgreementCascade(max_tier=2, min_samples=2, governor="pure_regok")
    # ships on regok True even when every repro failed ...
    assert P.decision({"n_samples": 2, "n_resolved": 0,
                       "selected_regression_ok": True}, current_tier=1) == P.STOP
    # ... and declines a repro-pass whose regok is False or None (never rescues)
    assert P.decision({"n_samples": 2, "n_resolved": 1,
                       "selected_regression_ok": False}, current_tier=1) == P.ABANDON
    assert P.decision({"n_samples": 2, "n_resolved": 1,
                       "selected_regression_ok": None}, current_tier=1) == P.ABANDON


def test_ship_gate_and_mode_demotes_repro_pass_on_broken_regression():
    A = CrossSampleAgreementCascade(max_tier=2, min_samples=2,
                                    governor="agreement", ship_gate="repro_and_regok")
    resolved = {"n_samples": 2, "n_resolved": 1}
    # the AND-gate requires regok is not False: True/None ship, False demotes
    assert A.decision({**resolved, "selected_regression_ok": True}, current_tier=1) == A.STOP
    assert A.decision({**resolved, "selected_regression_ok": None}, current_tier=1) == A.STOP
    assert A.decision({**resolved, "selected_regression_ok": False}, current_tier=1) == A.ABANDON
    # demotion goes to the ESCALATE path when escalation is opted in
    AE = CrossSampleAgreementCascade(max_tier=2, min_samples=2, governor="agreement",
                                     ship_gate="repro_and_regok",
                                     escalate_on_agreement=True)
    assert AE.decision({**resolved, "selected_regression_ok": False},
                       current_tier=1) == AE.ESCALATE_TIER
    # default OR gate is unchanged: repro-pass ships regardless of regok
    assert C.decision({**resolved, "selected_regression_ok": False}, current_tier=1) == C.STOP


def test_agreement_reading_counts():
    grade = {"p_res": RESOLVED, "p_flat": FLAT, "p_res2": RESOLVED}
    cands = [Candidate("a", "p_res"), Candidate("b", "p_flat"), Candidate("c", "p_res2")]
    r = agreement_reading(cands, lambda p: grade[p])
    assert r == {"n_samples": 3, "n_resolved": 2, "n_nopatch": 0,
                 "failure_kind": None, "all_failed": False}


def test_agreement_reading_all_failed_and_empty_patch():
    # empty patch -> NOPATCH (not resolved), and a FLAT (a REAL execution
    # reading) confirms doom -> failure_kind "all_failed"
    cands = [Candidate("a", "   "), Candidate("b", "p_flat")]
    r = agreement_reading(cands, lambda p: FLAT)
    assert r == {"n_samples": 2, "n_resolved": 0, "n_nopatch": 1,
                 "failure_kind": "all_failed", "all_failed": True}


def test_agreement_reading_no_signal_split():
    # The APPROVED zero-resolved split (sign-off 2026-06-10): every non-RESOLVED
    # grade in {UNKNOWN, NOPATCH} = NO usable execution reading -> "no_signal"
    # (and the back-compat all_failed key is False -- it is NOT a doom call).
    from code_capsules.controller.verifier import UNKNOWN
    cands = [Candidate("a", "   "), Candidate("b", "p_unk")]
    r = agreement_reading(cands, lambda p: UNKNOWN)
    assert r == {"n_samples": 2, "n_resolved": 0, "n_nopatch": 1,
                 "failure_kind": "no_signal", "all_failed": False}
    # one REAL grade (FLAT/PARTIAL/BROKEN) among the failures flips it to doom
    grade = {"p_unk": UNKNOWN, "p_flat": FLAT}
    r = agreement_reading([Candidate("a", "p_unk"), Candidate("b", "p_flat")],
                          lambda p: grade[p])
    assert r["failure_kind"] == "all_failed" and r["all_failed"] is True


def test_cascade_no_signal_ships_fallback_or_escalates():
    # no_signal -> NOT doom: escalate when a stronger stage remains (opt-in
    # escalation), else STOP = ship-fallback (mirrors the benchmark harness
    # NO_REPRO_FALLBACK -- the framework used to ABANDON this band).
    sig = {"n_samples": 2, "n_resolved": 0, "failure_kind": "no_signal",
           "all_failed": False}
    assert C.decision(sig, current_tier=1) == C.STOP          # no escalation -> ship
    assert E.decision(sig, current_tier=1) == E.ESCALATE_TIER  # a stage remains
    assert E.decision(sig, current_tier=2) == E.STOP           # top of ladder -> ship
    # absent failure_kind (hand-built / logged signals) -> branch inert
    legacy = {"n_samples": 2, "n_resolved": 0, "all_failed": True}
    assert C.decision(legacy, current_tier=1) == C.ABANDON


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"all cross-sample-agreement checks PASS ({len(fns)}/{len(fns)})")

"""Tests for the escalation ladder (plan §4 item 4): per-stage configs +
per-stage triggers {regok_true, has_patch, no_signal, always, voi}, the
recorded decline reasons, and the APPROVED no_signal split's default behavior
(escalate if a stage remains, else ship-fallback = the harness
NO_REPRO_FALLBACK, now in the framework).

Mock sampler/grade_fn -> no docker. Runs under pytest or `python3` directly.
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from code_capsules.controller.orchestrator import (  # noqa: E402
    run_controller, run_round, EscalationStage,
)
from code_capsules.controller.escalation_policy import (  # noqa: E402
    TRIGGERS, stage_trigger_fires,
)
from code_capsules.controller.cascade_triggers import CrossSampleAgreementCascade  # noqa: E402
from code_capsules.controller.selector import Selection  # noqa: E402
from code_capsules.controller.verifier import (  # noqa: E402
    RESOLVED, FLAT, BROKEN, NOPATCH, UNKNOWN,
)

FIXTURE_FIRST150 = _ROOT / "evals/leakfree/exp4_lever_floor100_siginject.jsonl"

SONNET2 = ("floor", "siginject")     # the validated 2-config cheap-tier ensemble


def _sampler(patches):
    """patches keyed by (config, tier) -> patch string; records every call."""
    calls = []

    def sampler(config, tier):
        calls.append((config, tier))
        return patches[(config, tier)]

    return sampler, calls


def _sel(grade):
    return Selection(config="c", patch="p", grade=grade, n_candidates=2,
                     resolved_signal=(grade == RESOLVED))


# ── EscalationStage validation ────────────────────────────────────────────────

def test_stage_validation():
    assert "regok_true" in TRIGGERS and "voi" in TRIGGERS
    try:
        EscalationStage("opus", ("floor",), trigger="nope")
        assert False, "expected ValueError for unknown trigger"
    except ValueError:
        pass
    try:
        EscalationStage("opus", ("floor",), trigger="voi")   # no C/V configured
        assert False, "expected ValueError for voi without cost/value"
    except ValueError:
        pass
    try:
        EscalationStage("opus", ())
        assert False, "expected ValueError for empty configs"
    except ValueError:
        pass


def test_stages_and_tiers_are_mutually_exclusive():
    s = [EscalationStage("sonnet", SONNET2)]
    try:
        run_controller(["sonnet"], SONNET2, lambda c, t: "p", lambda p: FLAT, stages=s)
        assert False, "expected ValueError for stages= plus tiers="
    except ValueError:
        pass
    try:
        run_controller(sampler=lambda c, t: "p", grade_fn=lambda p: FLAT)
        assert False, "expected ValueError for neither stages= nor tiers=+configs="
    except ValueError:
        pass


# ── per-stage configs ─────────────────────────────────────────────────────────

def test_per_stage_configs_honored():
    # 2-config sonnet ensemble, single-config opus rung (the governor shape):
    # opus must sample ONLY its own configs tuple.
    stages = [EscalationStage("sonnet", SONNET2),
              EscalationStage("opus", ("floor",), trigger="always")]
    sampler, calls = _sampler({("floor", "sonnet"): "sf", ("siginject", "sonnet"): "ss",
                               ("floor", "opus"): "of"})
    grades = {"sf": FLAT, "ss": FLAT, "of": RESOLVED}
    run = run_controller(sampler=sampler, grade_fn=lambda p: grades[p], stages=stages)
    assert run.outcome == "RESOLVED" and run.final_tier == "opus"
    assert calls == [("floor", "sonnet"), ("siginject", "sonnet"), ("floor", "opus")]
    assert run.decline_reason is None


# ── trigger: regok_true (escalate-on-regression-green: 84/150 @ $1.60/res; Opus 20/21 in regok=1) ──

def _regok_ladder(regok):
    stages = [EscalationStage("sonnet", SONNET2),
              EscalationStage("opus", SONNET2, trigger="regok_true")]
    sampler, calls = _sampler({("floor", "sonnet"): "sf", ("siginject", "sonnet"): "ss",
                               ("floor", "opus"): "of", ("siginject", "opus"): "os"})
    grades = {"sf": FLAT, "ss": FLAT, "of": RESOLVED, "os": FLAT}
    run = run_controller(sampler=sampler, grade_fn=lambda p: grades[p], stages=stages,
                         regression_fn=(None if regok is None else (lambda p: regok)))
    return run, calls


def test_regok_true_trigger_fires_on_true():
    run, calls = _regok_ladder(True)
    assert run.outcome == "RESOLVED" and run.final_tier == "opus"
    assert ("floor", "opus") in calls                  # escalated into the opus rung


def test_regok_true_trigger_does_not_fire_on_false_or_none():
    for regok in (False, None):
        run, calls = _regok_ladder(regok)
        assert run.outcome == "ABANDONED" and run.final_tier == "sonnet"
        assert run.decline_reason == "trigger_not_fired"
        assert not any(t == "opus" for _c, t in calls)  # opus never sampled
        assert run.selection is not None                # best failed patch carried


def test_regok_true_unit():
    stage = EscalationStage("opus", ("floor",), trigger="regok_true")
    assert stage_trigger_fires(stage, {"selected_regression_ok": True}) is True
    assert stage_trigger_fires(stage, {"selected_regression_ok": False}) is False
    assert stage_trigger_fires(stage, {"selected_regression_ok": None}) is False
    assert stage_trigger_fires(stage, {}) is False


# ── trigger: has_patch (signaled10->opus_floor on NO-patch = 86 @ $0.805) ────

def test_has_patch_trigger_escalates_the_no_patch_case():
    # one candidate produced NO patch -> n_nopatch=1 -> fire (mirrors the
    # validated signaled10->opus cascade: escalate "no patch yet").
    stages = [EscalationStage("sonnet", SONNET2),
              EscalationStage("opus", ("floor",), trigger="has_patch")]
    sampler, calls = _sampler({("floor", "sonnet"): "sf", ("siginject", "sonnet"): "",
                               ("floor", "opus"): "of"})
    grades = {"sf": FLAT, "of": RESOLVED}              # "" never reaches grade_fn
    run = run_controller(sampler=sampler, grade_fn=lambda p: grades[p], stages=stages)
    assert run.outcome == "RESOLVED" and run.final_tier == "opus"
    assert ("floor", "opus") in calls


def test_has_patch_trigger_declines_when_every_candidate_patched():
    stages = [EscalationStage("sonnet", SONNET2),
              EscalationStage("opus", ("floor",), trigger="has_patch")]
    sampler, calls = _sampler({("floor", "sonnet"): "sf", ("siginject", "sonnet"): "ss",
                               ("floor", "opus"): "of"})
    grades = {"sf": FLAT, "ss": FLAT, "of": RESOLVED}
    run = run_controller(sampler=sampler, grade_fn=lambda p: grades[p], stages=stages)
    assert run.outcome == "ABANDONED" and run.decline_reason == "trigger_not_fired"
    assert not any(t == "opus" for _c, t in calls)


def test_has_patch_unit_and_selection_fallback():
    stage = EscalationStage("opus", ("floor",), trigger="has_patch")
    assert stage_trigger_fires(stage, {"n_nopatch": 1}) is True
    assert stage_trigger_fires(stage, {"n_nopatch": 0}) is False
    # hand-built signals without n_nopatch fall back to the selection's grade
    assert stage_trigger_fires(stage, {}, _sel(NOPATCH)) is True
    assert stage_trigger_fires(stage, {}, _sel(FLAT)) is False
    assert stage_trigger_fires(stage, {}, None) is True   # no candidate at all


# ── trigger: always / never ──────────────────────────────────────────────────

def test_always_trigger_is_unconditional():
    stage = EscalationStage("opus", ("floor",), trigger="always")
    assert stage_trigger_fires(stage, {}) is True
    assert stage_trigger_fires(stage, {"selected_regression_ok": False}) is True


def test_never_trigger_never_fires():
    stage = EscalationStage("opus", ("floor",), trigger="never")
    assert stage_trigger_fires(stage, {"selected_regression_ok": True,
                                       "n_nopatch": 2}) is False


# ── trigger: voi (imports value_of_information.voi_escalate) ─────────────────

def test_voi_trigger_breakeven_on_signal_belief():
    # escalate iff p*V > C: stage-configured C and V, p from signals.
    cheap = EscalationStage("opus", ("floor",), trigger="voi",
                            voi_cost=1.0, voi_value=10.0)   # breakeven p = 0.1
    assert stage_trigger_fires(cheap, {"p_resolve": 0.5}) is True
    assert stage_trigger_fires(cheap, {"p_resolve": 0.05}) is False
    pricey = EscalationStage("opus", ("floor",), trigger="voi",
                             voi_cost=9.0, voi_value=10.0)  # breakeven p = 0.9
    assert stage_trigger_fires(pricey, {"p_resolve": 0.5}) is False


def test_voi_trigger_grade_belief_fallback():
    # no p_resolve in signals -> per-grade belief on the SELECTED grade
    # (BROKEN -> 1/7 ~ 0.143; FLAT -> 0.0).
    s = lambda c, v: EscalationStage("opus", ("floor",), trigger="voi",  # noqa: E731
                                     voi_cost=c, voi_value=v)
    assert stage_trigger_fires(s(1.0, 10.0), {}, _sel(BROKEN)) is True    # 1.43 > 1
    assert stage_trigger_fires(s(2.0, 10.0), {}, _sel(BROKEN)) is False   # 1.43 < 2
    assert stage_trigger_fires(s(0.01, 10.0), {}, _sel(FLAT)) is False    # p = 0


def test_voi_trigger_through_the_ladder():
    # both sonnet candidates BROKEN (real failure -> all_failed; selected grade
    # BROKEN -> p ~ 0.143): C=1,V=10 escalates; C=2,V=10 declines.
    def ladder(cost):
        stages = [EscalationStage("sonnet", SONNET2),
                  EscalationStage("opus", ("floor",), trigger="voi",
                                  voi_cost=cost, voi_value=10.0)]
        sampler, calls = _sampler({("floor", "sonnet"): "sf",
                                   ("siginject", "sonnet"): "ss",
                                   ("floor", "opus"): "of"})
        grades = {"sf": BROKEN, "ss": BROKEN, "of": RESOLVED}
        return run_controller(sampler=sampler, grade_fn=lambda p: grades[p],
                              stages=stages), calls

    run, calls = ladder(1.0)
    assert run.outcome == "RESOLVED" and ("floor", "opus") in calls
    run, calls = ladder(2.0)
    assert run.outcome == "ABANDONED" and run.decline_reason == "trigger_not_fired"
    assert not any(t == "opus" for _c, t in calls)


# ── the APPROVED no_signal split (default policy) ────────────────────────────

def test_no_signal_escalates_when_a_stage_remains_regardless_of_trigger():
    # all grades UNKNOWN (real patches, no usable reading) -> failure_kind
    # no_signal -> climb even though the next trigger (regok_true, no channel)
    # would NOT fire: no-reading is never a doom call.
    stages = [EscalationStage("sonnet", SONNET2),
              EscalationStage("opus", ("floor",), trigger="regok_true")]
    sampler, calls = _sampler({("floor", "sonnet"): "sf", ("siginject", "sonnet"): "ss",
                               ("floor", "opus"): "of"})
    grades = {"sf": UNKNOWN, "ss": UNKNOWN, "of": RESOLVED}
    run = run_controller(sampler=sampler, grade_fn=lambda p: grades[p], stages=stages)
    assert run.outcome == "RESOLVED" and run.final_tier == "opus"
    assert ("floor", "opus") in calls


def test_no_signal_ships_fallback_at_top_of_ladder():
    # single-stage ladder, all UNKNOWN -> ship-fallback (mirrors the benchmark
    # harness NO_REPRO_FALLBACK): outcome is a SHIP decision with n_resolved=0,
    # NOT the silent ABANDON the framework used to produce on this band.
    stages = [EscalationStage("sonnet", SONNET2)]
    sampler, _ = _sampler({("floor", "sonnet"): "sf", ("siginject", "sonnet"): "ss"})
    run = run_controller(sampler=sampler, grade_fn=lambda p: UNKNOWN, stages=stages)
    assert run.outcome == "RESOLVED"                   # ship decision (fallback)
    assert run.selection is not None and run.selection.grade == UNKNOWN
    assert run.steps[0].n_resolved == 0 and run.decline_reason is None


def test_no_signal_ships_fallback_after_climbing_the_ladder():
    # climbs on no_signal at sonnet, hits no_signal again at opus (top) -> ship
    stages = [EscalationStage("sonnet", SONNET2),
              EscalationStage("opus", SONNET2, trigger="always")]
    sampler, _ = _sampler({("floor", "sonnet"): "sf", ("siginject", "sonnet"): "ss",
                           ("floor", "opus"): "of", ("siginject", "opus"): "os"})
    run = run_controller(sampler=sampler, grade_fn=lambda p: UNKNOWN, stages=stages)
    assert run.outcome == "RESOLVED" and run.final_tier == "opus"
    assert all(s.n_resolved == 0 for s in run.steps)


def test_no_signal_as_a_stage_trigger_gates_on_failure_kind():
    # trigger="no_signal" builds a rung that ONLY handles the no-reading band:
    # an all_failed doom does NOT enter it.
    stage = EscalationStage("opus", ("floor",), trigger="no_signal")
    assert stage_trigger_fires(stage, {"failure_kind": "no_signal"}) is True
    assert stage_trigger_fires(stage, {"failure_kind": "all_failed"}) is False


# ── decline reasons recorded ─────────────────────────────────────────────────

def test_all_failed_trigger_not_fired_declines_with_reason():
    run, _ = _regok_ladder(False)
    assert run.outcome == "ABANDONED" and run.decline_reason == "trigger_not_fired"


def test_top_of_ladder_decline():
    # doom at the LAST rung of a multi-stage ladder -> top_of_ladder
    stages = [EscalationStage("sonnet", SONNET2),
              EscalationStage("opus", SONNET2, trigger="always")]
    sampler, _ = _sampler({("floor", "sonnet"): "sf", ("siginject", "sonnet"): "ss",
                           ("floor", "opus"): "of", ("siginject", "opus"): "os"})
    run = run_controller(sampler=sampler, grade_fn=lambda p: FLAT, stages=stages)
    assert run.outcome == "ABANDONED" and run.final_tier == "opus"
    assert run.decline_reason == "top_of_ladder"
    assert run.selection is not None                   # carried for logging


def test_single_stage_doom_is_declined_doomed():
    stages = [EscalationStage("sonnet", SONNET2)]
    sampler, _ = _sampler({("floor", "sonnet"): "sf", ("siginject", "sonnet"): "ss"})
    run = run_controller(sampler=sampler, grade_fn=lambda p: FLAT, stages=stages)
    assert run.outcome == "ABANDONED" and run.decline_reason == "declined_doomed"


def test_hybrid_governor_keeps_terminal_say_when_trigger_not_fired():
    # agreement-failure, regok=True, but the next trigger is has_patch (not
    # fired: every candidate patched) -> the governor's terminal word still
    # rescues (no-silent-abandon), shipping the sonnet patch.
    stages = [EscalationStage("sonnet", SONNET2),
              EscalationStage("opus", ("floor",), trigger="has_patch")]
    sampler, calls = _sampler({("floor", "sonnet"): "sf", ("siginject", "sonnet"): "ss",
                               ("floor", "opus"): "of"})
    run = run_controller(sampler=sampler, grade_fn=lambda p: FLAT, stages=stages,
                         regression_fn=lambda p: True)
    assert run.outcome == "RESOLVED" and run.final_tier == "sonnet"
    assert not any(t == "opus" for _c, t in calls)
    assert run.decline_reason is None


# ── back-compat: the tiers= signature is unchanged ───────────────────────────

def test_tiers_backcompat_climbs_and_resolves():
    grades = {"sonnet:floor": FLAT, "sonnet:siginject": FLAT,
              "opus:floor": RESOLVED, "opus:siginject": FLAT}
    run = run_controller(["sonnet", "opus"], SONNET2,
                         lambda c, t: f"{t}:{c}", lambda p: grades[p])
    assert run.outcome == "RESOLVED" and run.final_tier == "opus"
    assert [s.decision for s in run.steps][0] == "ESCALATE_TIER"


def test_tiers_backcompat_single_tier_abandons_with_reason():
    run = run_controller(["sonnet"], SONNET2, lambda c, t: f"{t}:{c}", lambda p: FLAT)
    assert run.outcome == "ABANDONED" and run.final_tier == "sonnet"
    assert run.decline_reason == "declined_doomed"
    assert run.selection is not None and run.selection.patch == "sonnet:floor"


def test_tiers_backcompat_top_tier_abandon():
    run = run_controller(["sonnet", "opus"], SONNET2, lambda c, t: f"{t}:{c}",
                         lambda p: FLAT)
    assert run.outcome == "ABANDONED" and run.final_tier == "opus"
    assert run.decline_reason == "top_of_ladder"
    assert [s.decision for s in run.steps] == ["ESCALATE_TIER", "ABANDON"]


def test_tiers_backcompat_escalate_on_agreement_false_stays_at_tier1():
    # converted internally to trigger="never": tier-1 doom declines, opus never runs
    casc = CrossSampleAgreementCascade(max_tier=2, escalate_on_agreement=False)
    calls = []

    def sampler(c, t):
        calls.append((c, t))
        return f"{t}:{c}"

    run = run_controller(["sonnet", "opus"], SONNET2, sampler, lambda p: FLAT,
                         cascade=casc)
    assert run.outcome == "ABANDONED" and run.final_tier == "sonnet"
    assert run.decline_reason == "trigger_not_fired"
    assert not any(t == "opus" for _c, t in calls)


def test_tiers_backcompat_single_config_exhausts_not_abandons():
    # K=1 forms no doom call -> EXHAUSTED with the patch carried (unchanged)
    run = run_controller(["sonnet"], ["floor"], lambda c, t: f"{t}:{c}",
                         lambda p: FLAT)
    assert run.outcome == "EXHAUSTED" and run.decline_reason is None
    assert run.selection is not None


def test_run_round_exposes_signals_for_the_ladder():
    # the trigger arbitration reads ControllerResult.signals: agreement counts +
    # failure_kind + n_nopatch + the regression channel must all be present
    r = run_round(SONNET2, lambda c, t: {"floor": "pf", "siginject": ""}[c],
                  lambda p: FLAT, tier=1, regression_fn=lambda p: True)
    assert r.signals["n_samples"] == 2 and r.signals["n_nopatch"] == 1
    assert r.signals["failure_kind"] == "all_failed"
    assert r.signals["selected_regression_ok"] is True


# ── governor set-membership property on the first-150 fixture ────────────

def test_gov_v2_regok_true_escalation_set_on_first150_fixture():
    """Escalation candidates under the regok_true trigger == EXACTLY the logged
    ABANDON rows whose selected candidate's regression_ok is True (count 21) --
    the escalate-regok=1-to-Opus destination set (84/150 @ $1.60/res;
    Opus recoveries 20/21 in regok=1 vs 3/17 in regok=0)."""
    if not FIXTURE_FIRST150.exists():
        import pytest
        pytest.skip(f"fixture not on disk: {FIXTURE_FIRST150}")
    rows = [json.loads(line) for line in FIXTURE_FIRST150.read_text().splitlines()
            if line.strip()]
    assert len(rows) == 150
    opus_rung = EscalationStage("opus", ("floor",), trigger="regok_true")
    casc = CrossSampleAgreementCascade(max_tier=2, min_samples=2,
                                       escalate_on_agreement=True)

    abandons = [r for r in rows if r["lever_decision"] == "ABANDON"]
    fired, expected = set(), set()
    for r in abandons:
        regok = r["candidates"][r["selected_config"]].get("regression_ok")
        sig = dict(r["agreement"])
        sig["selected_regression_ok"] = regok
        # every logged ABANDON is an agreement-failure the ladder arbitrates
        assert casc.decision(sig, current_tier=1) == casc.ESCALATE_TIER
        if stage_trigger_fires(opus_rung, sig):
            fired.add(r["instance_id"])
        if regok is True:
            expected.add(r["instance_id"])
    assert fired == expected
    assert len(fired) == 21


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"all escalation-ladder checks PASS ({len(fns)}/{len(fns)})")

"""Tests for the runtime orchestrator (composes select + agreement + cascade).
Mock sampler/grade_fn -> no docker. Runs under pytest or `python3` directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from code_capsules.controller.orchestrator import run_round, run_controller  # noqa: E402
from code_capsules.controller.cascade_triggers import CrossSampleAgreementCascade  # noqa: E402
from code_capsules.controller.verifier import RESOLVED, FLAT  # noqa: E402

CONFIGS = ["floor", "siginject"]


def _sampler(patches):
    return lambda config, tier: patches[config]


def _grade(grades):
    # grade only gets called for non-empty patches
    calls = {"n": 0}
    def g(patch):
        calls["n"] += 1
        return grades[patch]
    return g, calls


class _SpyCascade(CrossSampleAgreementCascade):
    """Records the signals dict each decision() saw (to assert the threading)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = []

    def decision(self, signals, current_tier):
        self.seen.append(dict(signals))
        return super().decision(signals, current_tier)


def test_stop_when_a_sample_resolves():
    sampler = _sampler({"floor": "pf", "siginject": "ps"})
    g, _ = _grade({"pf": FLAT, "ps": RESOLVED})
    r = run_round(CONFIGS, sampler, g, tier=1)
    assert r.decision == CrossSampleAgreementCascade.STOP
    assert r.resolved is True and r.n_resolved == 1
    assert r.selection.grade == RESOLVED            # quality lever picked the resolver


def test_abandon_when_all_fail_default():
    sampler = _sampler({"floor": "pf", "siginject": "ps"})
    g, _ = _grade({"pf": FLAT, "ps": FLAT})
    r = run_round(CONFIGS, sampler, g, tier=1)
    assert r.decision == CrossSampleAgreementCascade.ABANDON   # validated economical action
    assert r.resolved is False and r.n_resolved == 0


def test_escalate_tier_only_when_opted_in():
    sampler = _sampler({"floor": "pf", "siginject": "ps"})
    g, _ = _grade({"pf": FLAT, "ps": FLAT})
    casc = CrossSampleAgreementCascade(escalate_on_agreement=True, max_tier=2)
    r = run_round(CONFIGS, sampler, g, tier=1, cascade=casc)
    assert r.decision == CrossSampleAgreementCascade.ESCALATE_TIER


def test_gather_samples_when_k_too_small():
    sampler = _sampler({"floor": "pf"})
    g, _ = _grade({"pf": FLAT})
    r = run_round(["floor"], sampler, g, tier=1)        # K=1 < min_samples
    assert r.decision == CrossSampleAgreementCascade.GATHER_SAMPLES


def test_grade_called_once_per_distinct_patch():
    # both levers share one grading pass (grading is the expensive op)
    sampler = _sampler({"floor": "pf", "siginject": "ps"})
    g, calls = _grade({"pf": FLAT, "ps": FLAT})
    run_round(CONFIGS, sampler, g, tier=1)
    assert calls["n"] == 2          # 2 distinct patches graded once each, not 4


def test_run_round_threads_regression_signals():
    # selected_regression_ok (the SELECTED candidate's verdict) + n_regression_ok
    # (count over all candidates) must reach the cascade's signals dict.
    sampler = _sampler({"floor": "pf", "siginject": "ps"})
    g, _ = _grade({"pf": FLAT, "ps": FLAT})
    spy = _SpyCascade(governor="agreement")
    regok = {"pf": True, "ps": False}
    run_round(CONFIGS, sampler, g, tier=1, cascade=spy,
              regression_fn=lambda p: regok[p])
    (sig,) = spy.seen
    assert sig["selected_regression_ok"] is True        # tie -> floor selected
    assert sig["n_regression_ok"] == 1


def test_run_round_without_regression_fn_signals_are_inert():
    # No regression adapter -> None/0 in the signals; decision unchanged even
    # under the default hybrid_regok governor (None never rescues).
    sampler = _sampler({"floor": "pf", "siginject": "ps"})
    g, _ = _grade({"pf": FLAT, "ps": FLAT})
    spy = _SpyCascade()                                  # default governor
    r = run_round(CONFIGS, sampler, g, tier=1, cascade=spy)
    (sig,) = spy.seen
    assert sig["selected_regression_ok"] is None and sig["n_regression_ok"] == 0
    assert r.decision == CrossSampleAgreementCascade.ABANDON


def test_run_round_regression_fn_once_per_distinct_patch():
    # The regression run is as expensive as grading: one run per DISTINCT patch
    # (selected + per-candidate counting share the cache); empty patches skipped.
    sampler = _sampler({"floor": "same", "siginject": "same"})
    g, _ = _grade({"same": FLAT})
    calls = []
    run_round(CONFIGS, sampler, g, tier=1,
              regression_fn=lambda p: (calls.append(p), True)[1])
    assert calls == ["same"]
    calls.clear()
    run_round(CONFIGS, _sampler({"floor": "real", "siginject": ""}),
              _grade({"real": FLAT})[0], tier=1,
              regression_fn=lambda p: (calls.append(p), True)[1])
    assert calls == ["real"]                             # empty patch never run


def test_governor_modes_end_to_end_through_run_round():
    sampler = _sampler({"floor": "pf", "siginject": "ps"})
    all_flat = {"pf": FLAT, "ps": FLAT}
    regok_true = lambda p: True   # noqa: E731

    # agreement mode preserves the old behavior: regok True cannot rescue
    r = run_round(CONFIGS, sampler, _grade(all_flat)[0], tier=1,
                  cascade=CrossSampleAgreementCascade(governor="agreement"),
                  regression_fn=regok_true)
    assert r.decision == CrossSampleAgreementCascade.ABANDON

    # hybrid_regok (default): the would-be ABANDON is rescued -> STOP (ship)
    r = run_round(CONFIGS, sampler, _grade(all_flat)[0], tier=1,
                  regression_fn=regok_true)
    assert r.decision == CrossSampleAgreementCascade.STOP
    assert r.selection is not None                       # the carried patch ships

    # pure_regok: ships on the regression channel alone ...
    r = run_round(CONFIGS, sampler, _grade(all_flat)[0], tier=1,
                  cascade=CrossSampleAgreementCascade(governor="pure_regok"),
                  regression_fn=regok_true)
    assert r.decision == CrossSampleAgreementCascade.STOP
    # ... and declines a repro-pass whose regression channel is False
    r = run_round(CONFIGS, sampler, _grade({"pf": RESOLVED, "ps": FLAT})[0], tier=1,
                  cascade=CrossSampleAgreementCascade(governor="pure_regok"),
                  regression_fn=lambda p: False)
    assert r.decision == CrossSampleAgreementCascade.ABANDON


def test_ship_gate_and_mode_through_run_round():
    # AND mode demotes a repro-pass SHIP whose selected regression_ok is False;
    # None (no verdict) does NOT demote.
    sampler = _sampler({"floor": "pf", "siginject": "ps"})
    grades = {"pf": RESOLVED, "ps": FLAT}
    casc = lambda: CrossSampleAgreementCascade(  # noqa: E731
        governor="agreement", ship_gate="repro_and_regok")
    r = run_round(CONFIGS, sampler, _grade(grades)[0], tier=1, cascade=casc(),
                  regression_fn=lambda p: False)
    assert r.decision == CrossSampleAgreementCascade.ABANDON   # demoted
    r = run_round(CONFIGS, sampler, _grade(grades)[0], tier=1, cascade=casc())
    assert r.decision == CrossSampleAgreementCascade.STOP      # None passes the gate


def test_run_controller_forwards_regression_fn():
    # Single-tier ladder, both candidates repro-FLAT, regression green -> the
    # default hybrid governor ships the carried patch instead of abandoning.
    run = run_controller(["sonnet"], CONFIGS, lambda c, t: f"{t}:{c}",
                         lambda p: FLAT, regression_fn=lambda p: True)
    assert run.outcome == "RESOLVED"
    assert run.selection is not None and run.selection.patch == "sonnet:floor"
    # without the channel the same run abandons (regression evidence was the rescue)
    run = run_controller(["sonnet"], CONFIGS, lambda c, t: f"{t}:{c}", lambda p: FLAT)
    assert run.outcome == "ABANDONED"


def test_tier_loop_stops_at_tier1_when_resolved():
    # tier1 (sonnet) resolves -> never climbs to tier2 (opus)
    def sampler(config, tier):
        return f"{tier}:{config}"
    grades = {"sonnet:floor": FLAT, "sonnet:siginject": RESOLVED}
    run = run_controller(["sonnet", "opus"], CONFIGS, sampler, lambda p: grades[p])
    assert run.outcome == "RESOLVED" and run.final_tier == "sonnet"
    assert len(run.steps) == 1                       # opus tier never sampled


def test_tier_loop_escalates_then_resolves():
    # tier1 agreement-fail -> escalate -> tier2 (opus) resolves
    def sampler(config, tier):
        return f"{tier}:{config}"
    grades = {"sonnet:floor": FLAT, "sonnet:siginject": FLAT,
              "opus:floor": RESOLVED, "opus:siginject": FLAT}
    run = run_controller(["sonnet", "opus"], CONFIGS, sampler, lambda p: grades[p])
    assert run.outcome == "RESOLVED" and run.final_tier == "opus"
    assert [s.decision for s in run.steps][0] == "ESCALATE_TIER"
    assert len(run.steps) == 2


def test_tier_loop_abandons_at_top_tier_when_all_fail():
    # climb sonnet -> opus, agreement-fail at the top tier -> ABANDON (the cascade
    # abandons at current_tier>=max_tier); EXHAUSTED is only a mis-sized-ladder net
    def sampler(config, tier):
        return f"{tier}:{config}"
    run = run_controller(["sonnet", "opus"], CONFIGS, sampler, lambda p: FLAT)
    assert run.outcome == "ABANDONED" and run.final_tier == "opus"
    assert [s.decision for s in run.steps] == ["ESCALATE_TIER", "ABANDON"]


def test_single_tier_abandons_on_agreement_fail():
    # one-tier ladder + agreement-fail -> ABANDON at tier1 (the economical default)
    run = run_controller(["sonnet"], CONFIGS, lambda c, t: f"{t}:{c}", lambda p: FLAT)
    assert run.outcome == "ABANDONED" and run.final_tier == "sonnet"


def test_abandon_carries_selection_for_logging():
    # ABANDON must NOT lose the patch: the governor says don't-ship, but the caller
    # still needs the best failed candidate to log/score it (outcome says ship, not None).
    run = run_controller(["sonnet"], CONFIGS, lambda c, t: f"{t}:{c}", lambda p: FLAT)
    assert run.outcome == "ABANDONED"
    assert run.selection is not None and run.selection.patch == "sonnet:floor"


def test_single_config_resolves_returns_selection():
    # CONVERGENCE: a one-config policy == "just run this one variant". Resolves -> RESOLVED.
    run = run_controller(["sonnet"], ["floor"], lambda c, t: f"{t}:{c}",
                         lambda p: RESOLVED)
    assert run.outcome == "RESOLVED" and run.final_tier == "sonnet"
    assert run.selection.patch == "sonnet:floor" and run.selection.grade == RESOLVED


def test_single_config_unresolved_is_exhausted_not_abandoned():
    # CONVERGENCE: one config can't form a cross-sample DOOM call (K=1 < min_samples),
    # so an unresolved single-variant run is EXHAUSTED (ladder done), NOT ABANDONED,
    # and it still carries the patch. This is the "run one variant, get its result" path.
    run = run_controller(["sonnet"], ["floor"], lambda c, t: f"{t}:{c}",
                         lambda p: FLAT)
    assert run.outcome == "EXHAUSTED"
    assert run.selection is not None and run.selection.patch == "sonnet:floor"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"all orchestrator checks PASS ({len(fns)}/{len(fns)})")

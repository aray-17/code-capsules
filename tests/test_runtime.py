"""Tests for the Phase 11 runtime entry point (CodeCapsulesRunner). Mock adapters
-> no docker. The Runner must drive the validated controller end-to-end and honor
the policy — including the plan §4 item 6 policy.yaml surface (governor /
ship_gate / no_signal / escalation{mode,ladder,voi} / p_source /
prompt_includes_fail_to_pass). Runs under pytest or `python3` directly."""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))   # tools.* importable (resolvability adapter)

from code_capsules.controller.runtime import (  # noqa: E402
    CodeCapsulesRunner, RunnerPolicy, PSourceSpec,
)
from code_capsules.controller.escalation_policy import EscalationStage  # noqa: E402
from code_capsules.controller.verifier import RESOLVED, FLAT, UNKNOWN  # noqa: E402


def test_default_policy_abandons_on_agreement_fail():
    # validated default: single tier, abandon-on-agreement (no tier-escalation)
    r = CodeCapsulesRunner()
    run = r.run(lambda c, t: f"{t}:{c}", lambda p: FLAT)
    assert run.outcome == "ABANDONED"


def test_default_policy_resolves_when_a_config_wins():
    r = CodeCapsulesRunner()
    grades = {"default:floor": FLAT, "default:siginject": RESOLVED}
    run = r.run(lambda c, t: f"{t}:{c}", lambda p: grades[p])
    assert run.outcome == "RESOLVED"
    assert run.selection.grade == RESOLVED


def test_multi_tier_policy_escalates_then_resolves():
    pol = RunnerPolicy(tiers=("sonnet", "opus"), escalate_on_agreement=True)
    r = CodeCapsulesRunner(pol)
    grades = {"sonnet:floor": FLAT, "sonnet:siginject": FLAT,
              "opus:floor": RESOLVED, "opus:siginject": FLAT}
    run = r.run(lambda c, t: f"{t}:{c}", lambda p: grades[p])
    assert run.outcome == "RESOLVED" and run.final_tier == "opus"
    assert [s.decision for s in run.steps][0] == "ESCALATE_TIER"


def test_from_policy_dict_roundtrips_and_defaults():
    pol = RunnerPolicy.from_dict({"tiers": ["a", "b"], "escalate_on_agreement": True})
    assert pol.tiers == ("a", "b") and pol.escalate_on_agreement is True
    assert pol.configs == ("floor", "siginject")     # validated default kept
    assert pol.min_samples == 2
    assert pol.governor == "hybrid_regok"            # shipped governor default
    assert pol.ship_gate == "repro_or_regok"         # AND-gate is opt-in
    # empty/None -> all validated defaults
    assert RunnerPolicy.from_dict({}).tiers == ("default",)


def test_policy_governor_and_ship_gate_parse_and_forward():
    pol = RunnerPolicy.from_dict({"governor": "agreement",
                                  "ship_gate": "repro_and_regok"})
    assert pol.governor == "agreement" and pol.ship_gate == "repro_and_regok"
    casc = CodeCapsulesRunner(pol)._cascade()
    assert casc.governor == "agreement" and casc.ship_gate == "repro_and_regok"
    # defaults forward too
    casc = CodeCapsulesRunner()._cascade()
    assert casc.governor == "hybrid_regok" and casc.ship_gate == "repro_or_regok"


def test_run_regression_fn_rescues_abandon_under_default_governor():
    # Both configs repro-FLAT; the regression channel says the selected patch
    # keeps the suite green -> hybrid_regok ships the carried patch (no silent
    # abandon). Without the adapter the same run abandons.
    r = CodeCapsulesRunner()
    run = r.run(lambda c, t: f"{t}:{c}", lambda p: FLAT, regression_fn=lambda p: True)
    assert run.outcome == "RESOLVED" and run.selection is not None
    assert r.run(lambda c, t: f"{t}:{c}", lambda p: FLAT).outcome == "ABANDONED"


def test_run_agreement_governor_preserves_old_behavior():
    pol = RunnerPolicy(governor="agreement")
    run = CodeCapsulesRunner(pol).run(lambda c, t: f"{t}:{c}", lambda p: FLAT,
                                      regression_fn=lambda p: True)
    assert run.outcome == "ABANDONED"                # regok cannot rescue here


def test_from_policy_file_loads_and_falls_back(tmp_path=None):
    import tempfile, os
    # explicit controller block -> overrides applied
    p = tempfile.mktemp(suffix=".yaml")
    open(p, "w").write("controller:\n  tiers: [sonnet, opus]\n  escalate_on_agreement: true\n")
    r = CodeCapsulesRunner.from_policy_file(p)
    assert r.policy.tiers == ("sonnet", "opus") and r.policy.escalate_on_agreement is True
    assert r.policy.configs == ("floor", "siginject")     # default kept
    os.unlink(p)
    # missing file / missing block -> validated defaults (no crash)
    assert CodeCapsulesRunner.from_policy_file("/no/such/file.yaml").policy.tiers == ("default",)
    q = tempfile.mktemp(suffix=".yaml"); open(q, "w").write("routing: {}\n")  # no controller block
    assert CodeCapsulesRunner.from_policy_file(q).policy.configs == ("floor", "siginject")
    os.unlink(q)


# ── plan §4 item 6: the full controller-block policy surface ─────────────────

FULL_BLOCK_YAML = """\
controller:
  configs: [floor, siginject]
  tiers: [sonnet]
  governor: hybrid_regok
  ship_gate: repro_and_regok
  no_signal: escalate
  min_samples: 2
  escalation:
    mode: gate
    ladder:
      - {tier: opus, configs: [floor], trigger: regok_true}
      - {tier: opus, configs: [floor, siginject], trigger: voi}
    voi: {value_per_resolve: 1.60, cost_per_escalation: 0.42}
  p_source: {kind: resolvability_gbm, model: evals/models/doomed.joblib, features: static}
  prompt_includes_fail_to_pass: false
"""


def test_full_controller_block_round_trips_from_yaml():
    # every new key in one block, loaded through the REAL yaml path
    import tempfile, os
    p = tempfile.mktemp(suffix=".yaml")
    open(p, "w").write(FULL_BLOCK_YAML)
    pol = CodeCapsulesRunner.from_policy_file(p).policy
    os.unlink(p)
    assert pol.configs == ("floor", "siginject") and pol.tiers == ("sonnet",)
    assert pol.governor == "hybrid_regok" and pol.ship_gate == "repro_and_regok"
    assert pol.no_signal == "escalate" and pol.min_samples == 2
    assert pol.escalation_mode == "gate"
    assert pol.escalate_on_agreement is True          # mode gate implies the alias
    s0, s1 = pol.escalation_ladder
    assert isinstance(s0, EscalationStage)            # ladder rows ARE EscalationStages
    assert s0.tier == "opus" and s0.configs == ("floor",) and s0.trigger == "regok_true"
    assert s1.configs == ("floor", "siginject") and s1.trigger == "voi"
    assert s1.voi_value == 1.60 and s1.voi_cost == 0.42   # block voi = row default
    assert pol.escalation_voi == (1.60, 0.42)
    assert pol.p_source == PSourceSpec(kind="resolvability_gbm",
                                       model="evals/models/doomed.joblib",
                                       features="static")
    assert pol.prompt_includes_fail_to_pass is False


def test_empty_controller_block_is_todays_behavior():
    # defaults-on-absent: every new key takes the validated default and the
    # run behaves exactly like the pre-item-6 runner
    pol = RunnerPolicy.from_dict({})
    assert pol.governor == "hybrid_regok" and pol.ship_gate == "repro_or_regok"
    assert pol.no_signal == "ship_fallback"
    assert pol.escalation_mode == "off" and pol.escalation_ladder == ()
    assert pol.escalation_voi is None and pol.p_source is None
    assert pol.prompt_includes_fail_to_pass is True
    assert pol == RunnerPolicy()                       # identical to direct defaults
    r = CodeCapsulesRunner(pol)
    assert r.run(lambda c, t: f"{t}:{c}", lambda p: FLAT).outcome == "ABANDONED"
    grades = {"default:floor": FLAT, "default:siginject": RESOLVED}
    assert r.run(lambda c, t: f"{t}:{c}", lambda p: grades[p]).outcome == "RESOLVED"


def test_escalate_on_agreement_alias_maps_to_gate_mode():
    # deprecated alias: true -> escalation {mode: gate} with an empty ladder
    # (the legacy unconditional climb over `tiers`); false/absent -> mode off
    pol = RunnerPolicy.from_dict({"tiers": ["sonnet", "opus"],
                                  "escalate_on_agreement": True})
    assert pol.escalation_mode == "gate" and pol.escalation_ladder == ()
    assert pol.escalate_on_agreement is True
    grades = {"sonnet:floor": FLAT, "sonnet:siginject": FLAT,
              "opus:floor": RESOLVED, "opus:siginject": FLAT}
    run = CodeCapsulesRunner(pol).run(lambda c, t: f"{t}:{c}", lambda p: grades[p])
    assert run.outcome == "RESOLVED" and run.final_tier == "opus"   # legacy climb
    assert RunnerPolicy.from_dict({"escalate_on_agreement": False}).escalation_mode == "off"
    # an explicit escalation block WINS over the alias (mode off silences it)
    pol = RunnerPolicy.from_dict({"escalate_on_agreement": True,
                                  "escalation": {"mode": "off"}})
    assert pol.escalation_mode == "off" and pol.escalate_on_agreement is False


def test_yaml_bare_off_normalizes():
    # YAML 1.1 parses bare `off` as boolean False; the parser must normalize
    import tempfile, os
    p = tempfile.mktemp(suffix=".yaml")
    open(p, "w").write("controller:\n  escalation:\n    mode: off\n")
    pol = CodeCapsulesRunner.from_policy_file(p).policy
    os.unlink(p)
    assert pol.escalation_mode == "off" and pol.escalate_on_agreement is False


def test_unknown_values_raise_value_error():
    bads = [
        {"governor": "bogus"},
        {"ship_gate": "bogus"},
        {"no_signal": "bogus"},
        {"escalation": {"mode": "bogus"}},
        {"escalation": {"mode": "gate",
                        "ladder": [{"tier": "opus", "configs": ["floor"],
                                    "trigger": "bogus"}]}},
        {"escalation": {"mode": "gate", "ladder": [{"configs": ["floor"]}]}},  # no tier
        {"escalation": {"mode": "gate", "voi": {"value_per_resolve": 1.0}}},   # half a voi
        {"prompt_includes_fail_to_pass": "yes"},       # validated bool, not truthy
        {"p_source": {"kind": "bogus", "model": "m.joblib"}},
        {"p_source": {"kind": "resolvability_gbm"}},   # missing model path
        {"p_source": {"kind": "resolvability_gbm", "model": "m.joblib",
                      "features": "bogus"}},
        {"p_source": "not-a-mapping"},
        {"tiers": ["sonnet", "opus"],                  # ladder + multi-tiers ambiguous
         "escalation": {"mode": "gate",
                        "ladder": [{"tier": "opus", "configs": ["floor"]}]}},
    ]
    for bad in bads:
        try:
            RunnerPolicy.from_dict(bad)
            assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass


def test_runner_drives_the_policy_ladder():
    # mode gate + ladder -> staged run: base (tiers[0] x configs) then the rungs
    pol = RunnerPolicy.from_dict({
        "tiers": ["sonnet"],
        "escalation": {"mode": "gate",
                       "ladder": [{"tier": "opus", "configs": ["floor"],
                                   "trigger": "regok_true"}]}})
    grades = {"sonnet:floor": FLAT, "sonnet:siginject": FLAT, "opus:floor": RESOLVED}
    run = CodeCapsulesRunner(pol).run(lambda c, t: f"{t}:{c}", lambda p: grades[p],
                                      regression_fn=lambda p: True)
    assert run.outcome == "RESOLVED" and run.final_tier == "opus"   # GOV-V2 shape
    # trigger not fired (regok False) -> decline with reason, opus never sampled
    calls = []
    def sampler(c, t):
        calls.append((c, t)); return f"{t}:{c}"
    run = CodeCapsulesRunner(pol).run(sampler, lambda p: FLAT,
                                      regression_fn=lambda p: False)
    assert run.outcome == "ABANDONED" and run.decline_reason == "trigger_not_fired"
    assert not any(t == "opus" for _c, t in calls)


def test_mode_off_keeps_the_ladder_inert():
    # the shipped default: a configured ladder is documentation until mode: gate
    pol = RunnerPolicy.from_dict({
        "escalation": {"mode": "off",
                       "ladder": [{"tier": "opus", "configs": ["floor"],
                                   "trigger": "always"}]}})
    assert pol.escalation_ladder and pol.escalation_mode == "off"
    calls = []
    def sampler(c, t):
        calls.append((c, t)); return f"{t}:{c}"
    run = CodeCapsulesRunner(pol).run(sampler, lambda p: FLAT)
    assert run.outcome == "ABANDONED"                  # single-stage, same as today
    assert not any(t == "opus" for _c, t in calls)


def test_no_signal_policy_forwards_and_changes_behavior():
    # ship_fallback (default): the no-reading band ships the carried selection
    run = CodeCapsulesRunner().run(lambda c, t: f"{t}:{c}", lambda p: UNKNOWN)
    assert run.outcome == "RESOLVED" and run.selection.grade == UNKNOWN
    # abandon: the same round declines (never ship unverified), reason recorded
    pol = RunnerPolicy.from_dict({"no_signal": "abandon"})
    assert CodeCapsulesRunner(pol)._cascade().no_signal == "abandon"   # forwarded
    run = CodeCapsulesRunner(pol).run(lambda c, t: f"{t}:{c}", lambda p: UNKNOWN)
    assert run.outcome == "ABANDONED" and run.decline_reason == "no_signal_abandon"
    # escalate: climbs while a rung remains, declines (not ships) at the top
    pol = RunnerPolicy.from_dict({"no_signal": "escalate"})
    run = CodeCapsulesRunner(pol).run(lambda c, t: f"{t}:{c}", lambda p: UNKNOWN)
    assert run.outcome == "ABANDONED" and run.decline_reason == "no_signal_abandon"
    pol = RunnerPolicy.from_dict({
        "no_signal": "escalate", "tiers": ["sonnet"],
        "escalation": {"mode": "gate",
                       "ladder": [{"tier": "opus", "configs": ["floor"],
                                   "trigger": "regok_true"}]}})
    grades = {"sonnet:floor": UNKNOWN, "sonnet:siginject": UNKNOWN,
              "opus:floor": RESOLVED}
    run = CodeCapsulesRunner(pol).run(lambda c, t: f"{t}:{c}", lambda p: grades[p])
    assert run.outcome == "RESOLVED" and run.final_tier == "opus"   # climbed the band
    # abandon: never spends escalation on the band even when a rung remains
    pol = RunnerPolicy.from_dict({
        "no_signal": "abandon", "tiers": ["sonnet"],
        "escalation": {"mode": "gate",
                       "ladder": [{"tier": "opus", "configs": ["floor"],
                                   "trigger": "always"}]}})
    calls = []
    def sampler(c, t):
        calls.append((c, t)); return f"{t}:{c}"
    run = CodeCapsulesRunner(pol).run(sampler, lambda p: UNKNOWN)
    assert run.outcome == "ABANDONED" and run.decline_reason == "no_signal_abandon"
    assert not any(t == "opus" for _c, t in calls)


def _ml_modules():
    return {m for m in sys.modules if m.split(".")[0] in ("sklearn", "pandas", "joblib")}


def test_p_source_null_vs_dict_never_imports_sklearn_at_parse():
    # null (the shipped default) -> disabled
    assert RunnerPolicy.from_dict({"p_source": None}).p_source is None
    assert RunnerPolicy.from_dict({}).p_source is None
    # dict -> validated spec; the parse must NOT touch the ML stack (plan G9:
    # sklearn is broken in this env -- parse-time import would crash policies)
    before = _ml_modules()
    pol = RunnerPolicy.from_dict({"p_source": {"kind": "resolvability_gbm",
                                               "model": "evals/models/doomed.joblib"}})
    assert pol.p_source.kind == "resolvability_gbm"
    assert pol.p_source.model == "evals/models/doomed.joblib"
    assert pol.p_source.features == "static"           # default feature set
    assert _ml_modules() == before                     # no sklearn/pandas/joblib import


# NOTE: the spec-string p_source path ("resolvability_gbm" -> a trained GBM model)
# loads the optional ML resolvability adapter from the operational harness, which is
# not distributed in this public repo. The public path is to inject a PSource object
# directly (CodeCapsulesRunner(..., p_source=<obj>)), exercised by the test below and
# by tests/test_standalone_package.py.


def test_p_source_belief_reaches_the_voi_trigger():
    # a stub PSource (the protocol, not sklearn) wired through run_controller
    # decides the voi rung: p=0.5 with C=1,V=10 escalates; p=0.05 declines
    from code_capsules.controller.orchestrator import run_controller
    class StubP:
        def __init__(self, p): self.p = p
        def p_resolve(self, evidence): return self.p
    def ladder_run(p):
        stages = (EscalationStage("sonnet", ("floor", "siginject")),
                  EscalationStage("opus", ("floor",), trigger="voi",
                                  voi_cost=1.0, voi_value=10.0))
        grades = {"sonnet:floor": FLAT, "sonnet:siginject": FLAT,
                  "opus:floor": RESOLVED}
        return run_controller(sampler=lambda c, t: f"{t}:{c}",
                              grade_fn=lambda p_: grades[p_],
                              stages=stages, p_source=StubP(p))
    assert ladder_run(0.5).outcome == "RESOLVED"       # 0.5*10 > 1 -> escalate
    assert ladder_run(0.05).outcome == "ABANDONED"     # 0.05*10 < 1 -> decline


def test_prompt_flag_parses_validates_and_plumbs_to_build_prompt():
    from code_capsules.controller.prompt import build_prompt
    inst = {"problem_statement": "It crashes.",
            "FAIL_TO_PASS": ["tests/test_a.py::test_x"]}
    # default True == ALL historical evals: the section is present
    pol = RunnerPolicy.from_dict({})
    assert pol.prompt_includes_fail_to_pass is True
    out = build_prompt(inst, include_fail_to_pass=pol.prompt_includes_fail_to_pass)
    assert "## Tests that must pass after your fix" in out
    # deployments can disable via the controller block (G2 disclosure flag)
    pol = RunnerPolicy.from_dict({"prompt_includes_fail_to_pass": False})
    out = build_prompt(inst, include_fail_to_pass=pol.prompt_includes_fail_to_pass)
    assert "Tests that must pass" not in out and "test_a.py" not in out


def test_shipped_policy_yaml_parses_with_the_new_surface():
    # the repo's own policy.yaml must round-trip through the new parser:
    # menu present as data, controller block carries the new keys, mode off
    pol = CodeCapsulesRunner.from_policy_file(_ROOT / "policy.yaml").policy
    assert pol.configs == ("floor", "siginject") and pol.tiers == ("sonnet",)
    assert pol.governor == "hybrid_regok" and pol.ship_gate == "repro_or_regok"
    assert pol.no_signal == "ship_fallback" and pol.min_samples == 2
    assert pol.escalation_mode == "off" and pol.escalate_on_agreement is False
    assert len(pol.escalation_ladder) == 1             # the GOV-V2 rung, inert
    assert pol.escalation_ladder[0].trigger == "regok_true"
    assert pol.escalation_voi == (1.37, 0.42)
    assert pol.p_source is None and pol.prompt_includes_fail_to_pass is True
    import yaml as _yaml
    raw = _yaml.safe_load((_ROOT / "policy.yaml").read_text())
    assert set(raw["menu"]) == {"cost_min", "balanced", "quality",
                                "quality_max", "ceiling"}
    assert raw["menu"]["quality_max"]["resolved"] == 128


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"all runtime checks PASS ({len(fns)}/{len(fns)})")

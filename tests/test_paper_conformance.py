"""Paper ↔ code conformance gate.

The paper (paper/paper.tex) is the frozen spec. This test asserts that every
load-bearing, paper-named extension surface and consumer API actually exists and
resolves in the code, so the two cannot silently drift:

  * Table 2 (the eight extension primitives): every shipped implementation NAME
    the paper lists resolves through the registry (vendor-SDK-gated clients are
    checked only when their SDK is importable, per the paper's "iff importable").
  * Listing 1 (the runnable consumer code): both option blocks construct verbatim.
  * Table 1 (the shipped cost-quality menu): all five knees resolve via policy_for
    with the variant the table names.
  * from_yaml loads the shipped repo-root policy.yaml.
  * The routing block wires a workload classifier + routing strategy end to end.

These are exactly the contracts the conformance audit found broken; this file
keeps them fixed.
"""
import importlib.util
from pathlib import Path

import pytest

from code_capsules.api import registry as R
from code_capsules.api import TaskDescriptor, VariantConfig

ROOT = Path(__file__).resolve().parents[1]


# ── Table 2: the eight extension primitives resolve by their paper names ──────

# (kind, name, requires_module). requires_module gates vendor-SDK clients on the
# paper's "registers iff the vendor SDK is importable" contract.
_TABLE2 = [
    ("model_client", "claude_cli", None),
    ("model_client", "openai_api", "openai"),
    ("model_client", "openai_responses", "openai"),
    ("model_client", "gemini_api", "google.genai"),
    ("classifier", "generic_prompt", None),
    ("classifier", "benchmark_saturated", None),
    ("classifier", "yaml_mapping", None),
    ("routing_strategy", "rule_based", None),
    ("routing_strategy", "cascade", None),
    ("routing_strategy", "default", None),
    ("signal", "cap_pressure", None),
    ("signal", "file_thrash", None),
    ("signal", "test_failure", None),
    ("signal", "traceback", None),
    ("quality_gate", "docker_eval", None),
    ("quality_gate", "binary_tests", None),
    ("quality_gate", "python_ast", None),
    ("quality_gate", "always_pass", None),
    ("cascade_trigger", "heuristic", None),
    ("cascade_trigger", "always_escalate", None),
    ("cascade_trigger", "never_escalate", None),
    ("cascade_trigger", "diverse_agreement", None),
    ("cascade_trigger", "verifier_gate", None),
    ("cost_model", "anthropic_public", None),
    ("cost_model", "openai_public", None),
    ("cost_model", "gemini_public", None),
]


@pytest.mark.parametrize("kind,name,requires", _TABLE2,
                         ids=[f"{k}:{n}" for k, n, _ in _TABLE2])
def test_table2_name_resolves(kind, name, requires):
    if requires is not None and importlib.util.find_spec(requires) is None:
        pytest.skip(f"{name} is vendor-SDK-gated; {requires} not importable")
    impl = R.get(kind, name)
    assert impl is not None
    assert getattr(impl, "name", None) == name


def test_diverse_agreement_backcompat_alias():
    """The pre-paper internal name still resolves to the same impl."""
    assert R.get("cascade_trigger", "cross_sample_agreement").name == "diverse_agreement"


def test_yaml_mapping_classifies_shipped_repos():
    clf = R.get("classifier", "yaml_mapping")
    assert clf.classify(TaskDescriptor(text="x", context={"repo": "matplotlib"})) == "visualization_workload"
    assert clf.classify(TaskDescriptor(text="x", context={"repo": "sympy"})) == "scientific_workload"
    assert clf.classify(TaskDescriptor(text="x", context={"repo": "django"})) == "hard_workload"


# ── Listing 1: the paper's runnable consumer code constructs verbatim ─────────

def test_listing1_option1_policy_for():
    from code_capsules import policy_for
    p = policy_for(workload="hard_workload", tier="sonnet", knee="balanced")
    assert p.variant == "two_pass_critique"


def test_listing1_option2_constructs_verbatim():
    from code_capsules import CodeCapsulesPolicy
    p = CodeCapsulesPolicy(
        variant="two_pass_critique",
        mode="escalating",
        turn_budget=10,
        escalating_start_budget=10,
        escalating_target_budget=25,
        always_escalate=True,
        prompt_variant="two_pass_critique",
        cascade_trigger="diverse_agreement",
        controller={
            "configs": ["unbounded_budget", "signaled_budget"],
            "tiers": ["sonnet"],
            "min_samples": 2,
            "governor": "hybrid_regok",
            "ship_gate": "repro_or_regok",
            "escalation": {
                "mode": "gate",
                "ladder": [{"tier": "opus", "configs": ["unbounded_budget"],
                            "trigger": "regok_true"}],
            },
        },
    )
    rp = p.runner_policy()
    assert rp.configs == ("unbounded_budget", "signaled_budget")
    assert rp.governor == "hybrid_regok"
    assert rp.escalate_on_agreement is True


def test_bad_controller_block_raises_at_construction():
    from code_capsules import CodeCapsulesPolicy
    with pytest.raises(ValueError):
        CodeCapsulesPolicy(controller={"governor": "bogus"})


# ── Table 1: the shipped menu resolves with the named variants ───────────────

@pytest.mark.parametrize("tier,knee,variant", [
    ("sonnet", "cost_min", "relevance_ranker"),
    ("sonnet", "balanced", "two_pass_critique"),
    ("sonnet", "quality", "unbounded_budget"),
    ("opus", "quality_max", "implicit_budget"),
    ("opus", "ceiling", "unbounded_budget"),
])
def test_shipped_menu_knees(tier, knee, variant):
    from code_capsules import policy_for
    p = policy_for(workload="hard_workload", tier=tier, knee=knee)
    assert p.variant == variant
    assert p.knee == knee


# ── from_yaml loads the shipped repo-root policy.yaml ─────────────────────────

@pytest.mark.parametrize("wl,tr,kn", [
    ("hard_workload", "sonnet", "cost_min"),
    ("hard_workload", "sonnet", "balanced"),
    ("hard_workload", "sonnet", "quality"),
    ("hard_workload", "opus", "quality_max"),
    ("hard_workload", "opus", "ceiling"),
    ("saturated_workload", "haiku", "recommended"),
])
def test_from_yaml_shipped_policy(wl, tr, kn):
    from code_capsules import CodeCapsulesPolicy
    p = CodeCapsulesPolicy.from_yaml(ROOT / "policy.yaml", workload=wl, tier=tr, knee=kn)
    assert p.workload_class == wl and p.tier == tr and p.knee == kn
    assert p.variant in __import__(
        "code_capsules.controller.policy", fromlist=["_VALID_VARIANTS"]
    )._VALID_VARIANTS


def test_from_policy_file_loads_controller_and_routing():
    from code_capsules.controller.runtime import CodeCapsulesRunner
    r = CodeCapsulesRunner.from_policy_file(str(ROOT / "policy.yaml"))
    assert r.policy.configs == ("floor", "siginject")
    assert r.policy.governor == "hybrid_regok"


# ── The routing block wires classifier + strategy end to end ─────────────────

def test_routing_block_wires_classifier_and_routes():
    from code_capsules.controller.yaml_dsl import load_policy
    from code_capsules.controller.routing_strategies import build_routing
    cfg = load_policy(str(ROOT / "policy.yaml"))
    clf, strat = build_routing(cfg)
    avail = {n: VariantConfig(name=n, mode="sequential")
             for n in ("stuck_signal_injection", "two_pass_critique", "relevance_ranker")}
    # scientific repo routes to the configured per-class variant; others -> default
    label = clf.classify(TaskDescriptor(text="x", context={"repo": "sympy"}))
    assert strat.route(label, avail).name == "stuck_signal_injection"
    label = clf.classify(TaskDescriptor(text="x", context={"repo": "django"}))
    assert strat.route(label, avail).name == "two_pass_critique"


# ── Extensibility: a custom impl of every primitive registers, conforms, selects ──
# The framework's headline claim (paper §"Extension surface"): every decision point
# is user-overridable by a one-line registry call, then selectable by name. These
# assert that end to end so it cannot silently regress (it had).

# (kind, Protocol-name, a minimal conforming impl)
def _stub_impls():
    from code_capsules import api
    return [
        ("classifier", api.WorkloadClassifier,
         type("XClf", (), {"name": "x_clf", "classify": lambda self, t: "x"})()),
        ("routing_strategy", api.RoutingStrategy,
         type("XRoute", (), {"name": "x_route", "route": lambda self, l, a: next(iter(a.values()))})()),
        ("variant", api.Variant,
         type("XVar", (), {"name": "x_var", "run": lambda self, t, c: None})()),
        ("signal", api.Signal,
         type("XSig", (), {"name": "x_sig", "compute": lambda self, s: True})()),
        ("quality_gate", api.QualityGate,
         type("XGate", (), {"name": "x_gate", "check": lambda self, a: True})()),
        ("cascade_trigger", api.CascadeTrigger,
         type("XTrig", (), {"name": "x_trig", "should_escalate": lambda self, s, ct: False})()),
        ("cost_model", api.CostModel,
         type("XCost", (), {"name": "x_cost", "pricing_date": "2026-01-01",
                            "cost": lambda self, i, o, m, cached_input_tokens=0: 0.0})()),
        ("model_client", api.ModelClient,
         type("XClient", (), {"name": "x_client", "invoke": lambda self, p, **k: None})()),
    ]


@pytest.mark.parametrize("kind,proto,impl", _stub_impls(),
                         ids=[k for k, _, _ in _stub_impls()])
def test_custom_impl_registers_resolves_and_conforms(kind, proto, impl):
    R.register(kind, impl)
    got = R.get(kind, impl.name)
    assert got is impl
    assert isinstance(got, proto)          # Protocols are @runtime_checkable


def test_custom_names_selectable_through_typed_policy():
    """A registered custom variant/quality_gate/cascade_trigger is accepted by
    CodeCapsulesPolicy by name; an unregistered name still raises."""
    from code_capsules import CodeCapsulesPolicy
    for kind, _, impl in _stub_impls():
        R.register(kind, impl)
    CodeCapsulesPolicy(variant="x_var")
    CodeCapsulesPolicy(quality_gate="x_gate")
    CodeCapsulesPolicy(cascade_trigger="x_trig")
    with pytest.raises(ValueError):
        CodeCapsulesPolicy(quality_gate="not_registered_anywhere")

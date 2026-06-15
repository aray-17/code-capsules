"""Tests for code_capsules.policy.CodeCapsulesPolicy (declarative DSL)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from code_capsules import (
    CodeCapsulesPolicy,
    SHIPPED_PRESETS,
    policy_for,
)
from code_capsules.controller.policy import PolicyError


# ── Construction + defaults ───────────────────────────────────────────────

def test_default_construction():
    p = CodeCapsulesPolicy()
    assert p.variant == "signaled_budget"
    assert p.mode == "sequential"
    assert p.turn_budget == 20
    assert p.prompt_budget_hint is None
    assert p.always_escalate is False
    assert p.quality_gate == "docker_eval"
    assert p.cascade_trigger == "heuristic"
    assert p.workload_class == "hard_workload"
    assert p.tier == "sonnet"
    assert p.knee == "balanced"


def test_declarative_construction_with_kwargs():
    p = CodeCapsulesPolicy(
        variant="two_pass_critique",
        mode="escalating",
        turn_budget=10,
        escalating_target_budget=25,
        always_escalate=True,
        prompt_variant="two_pass_critique",
    )
    assert p.variant == "two_pass_critique"
    assert p.mode == "escalating"
    assert p.turn_budget == 10
    assert p.escalating_target_budget == 25
    assert p.always_escalate is True


# ── Validation ────────────────────────────────────────────────────────────

def test_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        CodeCapsulesPolicy(variant="not_a_variant")


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        CodeCapsulesPolicy(mode="invalid_mode")


def test_rejects_bad_turn_budget():
    with pytest.raises(ValueError, match="turn_budget"):
        CodeCapsulesPolicy(turn_budget=0)


def test_rejects_inverted_escalation_budgets():
    with pytest.raises(ValueError, match="escalating_target_budget"):
        CodeCapsulesPolicy(
            escalating_start_budget=20, escalating_target_budget=10
        )


def test_rejects_unknown_quality_gate():
    with pytest.raises(ValueError, match="quality_gate"):
        CodeCapsulesPolicy(quality_gate="fictional_gate")


def test_rejects_unknown_cascade_trigger():
    with pytest.raises(ValueError, match="cascade_trigger"):
        CodeCapsulesPolicy(cascade_trigger="cascade_unknown")


def test_pass_rate_out_of_range_rejected():
    with pytest.raises(ValueError, match="pass_rate"):
        CodeCapsulesPolicy(pass_rate=1.5)


def test_negative_cost_rejected():
    with pytest.raises(ValueError, match="cost_per_task"):
        CodeCapsulesPolicy(cost_per_task=-0.01)


# ── Shipped presets ───────────────────────────────────────────────────────

def test_policy_for_default_knee():
    p = policy_for()  # default args: hard_workload / sonnet / balanced
    assert p.variant == "two_pass_critique"
    assert p.mode == "escalating"
    assert p.knee == "balanced"
    assert p.pass_rate == pytest.approx(0.440)


def test_policy_for_cost_min():
    p = policy_for(knee="cost_min")
    assert p.variant == "relevance_ranker"
    assert p.turn_budget == 10
    assert p.cost_per_task == pytest.approx(0.18)


def test_policy_for_balanced():
    p = policy_for(knee="balanced")
    assert p.variant == "two_pass_critique"
    assert p.mode == "escalating"
    assert p.cost_per_task == pytest.approx(0.41)


def test_policy_for_quality_max():
    p = policy_for(tier="opus", knee="quality_max")
    assert p.variant == "implicit_budget"
    assert p.tier == "opus"
    assert p.turn_budget == 20
    assert p.cost_per_task == pytest.approx(0.48)


def test_policy_for_missing_cell_raises():
    # ceiling is an Opus-only knee; there is no Sonnet ceiling cell.
    with pytest.raises(ValueError, match="No shipped preset"):
        policy_for(workload="hard_workload", tier="sonnet", knee="ceiling")


def test_shipped_presets_dict_intact():
    assert ("hard_workload", "sonnet", "balanced") in SHIPPED_PRESETS
    assert ("hard_workload", "opus", "quality_max") in SHIPPED_PRESETS
    assert len(SHIPPED_PRESETS) >= 5  # five cross-tier knees ship by default


def test_replace_override_pattern():
    """Operators customize a preset via dataclasses.replace."""
    base = policy_for(knee="balanced")
    tighter = replace(base, turn_budget=15)
    assert tighter.variant == base.variant
    assert tighter.turn_budget == 15
    assert base.turn_budget == 10  # original unchanged


# ── YAML loaders ──────────────────────────────────────────────────────────

_SAMPLE_YAML = """
deployment_defaults:
  hard_workload:
    sonnet:
      balanced:
        variant: two_pass_critique
        mode: escalating
        turn_budget: 10
        escalating_start_budget: 10
        escalating_target_budget: 25
        always_escalate: true
        prompt_variant: two_pass_critique
        pass_rate: 0.467
        cost_per_task: 0.308
      cost_min:
        variant: signaled_budget
        mode: sequential
        turn_budget: 10
        prompt_budget_hint: 10
        pass_rate: 0.373
        cost_per_task: 0.174
"""


def test_from_yaml_string_default_knee():
    p = CodeCapsulesPolicy.from_yaml_string(_SAMPLE_YAML)
    assert p.variant == "two_pass_critique"
    assert p.mode == "escalating"
    assert p.always_escalate is True
    assert p.workload_class == "hard_workload"
    assert p.tier == "sonnet"
    assert p.knee == "balanced"
    assert p.pass_rate == pytest.approx(0.467)


def test_from_yaml_string_selects_specific_knee():
    p = CodeCapsulesPolicy.from_yaml_string(_SAMPLE_YAML, knee="cost_min")
    assert p.variant == "signaled_budget"
    assert p.prompt_budget_hint == 10
    assert p.knee == "cost_min"


def test_from_yaml_string_legacy_config_alias():
    """Legacy 'config:' field is accepted as alias for 'variant:'."""
    text = """
deployment_defaults:
  hard_workload:
    sonnet:
      balanced:
        config: two_pass_critique
        mode: escalating
        turn_budget: 10
"""
    p = CodeCapsulesPolicy.from_yaml_string(text)
    assert p.variant == "two_pass_critique"


def test_from_yaml_string_missing_entry_raises():
    text = """
deployment_defaults:
  hard_workload:
    sonnet:
      knee:
        variant: signaled_budget
"""
    with pytest.raises(PolicyError, match="no deployment_defaults entry"):
        CodeCapsulesPolicy.from_yaml_string(text, knee="cost_min")


def test_from_yaml_string_unknown_field_raises():
    text = """
deployment_defaults:
  hard_workload:
    sonnet:
      balanced:
        variant: signaled_budget
        invented_field: 123
"""
    with pytest.raises(PolicyError, match="invented_field"):
        CodeCapsulesPolicy.from_yaml_string(text)


def test_to_yaml_entry_round_trip():
    original = CodeCapsulesPolicy(
        variant="two_pass_critique",
        mode="escalating",
        turn_budget=10,
        escalating_start_budget=10,
        escalating_target_budget=25,
        always_escalate=True,
        prompt_variant="two_pass_critique",
        pass_rate=0.467,
        cost_per_task=0.308,
    )
    entry = original.to_yaml_entry()
    # Cell coordinates are NOT in the entry — they're the keys above it.
    assert "workload_class" not in entry
    assert "tier" not in entry
    assert "knee" not in entry
    # Required fields always emitted.
    assert entry["variant"] == "two_pass_critique"
    assert entry["mode"] == "escalating"
    # Escalating cell emits its budgets.
    assert entry["escalating_target_budget"] == 25
    assert entry["always_escalate"] is True
    # Provenance fields emitted when set.
    assert entry["pass_rate"] == 0.467
    assert entry["cost_per_task"] == 0.308


def test_to_yaml_entry_omits_defaults():
    p = CodeCapsulesPolicy(variant="signaled_budget", mode="sequential")
    entry = p.to_yaml_entry()
    # Mode-conditional fields not emitted when mode != escalating.
    assert "escalating_target_budget" not in entry
    assert "always_escalate" not in entry
    # Default prompt_variant/quality_gate/cascade_trigger not emitted.
    assert "prompt_variant" not in entry
    assert "quality_gate" not in entry
    assert "cascade_trigger" not in entry


def test_yaml_file_round_trip(tmp_path):
    """Write a policy YAML and re-load it."""
    p1 = policy_for(knee="balanced")
    out = tmp_path / "policy.yaml"
    doc = {
        "deployment_defaults": {
            p1.workload_class: {p1.tier: {p1.knee: p1.to_yaml_entry()}}
        }
    }
    import yaml
    out.write_text(yaml.dump(doc, sort_keys=False))
    p2 = CodeCapsulesPolicy.from_yaml(
        out, workload=p1.workload_class, tier=p1.tier, knee=p1.knee
    )
    assert p2.variant == p1.variant
    assert p2.turn_budget == p1.turn_budget
    assert p2.prompt_variant == p1.prompt_variant
    assert p2.pass_rate == pytest.approx(p1.pass_rate)


# ── controller block (deployable composition) ─────────────────────────────

def test_controller_none_by_default():
    p = CodeCapsulesPolicy()
    assert p.controller is None
    assert p.runner_policy() is None


def test_controller_block_materialises_runner_policy():
    p = CodeCapsulesPolicy(controller={
        "configs": ["floor", "siginject"],
        "tiers": ["sonnet"],
        "min_samples": 2,
        "governor": "hybrid_regok",
        "ship_gate": "repro_or_regok",
    })
    rp = p.runner_policy()
    assert rp.configs == ("floor", "siginject")
    assert rp.tiers == ("sonnet",)
    assert rp.governor == "hybrid_regok"


def test_controller_escalation_ladder_maps_to_gate():
    p = CodeCapsulesPolicy(controller={
        "configs": ["floor"], "tiers": ["sonnet"],
        "escalation": {"mode": "gate",
                       "ladder": [{"tier": "opus", "configs": ["floor"],
                                   "trigger": "regok_true"}]},
    })
    assert p.runner_policy().escalate_on_agreement is True


def test_controller_must_be_dict():
    with pytest.raises(ValueError, match="controller must be a dict"):
        CodeCapsulesPolicy(controller=["not", "a", "dict"])


def test_controller_bad_governor_raises_at_construction():
    # A bad value in the controller block fails when the policy is built,
    # not later at run time.
    with pytest.raises(ValueError, match="governor"):
        CodeCapsulesPolicy(controller={"governor": "bogus"})


# ── unbounded_budget variant alias ─────────────────────────────────────────

def test_unbounded_budget_is_a_valid_variant():
    p = CodeCapsulesPolicy(variant="unbounded_budget", mode="sequential", turn_budget=100)
    assert p.variant == "unbounded_budget"


@pytest.mark.parametrize("tier,knee", [("sonnet", "quality"), ("opus", "ceiling")])
def test_unbounded_budget_used_in_shipped_menu(tier, knee):
    # Paper Table 1 names the b=100 quality/ceiling cells "unbounded budget".
    assert policy_for(workload="hard_workload", tier=tier, knee=knee).variant == "unbounded_budget"


# ── to_variant_config: single-variant policy -> runnable VariantConfig ──────

def test_to_variant_config_resolves_framing_aliases_to_executors():
    from code_capsules.api import registry as R
    registered = set(R.list_registered("variant"))
    # implicit/unbounded budget -> the signaled_budget executor, no in-prompt hint
    for framing in ("implicit_budget", "unbounded_budget"):
        cfg = CodeCapsulesPolicy(variant=framing, mode="sequential", turn_budget=100).to_variant_config()
        assert cfg.name == "signaled_budget"
        assert cfg.name in registered          # the runner can resolve it
        assert cfg.prompt_budget_hint is None   # implicit framing
    # tool_alloc -> per_tool_class_hint executor
    cfg = CodeCapsulesPolicy(variant="tool_alloc").to_variant_config()
    assert cfg.name == "per_tool_class_hint"
    assert cfg.name in registered


def test_to_variant_config_passes_registered_variant_through_with_budgets():
    p = CodeCapsulesPolicy(
        variant="two_pass_critique", mode="escalating",
        escalating_start_budget=10, escalating_target_budget=25,
        always_escalate=True, prompt_variant="two_pass_critique",
    )
    cfg = p.to_variant_config()
    assert cfg.name == "two_pass_critique"
    assert cfg.mode == "escalating"
    assert cfg.escalating_start_budget == 10
    assert cfg.escalating_target_budget == 25
    assert cfg.always_escalate is True
    assert cfg.prompt_variant == "two_pass_critique"


def test_to_variant_config_name_is_always_a_registered_executor():
    from code_capsules.api import registry as R
    registered = set(R.list_registered("variant"))
    from code_capsules.controller.policy import _VALID_VARIANTS
    for v in _VALID_VARIANTS:
        name = CodeCapsulesPolicy(variant=v, turn_budget=100).to_variant_config().name
        assert name in registered, f"{v} -> {name} is not a runnable variant"


# ── _from_entry provenance allow-list ──────────────────────────────────────

def _wrap(entry_fields: str) -> str:
    return (
        "deployment_defaults:\n"
        "  hard_workload:\n"
        "    sonnet:\n"
        "      balanced:\n" + entry_fields
    )


def test_from_entry_tolerates_provenance_and_harness_fields():
    # model/source/note/preselect_top_n/force_stage2/escalating_stage2_mode are
    # not typed policy fields but must not blow up the loader.
    text = _wrap(
        "        variant: two_pass_critique\n"
        "        mode: escalating\n"
        "        model: sonnet\n"
        "        source: exp5\n"
        "        note: a note\n"
        "        preselect_top_n: 10\n"
        "        force_stage2: true\n"
        "        escalating_stage2_mode: enriched\n"
        "        pass_rate: 0.44\n"
        "        cost_per_task: 0.41\n"
    )
    p = CodeCapsulesPolicy.from_yaml_string(text)
    assert p.variant == "two_pass_critique"
    assert p.pass_rate == pytest.approx(0.44)


def test_from_entry_still_rejects_genuinely_unknown_field():
    text = _wrap(
        "        variant: signaled_budget\n"
        "        not_a_real_field: 123\n"
    )
    with pytest.raises(PolicyError, match="not_a_real_field"):
        CodeCapsulesPolicy.from_yaml_string(text)


# ── Top-level re-exports ─────────────────────────────────────────────────

def test_top_level_imports():
    import code_capsules
    assert "CodeCapsulesPolicy" in code_capsules.__all__
    assert "policy_for" in code_capsules.__all__
    assert "SHIPPED_PRESETS" in code_capsules.__all__
    assert code_capsules.CodeCapsulesPolicy is CodeCapsulesPolicy
    assert code_capsules.policy_for is policy_for

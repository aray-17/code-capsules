"""Unit tests for controller/routing_strategies.py (RoutingStrategy impls + build_routing)."""
import pytest

from code_capsules.api import VariantConfig
from code_capsules.controller.routing_config import RoutingConfig
from code_capsules.controller.routing_strategies import (
    RuleBasedRouter,
    CascadeRouter,
    DefaultRouter,
    build_routing,
)


def _avail(*names) -> dict:
    return {n: VariantConfig(name=n, mode="sequential") for n in names}


class TestRuleBasedRouter:
    def test_first_matching_rule_wins(self):
        r = RuleBasedRouter(
            rules=[{"when": "scientific_workload", "use": "stuck_signal_injection"}],
            default="two_pass_critique",
        )
        avail = _avail("stuck_signal_injection", "two_pass_critique")
        assert r.route("scientific_workload", avail).name == "stuck_signal_injection"

    def test_falls_back_to_default_when_no_rule_matches(self):
        r = RuleBasedRouter(rules=[], default="two_pass_critique")
        assert r.route("anything", _avail("two_pass_critique")).name == "two_pass_critique"


class TestCascadeRouter:
    def test_zero_arg_constructible_with_default_tiers(self):
        # Registry ships this as the "cascade" routing strategy (paper Table 2).
        cr = CascadeRouter()
        assert cr.name == "cascade"
        cfg = cr.route("x", _avail("plan_then_execute", "two_pass_critique"))
        assert cfg.mode == "cascade"
        assert cfg.extra["tiers"] == ["plan_then_execute", "two_pass_critique"]

    def test_missing_tier_config_raises(self):
        with pytest.raises(ValueError, match="cascade tiers"):
            CascadeRouter(tiers=["nope"]).route("x", _avail("plan_then_execute"))


class TestDefaultRouter:
    def test_returns_fixed_config(self):
        assert DefaultRouter(config_name="two_pass_critique").route(
            "any", _avail("two_pass_critique")).name == "two_pass_critique"


class TestBuildRouting:
    def test_rule_based_injects_routes_and_default(self):
        cfg = RoutingConfig(
            classifier="yaml_mapping",
            routing_strategy="rule_based",
            routes=[{"when": "scientific_workload", "use": "stuck_signal_injection"}],
            route_default="two_pass_critique",
        )
        clf, strat = build_routing(cfg)
        assert clf.name == "yaml_mapping"
        assert strat.name == "rule_based"
        avail = _avail("stuck_signal_injection", "two_pass_critique")
        # configured route fires; unmatched falls back to the configured default
        assert strat.route("scientific_workload", avail).name == "stuck_signal_injection"
        assert strat.route("hard_workload", avail).name == "two_pass_critique"

    def test_none_classifier_returns_none(self):
        clf, strat = build_routing(RoutingConfig())   # no classifier configured
        assert clf is None
        assert strat.name == "rule_based"

    def test_non_rule_based_resolves_from_registry(self):
        clf, strat = build_routing(RoutingConfig(routing_strategy="cascade"))
        assert strat.name == "cascade"

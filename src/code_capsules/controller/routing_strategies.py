"""
Built-in RoutingStrategy implementations.

Three shipped defaults:

- RuleBasedRouter: reads policy.yaml routes; returns the first matching
  variant config. Default for most deployments.

- CascadeRouter: returns a tiered VariantConfig that wraps a primary
  + escalation tier. Caller's runtime executes tiers in order, calling
  the configured CascadeTrigger between tiers. (The cascade orchestration
  itself lives in the variant — this router just selects which cascade
  config to use for the class label.)

- DefaultRouter: trivial; always returns a hardcoded fallback config.
  Useful for instrumentation runs or when policy.yaml isn't available.

User-defined strategies conform to the RoutingStrategy Protocol.
Domain knowledge (specific class labels) stays in policy.yaml routes,
NOT in this module.
"""
from __future__ import annotations

from typing import Optional

from code_capsules.api import VariantConfig


class RuleBasedRouter:
    """First-match routing from a list of rules.

    Each rule: {when: class_label, use: config_name}. Plus a default.
    Picks the first rule where the input class_label matches; falls back
    to default if no rule matches.

    Rules typically come from policy.yaml routing section:

        routing:
          classifier: generic_prompt
          routes:
            - when: find_named
              use: plan_then_execute
            - when: scientific_workload
              use: stuck_signal_injection
          default: two_pass_critique
    """

    name = "rule_based"

    def __init__(self,
                 rules: Optional[list[dict]] = None,
                 default: str = "two_pass_critique"):
        self.rules = rules or []
        self.default = default

    def route(self,
              class_label: str,
              available: dict[str, VariantConfig]) -> VariantConfig:
        """Walk the rules in order; return the matching variant config."""
        for rule in self.rules:
            if rule.get("when") == class_label:
                use = rule.get("use")
                if use in available:
                    return available[use]
        # Fallback
        if self.default in available:
            return available[self.default]
        # Last resort: first available config
        if available:
            first = next(iter(available.values()))
            return first
        raise ValueError(f"no variants available; rules={self.rules}, default={self.default}")


class CascadeRouter:
    """Tiered routing: returns a config that wraps a primary + escalation tier.

    Used for Flavor B adaptive composability (try cheap variant first,
    escalate to stronger variant if signals indicate stuck-ness). The
    cascade orchestration itself runs inside the variant runner — this
    router just packages the tiered config.

    Example config:

        tiers: [signaled_budget, plan_then_execute, two_pass_critique]
        trigger: heuristic    # CascadeTrigger name from registry
    """

    name = "cascade"

    def __init__(self,
                 tiers: Optional[list[str]] = None,
                 trigger_name: str = "heuristic"):
        self.tiers = tiers or ["plan_then_execute", "two_pass_critique"]
        self.trigger_name = trigger_name

    def route(self,
              class_label: str,
              available: dict[str, VariantConfig]) -> VariantConfig:
        """Return a cascade VariantConfig wrapping the tier sequence.

        The wrapped config uses mode='cascade' with the tier names in
        extra['tiers'] and the trigger name in extra['cascade_trigger'].
        The caller's runtime reads these and executes the tier sequence,
        consulting the named CascadeTrigger between tiers.
        """
        # Validate that all tier configs exist
        missing = [t for t in self.tiers if t not in available]
        if missing:
            raise ValueError(f"cascade tiers not in available variants: {missing}")
        return VariantConfig(
            name=f"cascade[{','.join(self.tiers)}]",
            mode="cascade",
            extra={
                "tiers": self.tiers,
                "cascade_trigger": self.trigger_name,
            },
        )


class DefaultRouter:
    """Trivial: always returns a fixed config name from `available`.

    Useful for instrumentation runs where you want to bypass routing logic
    and run a specific variant on everything.
    """

    name = "default"

    def __init__(self, config_name: str = "two_pass_critique"):
        self.config_name = config_name

    def route(self,
              class_label: str,
              available: dict[str, VariantConfig]) -> VariantConfig:
        if self.config_name not in available:
            raise ValueError(f"config '{self.config_name}' not in available: {sorted(available)}")
        return available[self.config_name]


def build_routing(cfg):
    """Materialise the (classifier, routing strategy) configured in a RoutingConfig.

    Resolves ``cfg.classifier`` and ``cfg.routing_strategy`` from the registry
    and, for the rule-based strategy, injects ``cfg.routes`` / ``cfg.route_default``
    so the YAML routing block is load-bearing (paper §"Routing": the routing block
    configures the workload classifier and routing strategy). Returns
    ``(classifier_or_None, strategy)``; a None classifier means "use the framework
    default class label". Lazy registry import avoids an import cycle.

    Example:
        from code_capsules.controller.yaml_dsl import load_policy
        from code_capsules.controller.routing_strategies import build_routing
        clf, strat = build_routing(load_policy("policy.yaml"))
        label  = clf.classify(task) if clf else ""
        config = strat.route(label, available_variant_configs)
    """
    from code_capsules.api import registry as _registry

    classifier = (
        _registry.get("classifier", cfg.classifier) if getattr(cfg, "classifier", None)
        else None
    )
    name = getattr(cfg, "routing_strategy", "rule_based") or "rule_based"
    routes = getattr(cfg, "routes", None) or []
    default = getattr(cfg, "route_default", None)
    if name == "rule_based":
        # Build a configured RuleBasedRouter from the YAML routes/default rather
        # than the empty-rules registry singleton.
        strategy = RuleBasedRouter(
            rules=[dict(x) for x in routes],
            default=default or "two_pass_critique",
        )
    else:
        strategy = _registry.get("routing_strategy", name)
    return classifier, strategy


__all__ = ["RuleBasedRouter", "CascadeRouter", "DefaultRouter", "build_routing"]

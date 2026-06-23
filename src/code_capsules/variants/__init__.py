"""
Concrete Variant implementations shipped with Code-Capsules.

Seven variants, each conforming to the Variant Protocol from code_capsules.api.
They cover the three orchestration modes (sequential, escalating, injection_loop)
in the configurations validated cross-tier.

The package's `register_builtins()` registers all seven under their canonical
names so policy.yaml routes can reference them by string. The registry's lazy
init calls this on first access.

Cross-tier deployment guidance lives in policy.yaml `deployment_defaults`,
calibrated from the experimental results.
"""
from __future__ import annotations

from code_capsules.variants.signaled_budget import SignaledBudgetVariant
from code_capsules.variants.plan_then_execute import PlanThenExecuteVariant
from code_capsules.variants.per_tool_class_hint import PerToolClassHintVariant
from code_capsules.variants.phase_staged import PhaseStagedVariant
from code_capsules.variants.two_pass_critique import TwoPassCritiqueVariant
from code_capsules.variants.stuck_signal_injection import StuckSignalInjectionVariant
from code_capsules.variants.relevance_ranker import RelevanceRankerWrapper


def register_builtins() -> None:
    """Register all shipped variants in the controller.registry."""
    from code_capsules.api.registry import register
    register("variant", SignaledBudgetVariant())
    register("variant", PlanThenExecuteVariant())
    register("variant", PerToolClassHintVariant())
    register("variant", PhaseStagedVariant())
    register("variant", TwoPassCritiqueVariant())
    register("variant", StuckSignalInjectionVariant())
    register("variant", RelevanceRankerWrapper())


__all__ = [
    "SignaledBudgetVariant",
    "PlanThenExecuteVariant",
    "PerToolClassHintVariant",
    "PhaseStagedVariant",
    "TwoPassCritiqueVariant",
    "StuckSignalInjectionVariant",
    "RelevanceRankerWrapper",
    "register_builtins",
]

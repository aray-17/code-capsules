"""
Registry: name → concrete implementation for each extension primitive.

The runtime + policy.yaml reference implementations by string name. Users
register their own implementations by calling `register(...)` at import
time (typically via plugin entry points or an explicit registration call).

Built-in defaults are registered by `register_builtins()`, called lazily
on first registry access. This avoids circular imports during framework
bootstrap.

Lookup is plain dict access — no magic, no proxying. If a name isn't
registered, KeyError. Users wire their plugin module's `register(...)`
calls into their `policy.yaml` consumer.
"""
from __future__ import annotations

from typing import Any

from code_capsules.api import (
    WorkloadClassifier,
    RoutingStrategy,
    Variant,
    Signal,
    QualityGate,
    CascadeTrigger,
    CostModel,
    ModelClient,
)


# Per-primitive registries. Keyed by short name (matches policy.yaml `config:` fields).
_classifiers: dict[str, WorkloadClassifier] = {}
_routing_strategies: dict[str, RoutingStrategy] = {}
_variants: dict[str, Variant] = {}
_signals: dict[str, Signal] = {}
_quality_gates: dict[str, QualityGate] = {}
_cascade_triggers: dict[str, CascadeTrigger] = {}
_cost_models: dict[str, CostModel] = {}
_model_clients: dict[str, ModelClient] = {}

_REGISTRIES: dict[str, dict] = {
    "classifier": _classifiers,
    "routing_strategy": _routing_strategies,
    "variant": _variants,
    "signal": _signals,
    "quality_gate": _quality_gates,
    "cascade_trigger": _cascade_triggers,
    "cost_model": _cost_models,
    "model_client": _model_clients,
}

_builtins_registered = False


def register(kind: str, impl: Any) -> None:
    """Register an implementation. Looks up by `impl.name` attribute.

    kind: one of {classifier, routing_strategy, variant, signal, quality_gate,
                  cascade_trigger, cost_model}
    """
    if kind not in _REGISTRIES:
        raise ValueError(f"unknown kind: {kind!r}; valid: {sorted(_REGISTRIES)}")
    name = getattr(impl, "name", None)
    if not name:
        raise ValueError(f"implementation has no `name` attribute: {impl!r}")
    _REGISTRIES[kind][name] = impl


def get(kind: str, name: str) -> Any:
    """Look up a registered implementation by kind + name. Registers builtins lazily."""
    global _builtins_registered
    if not _builtins_registered:
        _register_builtins()
        _builtins_registered = True
    if kind not in _REGISTRIES:
        raise ValueError(f"unknown kind: {kind!r}; valid: {sorted(_REGISTRIES)}")
    reg = _REGISTRIES[kind]
    if name not in reg:
        raise KeyError(f"{kind!r}: no implementation named {name!r}; "
                       f"registered: {sorted(reg)}")
    return reg[name]


def list_registered(kind: str) -> list[str]:
    """Names of all registered implementations for a kind."""
    global _builtins_registered
    if not _builtins_registered:
        _register_builtins()
        _builtins_registered = True
    return sorted(_REGISTRIES[kind])


def _register_builtins() -> None:
    """Register the framework's shipped default implementations.

    Imports inline to avoid circular dependencies during module load.
    """
    # Classifiers
    from code_capsules.controller.classifiers import (
        GenericPromptClassifier, YamlMappingClassifier, BenchmarkSaturatedClassifier,
    )
    register("classifier", GenericPromptClassifier())
    register("classifier", BenchmarkSaturatedClassifier())
    # Default YamlMappingClassifier loads the shipped SWE-bench repo→class table
    # and registers under the literal name "yaml_mapping"; a user with another
    # domain constructs YamlMappingClassifier(path) and registers their own.
    register("classifier", YamlMappingClassifier())

    # Cost models — one per vendor at public-pricing rates
    from code_capsules.core.cost_models import (
        AnthropicPublicPricing, OpenAIPublicPricing, GeminiPublicPricing,
    )
    register("cost_model", AnthropicPublicPricing())
    register("cost_model", OpenAIPublicPricing())
    register("cost_model", GeminiPublicPricing())

    # Signals
    from code_capsules.runtime.signals import (
        CapPressure, FileThrash, TestFailure, Traceback, PatchAttemptFailed,
    )
    register("signal", CapPressure())
    register("signal", FileThrash())
    register("signal", TestFailure())
    register("signal", Traceback())
    register("signal", PatchAttemptFailed())

    # Cascade triggers
    from code_capsules.controller.cascade_triggers import (
        HeuristicCascade, AlwaysEscalate, NeverEscalate, VerifierGate,
        CrossSampleAgreementCascade,
    )
    register("cascade_trigger", HeuristicCascade())
    register("cascade_trigger", AlwaysEscalate())
    register("cascade_trigger", NeverEscalate())
    register("cascade_trigger", VerifierGate())
    # The regression-gated hybrid governor: the paper's shipped "diverse_agreement"
    # default. register() keys on .name ("diverse_agreement"); also expose the
    # back-compat alias ("cross_sample_agreement").
    _diverse = CrossSampleAgreementCascade()
    register("cascade_trigger", _diverse)
    for _alias in getattr(_diverse, "aliases", ()):
        _cascade_triggers[_alias] = _diverse

    # Quality gates (adaptors over the existing quality_gate.py impls)
    from code_capsules.evaluation.quality_gates_adaptors import (
        PythonAstGate, BinaryTestGate, AlwaysPassGate, DockerEvalGate,
    )
    register("quality_gate", PythonAstGate())
    register("quality_gate", BinaryTestGate())
    register("quality_gate", AlwaysPassGate())
    # DockerEvalGate ships present-by-name: with no injected evaluator it binds the
    # optional SWE-bench docker_eval lazily on first use; a deployer with their own
    # evaluator re-registers under the same "docker_eval" key (or injects via
    # evaluation.swe_bench_adapter.make_docker_eval_gate).
    register("quality_gate", DockerEvalGate())

    # Routing strategies
    from code_capsules.controller.routing_strategies import (
        RuleBasedRouter, DefaultRouter, CascadeRouter,
    )
    register("routing_strategy", RuleBasedRouter())  # empty rules; user reconfigures
    register("routing_strategy", DefaultRouter())
    register("routing_strategy", CascadeRouter())    # default tier sequence; user reconfigures

    # Model clients (must register before variants — variants resolve client by name)
    from code_capsules.adapters import register_builtins as _register_client_builtins
    _register_client_builtins()

    # Variants
    from code_capsules.variants import register_builtins as _register_variant_builtins
    _register_variant_builtins()


__all__ = ["register", "get", "list_registered"]

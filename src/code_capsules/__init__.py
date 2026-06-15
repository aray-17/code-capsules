"""Code-Capsules: calibrated cost-quality configuration for coding agents.

Top-level re-exports follow the Agent Capsules pattern: a small surface of
load-bearing primitives importable directly from the package root, with
deeper API available under explicit submodule paths.

Usage (the deployable lever -- the validated cost/quality controller):

    from code_capsules import CodeCapsulesRunner

    runner = CodeCapsulesRunner.from_policy_file("policy.yaml")
    run = runner.run(sampler, grade_fn)   # diverse solver pair -> verifier-select -> governor-abandon

ONE public runner, behavior = policy. To run a SINGLE variant, use a one-config
policy + ``make_variant_sampler`` (it turns a registered Variant into the
sampler/grade_fn the runner drives and captures the RunResult for logging):

    from code_capsules import CodeCapsulesRunner, RunnerPolicy, make_variant_sampler
    sampler, grade_fn, captured = make_variant_sampler(task, config, gate=gate)
    CodeCapsulesRunner(RunnerPolicy(configs=(config.name,), tiers=("default",))).run(sampler, grade_fn)
    result = captured["result"]

The optional per-task ``classify -> route`` step (for heterogeneous workloads) is
``code_capsules.controller.routing.route_to_config`` -- inert on monomorphic
benchmarks, so it is not needed by default.

The eight extension primitives (WorkloadClassifier, RoutingStrategy,
Variant, Signal, QualityGate, CascadeTrigger, ModelClient, CostModel)
are Protocols defined in ``code_capsules.api`` and re-exported here for
top-level access.
"""

# Core runtime entry point: the deployable VOI lever (diverse-sample solver pair ->
# verifier-select -> governor-abandon). This is the validated cost/quality controller
# AND the one public runner -- a single variant is just a one-config policy on it.
from code_capsules.controller.runtime import (
    CodeCapsulesRunner, RunnerPolicy, make_variant_sampler,
)

# Task / result dataclasses.
from code_capsules.api import (
    TaskDescriptor,
    VariantConfig,
    SessionState,
    Attempt,
    InvocationResult,
    RunResult,
)

# Eight extension primitives (Protocols).
from code_capsules.api import (
    WorkloadClassifier,
    RoutingStrategy,
    Variant,
    Signal,
    QualityGate,
    CascadeTrigger,
    ModelClient,
    CostModel,
)

# Shipped CostModel implementations.
from code_capsules.core.cost_models import (
    AnthropicPublicPricing,
    OpenAIPublicPricing,
    GeminiPublicPricing,
)

# Policy DSL: declarative per-cell policy + shipped preset selector.
from code_capsules.controller.policy import (
    CodeCapsulesPolicy,
    SHIPPED_PRESETS,
    policy_for,
)

__version__ = "0.1.0"

__all__ = [
    # The one public runner: the deployable lever / cost-quality controller.
    # A single variant is a one-config policy on it (see make_variant_sampler).
    "CodeCapsulesRunner",
    "RunnerPolicy",
    "make_variant_sampler",
    # Dataclasses.
    "TaskDescriptor",
    "VariantConfig",
    "SessionState",
    "Attempt",
    "InvocationResult",
    "RunResult",
    # Extension primitives (Protocols).
    "WorkloadClassifier",
    "RoutingStrategy",
    "Variant",
    "Signal",
    "QualityGate",
    "CascadeTrigger",
    "ModelClient",
    "CostModel",
    # Shipped CostModel implementations.
    "AnthropicPublicPricing",
    "OpenAIPublicPricing",
    "GeminiPublicPricing",
    # Policy DSL.
    "CodeCapsulesPolicy",
    "policy_for",
    "SHIPPED_PRESETS",
]

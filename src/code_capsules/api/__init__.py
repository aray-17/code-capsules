"""
Public API surface — re-exports of the typed dataclasses and the eight
extension Protocols.

The framework's public surface is two small modules:
  - :mod:`code_capsules.api.types`     — dataclasses (TaskDescriptor,
    VariantConfig, SessionState, Attempt, InvocationResult, RunResult)
  - :mod:`code_capsules.api.protocols` — eight Protocols (WorkloadClassifier,
    RoutingStrategy, Variant, Signal, QualityGate, CascadeTrigger,
    ModelClient, CostModel)

This package's ``__init__`` re-exports both for convenience: callers can
``from code_capsules.api import TaskDescriptor, ModelClient`` without
caring about the internal file split.
"""
from code_capsules.api.types import (
    TaskDescriptor,
    VariantConfig,
    SessionState,
    Attempt,
    InvocationResult,
    RunResult,
)

from code_capsules.api.protocols import (
    WorkloadClassifier,
    RoutingStrategy,
    Variant,
    Signal,
    QualityGate,
    CascadeTrigger,
    ModelClient,
    CostModel,
)

__all__ = [
    # Dataclasses (types).
    "TaskDescriptor",
    "VariantConfig",
    "SessionState",
    "Attempt",
    "InvocationResult",
    "RunResult",
    # Eight extension Protocols.
    "WorkloadClassifier",
    "RoutingStrategy",
    "Variant",
    "Signal",
    "QualityGate",
    "CascadeTrigger",
    "ModelClient",
    "CostModel",
]

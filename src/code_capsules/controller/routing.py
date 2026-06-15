"""controller/routing.py — the optional classify→route helper.

This is the *per-task variant selection* step that the deleted VariantPipelineRunner
used to own inline. It is now a small, OPTIONAL function decoupled from the runner:
the one public runner (CodeCapsulesRunner) is a domain-agnostic control loop over a
fixed policy; choosing WHICH config to run for a given task is a separate, pluggable
concern that most callers do not need.

When you DON'T need it (the common case, and every benchmark here): put the config(s)
you want directly in the policy and pass a sampler that runs them. SWE-bench/HE/MBPP
are monomorphic workloads -- one calibrated config covers the whole set -- so routing
is inert and we skip it entirely.

When you DO need it (a heterogeneous production workload): call route_to_config() to
map a task -> a VariantConfig via your WorkloadClassifier + RoutingStrategy, then build
a one-config policy + sampler from the result. Both primitives stay user-pluggable
(they are part of the extension surface); this helper just composes them.

    from code_capsules.controller.routing import route_to_config
    cfg = route_to_config(task, classifier=clf, routing_strategy=router,
                          variant_configs=configs)
    # ... build a sampler that runs cfg, then CodeCapsulesRunner(...).run(sampler, grade_fn)
"""
from __future__ import annotations

from code_capsules.api import (
    RoutingStrategy, TaskDescriptor, VariantConfig, WorkloadClassifier,
)


def route_to_config(
    task: TaskDescriptor,
    *,
    classifier: WorkloadClassifier,
    routing_strategy: RoutingStrategy,
    variant_configs: dict[str, VariantConfig],
) -> VariantConfig:
    """classify(task) -> class_label -> route(label, configs) -> VariantConfig.

    The optional per-task selection step. Returns the VariantConfig the routing
    strategy picks for the task's class. Inactive on monomorphic benchmarks (the
    classifier returns a single saturated label, so routing always picks the one
    configured variant); load-bearing only for heterogeneous production workloads.
    """
    class_label = classifier.classify(task) or ""
    return routing_strategy.route(class_label, variant_configs)


__all__ = ["route_to_config"]

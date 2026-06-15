"""
Workload routing: classify a task, route it to a shipped variant.

The `routing:` block of policy.yaml configures two extension primitives by name:
a WorkloadClassifier (which class is this task?) and a RoutingStrategy (which
variant for that class?). `build_routing` resolves both from the registry and
injects the YAML routes, so the per-class menu is load-bearing config, not code.

This is the `classify -> route` step that precedes running a single variant
(see examples/quickstart_policy.py for the variant/lever side). It is optional
and inert on monomorphic workloads (one config covers the whole set), so the
benchmarks skip it; it earns its keep on heterogeneous workloads.

All offline -- no API keys.

Run:
    python -m examples.workload_routing
"""

from __future__ import annotations

from pathlib import Path

from code_capsules.api import TaskDescriptor, VariantConfig
from code_capsules.controller.yaml_dsl import load_policy
from code_capsules.controller.routing_strategies import build_routing

_POLICY_YAML = str(Path(__file__).parent.parent / "policy.yaml")

# The variants a deployment has wired up (name -> its run config). In a real
# deployment these come from the deployment_defaults menu; here we stub the
# three the shipped routing block can select between.
AVAILABLE = {
    name: VariantConfig(name=name, mode="sequential")
    for name in ("stuck_signal_injection", "two_pass_critique", "relevance_ranker")
}

# A few tasks keyed by repo. The shipped classifier (yaml_mapping) reads
# context["repo"] (the org prefix of a SWE-bench instance_id) and maps it to a
# workload class via code_capsules/config/swe_bench_repo_classes.yaml.
TASKS = [
    ("sympy",       "simplify() drops a symbolic assumption on Piecewise"),
    ("scikit-learn", "OneHotEncoder raises on unseen categories with handle_unknown"),
    ("matplotlib",  "tight_layout warns when an axes has a twin"),
    ("django",      "QuerySet.bulk_create ignores update_conflicts on SQLite"),
]


def main() -> None:
    # 1) Load the policy and materialise the configured (classifier, strategy).
    cfg = load_policy(_POLICY_YAML)
    classifier, strategy = build_routing(cfg)

    print("Routing configuration (from policy.yaml `routing:` block)")
    print(f"  classifier      : {classifier.name if classifier else '(none)'}")
    print(f"  routing_strategy: {strategy.name}")
    print(f"  routes          : {cfg.routes}")
    print(f"  default         : {cfg.route_default}")
    print()

    # 2) For each task: classify -> route -> the variant that would run.
    print(f"{'repo':<13}{'class':<24}{'-> variant'}")
    print("-" * 60)
    for repo, problem in TASKS:
        task = TaskDescriptor(text=problem, context={"repo": repo, "task_id": repo})
        label = classifier.classify(task) if classifier else ""
        chosen = strategy.route(label, AVAILABLE)
        print(f"{repo:<13}{label:<24}-> {chosen.name}")

    print()
    print("scientific repos route to the per-class variant from the routes table;")
    print("everything else falls back to the configured default. From here, hand the")
    print("chosen VariantConfig to make_variant_sampler(task, cfg, ...) and run it")
    print("(see examples/quickstart_policy.py / examples/deployable_lever.py).")


if __name__ == "__main__":
    main()

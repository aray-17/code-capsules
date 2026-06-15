"""
Advanced: register a custom QualityGate and select it by name.

The :class:`~code_capsules.api.QualityGate` Protocol is one of the eight
extension primitives. The shipped implementations (``docker_eval``,
``binary_tests``, ``python_ast``, ``always_pass``) cover the benchmarks in the
paper, but production deployments typically need a custom gate that matches
their CI's test runner, their internal lint rules, or their domain success
criteria.

This example defines a custom gate, registers it through the framework's
one-line registry call, confirms it resolves by name and conforms to the
Protocol, and shows it is then selectable from a typed ``CodeCapsulesPolicy``
and splice-able into the runner via ``make_variant_sampler``.

Runs offline, no external tools.

Run:
    python -m examples.advanced.custom_quality_gate
"""

from __future__ import annotations

from code_capsules.api import Attempt, QualityGate
from code_capsules.api.registry import register, get


# ---------------------------------------------------------------------------
# A custom QualityGate
# ---------------------------------------------------------------------------
#
# The Protocol defines a single method: ``check(self, attempt) -> bool``
# (True iff the attempt passes the gate). The framework does not care whether
# the gate shells out to Docker, calls a remote CI, or evaluates a rule;
# anything with a ``name`` and a conforming ``check`` is a QualityGate.

class CIStyleQualityGate:
    """Composite gate mirroring a CI pipeline: patch-applies + tests-pass + style-clean.

    A real implementation would shell out to ``git apply --check``, ``pytest``,
    and ``mypy``; this one reads scripted flags off ``attempt.metadata`` so the
    example runs offline.
    """

    name = "ci_style"

    def __init__(self, require_tests: bool = True, require_style: bool = True):
        self.require_tests = require_tests
        self.require_style = require_style

    def check(self, attempt: Attempt) -> bool:
        meta = attempt.metadata
        patch_ok = bool(attempt.patch) and attempt.patch.startswith("diff --git")
        tests_ok = meta.get("tests_pass", True) if self.require_tests else True
        style_ok = meta.get("style_clean", True) if self.require_style else True
        return patch_ok and tests_ok and style_ok


def main() -> None:
    # 1) Register the custom gate -- the one-line extension call.
    register("quality_gate", CIStyleQualityGate())

    # 2) It resolves by name and conforms to the Protocol.
    gate = get("quality_gate", "ci_style")
    assert isinstance(gate, QualityGate), "must conform to the QualityGate Protocol"
    print(f"registered + resolved: {gate.name!r}; isinstance(QualityGate)=True\n")

    # 3) Run it on a few scripted attempts (offline).
    cases = [
        ("clean attempt", Attempt(patch="diff --git a/x b/x\n+1\n")),
        ("no patch",      Attempt(patch="")),
        ("tests fail",    Attempt(patch="diff --git a/x b/x\n", metadata={"tests_pass": False})),
        ("style fail",    Attempt(patch="diff --git a/x b/x\n", metadata={"style_clean": False})),
    ]
    for label, attempt in cases:
        flag = "PASS" if gate.check(attempt) else "FAIL"
        print(f"  [{flag}] {label}")

    # 4) Once registered, the gate is selectable BY NAME from a typed policy...
    from code_capsules import CodeCapsulesPolicy
    policy = CodeCapsulesPolicy(variant="signaled_budget", quality_gate="ci_style")
    print(f"\nselectable in CodeCapsulesPolicy: quality_gate={policy.quality_gate!r}")

    # 5) ...and splice-able into the runner via make_variant_sampler(gate=...),
    #    which registers the instance and wires it into the variant config:
    #
    #     from code_capsules import CodeCapsulesRunner, RunnerPolicy, make_variant_sampler
    #     from code_capsules.api import TaskDescriptor, VariantConfig
    #     cfg = VariantConfig(name="signaled_budget", mode="sequential", turn_budget=20)
    #     sampler, grade_fn, captured = make_variant_sampler(task, cfg, gate=CIStyleQualityGate())
    #     CodeCapsulesRunner(RunnerPolicy(configs=(cfg.name,), tiers=("default",))).run(sampler, grade_fn)
    #
    print("also wire-able via make_variant_sampler(task, cfg, gate=CIStyleQualityGate()).")


if __name__ == "__main__":
    main()

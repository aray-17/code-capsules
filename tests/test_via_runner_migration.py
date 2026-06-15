"""Convergence test: the single-variant path now runs as a ONE-CONFIG policy on the
ONE public runner (CodeCapsulesRunner) via the framework helper `make_variant_sampler`,
instead of the deleted VariantPipelineRunner.

Validates `controller.runtime.make_variant_sampler`: the sampler runs the variant ONCE
(registry path), captures the full RunResult for logging, and the grade_fn reads the
captured RunResult.resolved (no extra eval). This is exactly what the SWE-bench demo
harness + the cross-vendor examples now use. No docker. Run with
`PYTHONPATH=src:. python3.12 tests/test_via_runner_migration.py`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from code_capsules.api import RunResult, VariantConfig, TaskDescriptor  # noqa: E402
from code_capsules.api.registry import register  # noqa: E402
from code_capsules.controller.runtime import (  # noqa: E402
    CodeCapsulesRunner, RunnerPolicy, make_variant_sampler,
)
from code_capsules.controller.verifier import RESOLVED, FLAT, NOPATCH  # noqa: E402


class _FakeVariant:
    name = "fake_variant"

    def __init__(self, patch, resolved):
        self._patch, self._resolved = patch, resolved
        self.calls = 0

    def run(self, task, config):
        self.calls += 1
        return RunResult(task_id="t", resolved=self._resolved, patch=self._patch,
                         num_turns=7, cost_usd=0.42)


class _FakeGate:
    name = "fake_gate"


class _FakeCost:
    name = "fake_cost"


def _adapters(variant):
    register("variant", variant)
    cfg = VariantConfig(name="fake_variant", mode="sequential", turn_budget=10)
    task = TaskDescriptor(text="fix it")
    return make_variant_sampler(task, cfg, gate=_FakeGate(), cost_model=_FakeCost())


def test_resolved_variant_drives_to_RESOLVED_and_captures_result():
    v = _FakeVariant("the-patch", True)
    sampler, grade_fn, captured = _adapters(v)
    run = CodeCapsulesRunner(RunnerPolicy(configs=("fake_variant",), tiers=("default",))).run(sampler, grade_fn)
    assert run.outcome == "RESOLVED"
    assert v.calls == 1                                  # variant run exactly once
    assert captured["result"].cost_usd == 0.42          # telemetry captured for logging
    assert captured["result"].num_turns == 7


def test_unresolved_variant_is_EXHAUSTED_but_still_captured():
    v = _FakeVariant("a-failed-patch", False)
    sampler, grade_fn, captured = _adapters(v)
    run = CodeCapsulesRunner(RunnerPolicy(configs=("fake_variant",), tiers=("default",))).run(sampler, grade_fn)
    assert run.outcome == "EXHAUSTED"                    # K=1 -> no doom call -> not ABANDONED
    assert captured["result"] is not None               # patch still available to log
    assert captured["result"].patch == "a-failed-patch"
    assert run.selection is not None                     # selection carried on EXHAUSTED


def test_grade_fn_reads_captured_result_no_extra_eval():
    v = _FakeVariant("p", True)
    sampler, grade_fn, captured = _adapters(v)
    sampler("fake_variant", "default")                  # populate captured
    assert grade_fn("p") == RESOLVED                     # reads captured.resolved
    v2 = _FakeVariant("", False)
    s2, g2, cap2 = _adapters(v2)
    s2("fake_variant", "default")
    assert g2("") == NOPATCH                             # empty patch
    v3 = _FakeVariant("x", False)
    s3, g3, cap3 = _adapters(v3)
    s3("fake_variant", "default")
    assert g3("x") == FLAT                               # non-empty, unresolved


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"via-runner migration: {passed}/{len(fns)} pass")

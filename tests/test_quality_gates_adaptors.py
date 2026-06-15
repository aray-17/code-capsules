"""Unit tests for evaluation/quality_gates_adaptors.py (the Protocol QualityGate adaptors)."""
import pytest

from code_capsules.api import Attempt
from code_capsules.evaluation.quality_gates_adaptors import (
    AlwaysPassGate,
    DockerEvalGate,
)


def _attempt(patch="diff --git a b", **meta) -> Attempt:
    return Attempt(patch=patch, metadata=meta)


class TestAlwaysPassGate:
    def test_always_true(self):
        assert AlwaysPassGate().name == "always_pass"
        assert AlwaysPassGate().check(_attempt(patch="")) is True


class TestDockerEvalGate:
    def test_injected_evaluator_drives_check(self):
        calls = {}

        def fake_eval(patch, instance, timeout):
            calls["args"] = (patch, instance, timeout)
            return {"resolved": True}

        gate = DockerEvalGate(docker_eval_fn=fake_eval, timeout=42)
        assert gate.name == "docker_eval"
        assert gate.check(_attempt(patch="P", instance={"instance_id": "x"})) is True
        assert calls["args"] == ("P", {"instance_id": "x"}, 42)

    def test_injected_evaluator_unresolved(self):
        gate = DockerEvalGate(docker_eval_fn=lambda p, i, t: {"resolved": False})
        assert gate.check(_attempt()) is False

    def test_missing_resolved_key_is_false(self):
        gate = DockerEvalGate(docker_eval_fn=lambda p, i, t: {})
        assert gate.check(_attempt()) is False

    def test_zero_arg_is_present_by_name(self):
        # Registry ships docker_eval present-by-name with no injected evaluator;
        # construction must not require the SWE-bench harness.
        gate = DockerEvalGate()
        assert gate.name == "docker_eval"

    def test_lazy_resolution_uses_injected_first(self):
        # An injected fn is used directly and never triggers the lazy tools import.
        gate = DockerEvalGate(docker_eval_fn=lambda p, i, t: {"resolved": True})
        assert gate._resolve_fn() is not None
        assert gate.check(_attempt()) is True

"""
Built-in QualityGate adaptors conforming to the Protocol.

Wraps the existing controller.quality_gate gates (PythonQualityGate,
BinaryQualityGate) plus adds two new adaptors for SWE-bench-style and
trivial use:

- PythonAstGate: wraps the existing PythonQualityGate; passes iff
  patch is valid Python that survives compile + all tests pass
  (HumanEval / MBPP style).
- BinaryTestGate: wraps the existing BinaryQualityGate; passes iff
  every FAIL_TO_PASS test passes (SWE-bench resolved semantics).
- DockerEvalGate: thin adaptor calling out to an external docker_eval
  function (lazy import to avoid hard dependency at framework load).
- AlwaysPassGate: trivial; useful for instrumentation runs.

User-defined gates conform to the QualityGate Protocol from code_capsules.api.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from code_capsules.api import Attempt


class PythonAstGate:
    """Passes iff the patch parses as Python AND all tests pass.

    Wraps PythonQualityGate from code_capsules.evaluation.quality_gate. For HumanEval /
    MBPP style code-execution tasks where the patch is a complete function
    definition and tests are assert statements.

    Attempt.metadata may include:
      - tests: list[str] of executable assert statements
    """

    name = "python_ast"

    def __init__(self):
        from code_capsules.evaluation.quality_gate import PythonQualityGate
        self._inner = PythonQualityGate()

    def check(self, attempt: Attempt) -> bool:
        problem = {"tests": attempt.metadata.get("tests", [])}
        result = self._inner.score(attempt.patch, problem)
        # PythonQualityGate returns a score; treat ≥1.0 (all tests pass, lint clean) as PASS
        return result.test_pass_rate >= 1.0 and result.lint_clean


class BinaryTestGate:
    """Passes iff every test in attempt.metadata['tests'] passes.

    Wraps BinaryQualityGate from code_capsules.evaluation.quality_gate. Strict pass/fail
    semantics matching SWE-bench resolved.
    """

    name = "binary_tests"

    def __init__(self):
        from code_capsules.evaluation.quality_gate import BinaryQualityGate
        self._inner = BinaryQualityGate()

    def check(self, attempt: Attempt) -> bool:
        problem = {"tests": attempt.metadata.get("tests", [])}
        result = self._inner.score(attempt.patch, problem)
        return result.test_pass_rate >= 1.0


class DockerEvalGate:
    """Adaptor for external docker_eval functions (e.g. SWE-bench Docker).

    The docker_eval function is normally supplied at construction time
    (``make_docker_eval_gate`` / ``make_variant_sampler`` inject it), which
    keeps this module independent of any specific harness or container
    runtime. Constructed with no function, it resolves the canonical SWE-bench
    scorer shipped in the runtime
    (``code_capsules.evaluation.docker_eval.docker_eval``) lazily on first
    ``check`` - this is what lets the registry ship ``docker_eval`` as a
    name-selectable QualityGate default; a deployer with their own evaluator
    re-registers under the same ``docker_eval`` key.

    docker_eval signature expected:
      docker_eval(patch: str, instance: dict, timeout: int = 180) -> dict
        with key "resolved": bool|None
    """

    name = "docker_eval"

    def __init__(self,
                 docker_eval_fn: Optional[Callable[[str, dict, int], dict]] = None,
                 timeout: int = 180):
        self._fn = docker_eval_fn
        self._timeout = timeout

    def _resolve_fn(self) -> Callable[[str, dict, int], dict]:
        if self._fn is None:
            try:
                from code_capsules.evaluation.docker_eval import docker_eval
            except ImportError as exc:  # pragma: no cover - exercised only off-package
                raise RuntimeError(
                    "DockerEvalGate has no evaluator: construct it with an injected "
                    "docker_eval(patch, instance, timeout) via "
                    "code_capsules.evaluation.swe_bench_adapter.make_docker_eval_gate, "
                    "or register your own QualityGate under the 'docker_eval' name."
                ) from exc
            self._fn = lambda patch, instance, timeout: docker_eval(
                patch, instance, timeout=timeout)
        return self._fn

    def check(self, attempt: Attempt) -> bool:
        instance = attempt.metadata.get("instance", {})
        result = self._resolve_fn()(attempt.patch, instance, self._timeout)
        return bool(result.get("resolved"))


class AlwaysPassGate:
    """Trivial: always returns True. For instrumentation runs."""

    name = "always_pass"

    def check(self, attempt: Attempt) -> bool:
        return True


__all__ = [
    "PythonAstGate",
    "BinaryTestGate",
    "DockerEvalGate",
    "AlwaysPassGate",
]

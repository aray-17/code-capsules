"""
Pluggable quality gate for evaluating solution correctness.

BaseQualityGate is the abstract interface. Implementations:
  CodeQualityGate  — default: test_pass_rate + lint_weight × lint_clean
  BinaryQualityGate — strict pass/fail: 1.0 iff all tests pass, else 0.0

New domains (document analysis, SWE-bench binary resolved) subclass BaseQualityGate
and register via gate_from_config(). Weights come from RoutingConfig, not hardcoded.
"""
from __future__ import annotations

import ast
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code_capsules.controller.routing_config import RoutingConfig


@dataclass
class QualityResult:
    test_pass_rate: float   # fraction of tests that passed [0, 1]
    lint_clean: bool        # True if code passes basic static checks
    score: float            # final combined score (may exceed 1.0 when lint bonus applies)
    notes: list[str] = field(default_factory=list)


class BaseQualityGate(ABC):
    """Abstract interface for quality evaluation."""

    @abstractmethod
    def score(self, solution_code: str, problem: dict) -> QualityResult:
        """
        Evaluate solution_code against problem spec.

        problem dict keys (all optional, gate uses what's available):
            "tests"           list[str] — executable assert statements
            "expected_output" str       — stdout-based comparison
            "entry_point"     str       — function name (for inject-and-run)
        """


class PythonQualityGate(BaseQualityGate):
    """
    Quality gate for Python coding tasks.

    score = test_pass_rate + quality_lint_weight × lint_clean
    Max score = 1.0 + quality_lint_weight (typically 1.10).

    Uses ast.parse() for lint and exec() for test execution — Python-only.
    For other languages, subclass BaseQualityGate and use the appropriate
    subprocess runner (node, go test, javac, cargo test, etc.).
    """

    def __init__(self, config: "RoutingConfig | None" = None) -> None:
        if config is None:
            from code_capsules.controller.routing_config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        self._lint_weight = config.quality_lint_weight

    def score(self, solution_code: str, problem: dict) -> QualityResult:
        notes: list[str] = []

        # ── Lint: fast static check, no exec needed ───────────────────────────
        lint_clean = _lint_check(solution_code)
        if not lint_clean:
            notes.append("lint_failed")

        # ── Test execution ────────────────────────────────────────────────────
        tests = problem.get("tests") or []
        if not tests:
            pass_rate = 1.0
            notes.append("no_tests:vacuously_pass")
        else:
            passed, total = _run_tests(solution_code, tests)
            pass_rate = passed / total
            notes.append(f"tests:{passed}/{total}")

        combined = pass_rate + self._lint_weight * float(lint_clean)
        return QualityResult(
            test_pass_rate=pass_rate,
            lint_clean=lint_clean,
            score=combined,
            notes=notes,
        )


class BinaryQualityGate(BaseQualityGate):
    """
    Strict gate: 1.0 iff every test passes, else 0.0.
    Matches SWE-bench resolved (binary) semantics.
    """

    def score(self, solution_code: str, problem: dict) -> QualityResult:
        tests = problem.get("tests") or []
        if not tests:
            return QualityResult(
                test_pass_rate=1.0, lint_clean=True, score=1.0,
                notes=["no_tests:vacuously_pass"],
            )
        passed, total = _run_tests(solution_code, tests)
        all_pass = passed == total
        return QualityResult(
            test_pass_rate=float(passed) / total,
            lint_clean=True,
            score=1.0 if all_pass else 0.0,
            notes=[f"tests:{passed}/{total}"],
        )


# Backward-compatible alias — prefer PythonQualityGate in new code
CodeQualityGate = PythonQualityGate


def gate_from_config(config: "RoutingConfig") -> BaseQualityGate:
    """
    Factory — returns the gate named in config.quality_gate_mode.

    Supported modes:
      "python"  — PythonQualityGate (ast.parse + exec); default for Python benchmarks
      "binary"  — BinaryQualityGate (all-or-nothing); for SWE-bench resolved semantics
      "code"    — alias for "python" (legacy name; kept for backward compat)

    Other languages: subclass BaseQualityGate and register here, or pass an
    instance directly rather than using gate_from_config().
    """
    mode = getattr(config, "quality_gate_mode", "python")
    if mode == "binary":
        return BinaryQualityGate()
    return PythonQualityGate(config=config)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _lint_check(code: str) -> bool:
    """Return True if code parses and has no obvious style violations."""
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _run_tests(solution_code: str, tests: list[str]) -> tuple[int, int]:
    """
    Execute each test assert against solution_code in a fresh namespace.
    Returns (passed_count, total_count).
    """
    namespace: dict = {}
    # Compile solution separately so syntax errors are caught once
    try:
        exec(compile(textwrap.dedent(solution_code), "<solution>", "exec"), namespace)
    except Exception:
        return 0, len(tests)

    passed = 0
    for test in tests:
        try:
            exec(compile(test, "<test>", "exec"), dict(namespace))
            passed += 1
        except Exception:
            pass
    return passed, len(tests)

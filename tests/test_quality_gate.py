"""Unit tests for controller/quality_gate.py."""
import pytest
from code_capsules.evaluation.quality_gate import (
    BaseQualityGate,
    PythonQualityGate,
    CodeQualityGate,   # alias - must still import cleanly
    QualityResult,
    BinaryQualityGate,
)
from code_capsules.controller.routing_config import RoutingConfig


PASSING_CODE = """
def add(a, b):
    return a + b
"""

FAILING_CODE = """
def add(a, b):
    return a - b  # wrong
"""

LINTING_CODE = """
def add(a,b):
    x=a+b
    return x
"""


def make_problem(tests: list[str], expected_output: str = "") -> dict:
    return {"tests": tests, "expected_output": expected_output}


class TestQualityResult:
    def test_defaults(self):
        r = QualityResult(test_pass_rate=1.0, lint_clean=True, score=1.0)
        assert r.test_pass_rate == 1.0
        assert r.lint_clean is True
        assert r.score == 1.0
        assert r.notes == []

    def test_notes_populated(self):
        r = QualityResult(test_pass_rate=0.5, lint_clean=False, score=0.5,
                          notes=["lint failed", "2/4 tests pass"])
        assert len(r.notes) == 2


class TestPythonQualityGate:
    def test_all_pass(self):
        gate = PythonQualityGate()
        tests = [
            "assert add(1, 2) == 3",
            "assert add(0, 0) == 0",
            "assert add(-1, 1) == 0",
        ]
        result = gate.score(PASSING_CODE, make_problem(tests))
        assert result.test_pass_rate == 1.0
        assert result.score >= 1.0

    def test_all_fail(self):
        gate = PythonQualityGate()
        tests = [
            "assert add(1, 2) == 3",
            "assert add(2, 3) == 5",
        ]
        result = gate.score(FAILING_CODE, make_problem(tests))
        assert result.test_pass_rate == 0.0
        assert result.score < 0.2

    def test_partial_pass(self):
        gate = PythonQualityGate()
        # First passes (add(0,0) == 0 is true for subtraction too), second fails
        tests = [
            "assert add(0, 0) == 0",
            "assert add(1, 2) == 3",
        ]
        result = gate.score(FAILING_CODE, make_problem(tests))
        assert 0.0 < result.test_pass_rate < 1.0

    def test_empty_tests(self):
        gate = PythonQualityGate()
        result = gate.score(PASSING_CODE, make_problem([]))
        assert result.test_pass_rate == 1.0  # vacuously true
        assert result.score >= 1.0

    def test_score_range(self):
        gate = PythonQualityGate()
        for code in [PASSING_CODE, FAILING_CODE, LINTING_CODE]:
            result = gate.score(code, make_problem(["assert add(1,2)==3"]))
            assert 0.0 <= result.score <= 1.1  # max = 1.0 + lint_weight

    def test_lint_weight_from_config(self):
        cfg = RoutingConfig(quality_lint_weight=0.20)
        gate = PythonQualityGate(config=cfg)
        tests = ["assert add(1, 2) == 3"]
        result = gate.score(PASSING_CODE, make_problem(tests))
        # pass_rate=1.0, lint=True → score = 1.0 + 0.20 = 1.20
        assert result.score == pytest.approx(1.20, abs=0.01)

    def test_default_lint_weight(self):
        gate = PythonQualityGate()
        tests = ["assert add(1, 2) == 3"]
        result = gate.score(PASSING_CODE, make_problem(tests))
        # default lint_weight=0.10
        assert result.score == pytest.approx(1.10, abs=0.01)

    def test_zero_lint_weight(self):
        cfg = RoutingConfig(quality_lint_weight=0.0)
        gate = PythonQualityGate(config=cfg)
        tests = ["assert add(1, 2) == 3"]
        result = gate.score(PASSING_CODE, make_problem(tests))
        assert result.score == pytest.approx(1.0, abs=0.01)

    def test_syntax_error_fails_all(self):
        gate = PythonQualityGate()
        broken = "def add(a, b):\n  return a +"  # SyntaxError
        result = gate.score(broken, make_problem(["assert add(1,2)==3"]))
        assert result.test_pass_rate == 0.0

    def test_runtime_error_fails_test(self):
        gate = PythonQualityGate()
        code = "def add(a, b):\n    raise RuntimeError('boom')"
        result = gate.score(code, make_problem(["assert add(1,2)==3"]))
        assert result.test_pass_rate == 0.0

    def test_notes_non_empty(self):
        gate = PythonQualityGate()
        result = gate.score(PASSING_CODE, make_problem(["assert add(1,2)==3"]))
        assert len(result.notes) >= 1

    def test_is_base_subclass(self):
        gate = PythonQualityGate()
        assert isinstance(gate, BaseQualityGate)

    def test_alias_still_works(self):
        gate = CodeQualityGate()
        assert isinstance(gate, PythonQualityGate)


class TestBinaryQualityGate:
    def test_pass_returns_1(self):
        gate = BinaryQualityGate()
        result = gate.score(PASSING_CODE, make_problem(["assert add(1,2)==3"]))
        assert result.score == 1.0
        assert result.test_pass_rate == 1.0

    def test_fail_returns_0(self):
        gate = BinaryQualityGate()
        result = gate.score(FAILING_CODE, make_problem(["assert add(1,2)==3"]))
        assert result.score == 0.0
        assert result.test_pass_rate == 0.0

    def test_partial_pass_returns_0(self):
        gate = BinaryQualityGate()
        tests = ["assert add(0,0)==0", "assert add(1,2)==3"]
        result = gate.score(FAILING_CODE, make_problem(tests))
        assert result.score == 0.0

    def test_is_base_subclass(self):
        gate = BinaryQualityGate()
        assert isinstance(gate, BaseQualityGate)


class TestBaseQualityGate:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseQualityGate()

    def test_custom_subclass(self):
        class AlwaysOne(BaseQualityGate):
            def score(self, solution_code: str, problem: dict) -> QualityResult:
                return QualityResult(test_pass_rate=1.0, lint_clean=True, score=1.0)

        gate = AlwaysOne()
        result = gate.score("", {})
        assert result.score == 1.0


class TestGateFromConfig:
    def test_default_returns_python_gate(self):
        from code_capsules.evaluation.quality_gate import gate_from_config
        cfg = RoutingConfig()
        gate = gate_from_config(cfg)
        assert isinstance(gate, PythonQualityGate)

    def test_python_mode_explicit(self):
        from code_capsules.evaluation.quality_gate import gate_from_config
        cfg = RoutingConfig(quality_gate_mode="python")
        gate = gate_from_config(cfg)
        assert isinstance(gate, PythonQualityGate)

    def test_code_mode_alias(self):
        from code_capsules.evaluation.quality_gate import gate_from_config
        cfg = RoutingConfig(quality_gate_mode="code")
        gate = gate_from_config(cfg)
        assert isinstance(gate, PythonQualityGate)

    def test_binary_mode(self):
        from code_capsules.evaluation.quality_gate import gate_from_config
        cfg = RoutingConfig(quality_gate_mode="binary")
        gate = gate_from_config(cfg)
        assert isinstance(gate, BinaryQualityGate)

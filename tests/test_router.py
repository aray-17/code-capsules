"""Unit tests for controller/router.py."""
import pytest
from code_capsules.controller.router import Router, RouteResult, TURN_BUDGETS
from code_capsules.controller.formula import RoutingDecision
from code_capsules.controller.routing_config import RoutingConfig, DEFAULT_CONFIG


def make_router(**kwargs) -> Router:
    cfg = RoutingConfig(**kwargs) if kwargs else DEFAULT_CONFIG
    return Router(config=cfg)


class TestRouteResult:
    def test_fields_present(self):
        r = make_router()
        result = r.decide("fix the bug in auth.py", "bug_fix")
        assert isinstance(result, RouteResult)
        assert isinstance(result.decision, RoutingDecision)
        assert isinstance(result.score, float)
        assert isinstance(result.turn_budget, int)
        assert result.turn_budget > 0
        assert len(result.notes) >= 1

    def test_prediction_attached(self):
        r = make_router()
        result = r.decide("rename foo across all 10 files", "refactor")
        assert result.prediction is not None
        assert result.prediction.predicted_n_distinct_files >= 5


class TestSequentialFloor:
    def test_bug_fix_never_routes_fine(self):
        # bug_fix is in sequential_min_task_types — always elevated to SEQUENTIAL
        r = make_router()
        result = r.decide("fix the null pointer in auth.py", "bug_fix")
        assert result.decision == RoutingDecision.SEQUENTIAL

    def test_bug_fix_floor_note_in_output(self):
        r = make_router()
        result = r.decide("fix the null pointer in auth.py", "bug_fix")
        assert any("sequential_floor" in n for n in result.notes)

    def test_new_feature_can_still_route_fine(self):
        # new_feature is not in sequential_min_task_types — can be FINE
        r = make_router()
        result = r.decide(
            "implement a Python function has_close_elements(numbers, threshold)",
            "new_feature",
        )
        assert result.decision == RoutingDecision.FINE

    def test_sequential_min_task_types_configurable(self):
        # Operators can clear the floor via config (e.g. for benchmark-only runs)
        from code_capsules.controller.routing_config import RoutingConfig
        cfg = RoutingConfig(sequential_min_task_types=[])
        r = make_router.__class__  # just check config propagates
        from code_capsules.controller.router import Router
        router = Router(config=cfg)
        result = router.decide("fix the null pointer in auth.py", "bug_fix")
        # With empty floor list, a simple single-file bug_fix can route FINE
        assert result.decision == RoutingDecision.FINE


class TestFineRouting:

    def test_humaneval_prompt_routes_fine(self):
        r = make_router()
        result = r.decide(
            "implement a Python function has_close_elements(numbers, threshold) "
            "that returns True if any two numbers are closer than threshold",
            "new_feature",
        )
        assert result.decision == RoutingDecision.FINE

    def test_fine_budget_is_small(self):
        r = make_router()
        result = r.decide(
            "implement a Python function has_close_elements(numbers, threshold)",
            "new_feature",
        )
        assert result.turn_budget == TURN_BUDGETS[RoutingDecision.FINE]
        assert result.turn_budget <= 3


class TestSequentialRouting:
    def test_bash_heavy_routes_sequential(self):
        # Simulate an observed session with high bash_ratio
        r = make_router()
        result = r.decide(
            "investigate and fix the flaky test in the CI pipeline",
            "bug_fix",
            observed_bash_ratio=0.72,
        )
        assert result.decision == RoutingDecision.SEQUENTIAL

    def test_sequential_budget_larger_than_fine(self):
        r = make_router()
        fine_budget = TURN_BUDGETS[RoutingDecision.FINE]
        seq_budget = TURN_BUDGETS[RoutingDecision.SEQUENTIAL]
        assert seq_budget > fine_budget


class TestCompoundRouting:
    def test_wide_refactor_routes_compound(self):
        r = make_router()
        result = r.decide(
            "rename function process_data to handle_data across all 12 service files",
            "refactor",
        )
        assert result.decision == RoutingDecision.COMPOUND

    def test_compound_blocked_by_bug_fix_task_type(self):
        r = make_router()
        result = r.decide(
            "rename foo across all 12 files",
            "bug_fix",  # not in compound whitelist
        )
        assert result.decision != RoutingDecision.COMPOUND

    def test_compound_budget_is_adequate(self):
        assert TURN_BUDGETS[RoutingDecision.COMPOUND] >= 8


class TestObservedFeatureOverride:
    def test_observed_bash_overrides_prediction(self):
        r = make_router()
        # Wide refactor would normally predict COMPOUND, but high observed bash → SEQUENTIAL
        result = r.decide(
            "rename foo across all 10 files",
            "refactor",
            observed_bash_ratio=0.80,
        )
        assert result.decision == RoutingDecision.SEQUENTIAL

    def test_zero_bash_does_not_override(self):
        r = make_router()
        result = r.decide(
            "rename foo across all 10 files",
            "refactor",
            observed_bash_ratio=0.0,
        )
        assert result.decision == RoutingDecision.COMPOUND


class TestCustomConfig:
    def test_lower_fine_threshold_routes_more_to_sequential(self):
        # "refactor the auth module" → module scope, par_ratio≈0.10 → score≈0.05-0.15
        # default threshold (0.20): score < 0.20 → FINE
        # strict threshold (0.50): score < 0.50 still FINE, so use 0.001 to force seq
        # Better: raise compound_threshold so nothing reaches COMPOUND, lower fine_threshold
        # so the module-refactor score (>0.05) routes to SEQUENTIAL
        strict = make_router(fine_threshold=0.001, compound_threshold=0.99)
        default = make_router()
        # module-scoped refactor: par_ratio=0.10, write_concentration≈0.25 → score > 0
        prompt, tt = "refactor the auth module to remove duplication", "refactor"
        strict_result = strict.decide(prompt, tt)
        default_result = default.decide(prompt, tt)
        # strict threshold (0.001) means score > 0.001 → not FINE
        assert strict_result.decision in (RoutingDecision.SEQUENTIAL, RoutingDecision.COMPOUND)
        # default threshold (0.20) means low score → FINE
        assert default_result.decision == RoutingDecision.FINE

    def test_yaml_policy_applied(self, tmp_path):
        from code_capsules.controller.yaml_dsl import load_policy
        p = tmp_path / "p.yaml"
        # fine_threshold=0.001 forces module-scoped refactor to SEQUENTIAL
        p.write_text("routing:\n  fine_threshold: 0.001\n  compound_threshold: 0.99\n")
        cfg = load_policy(p)
        r = Router(config=cfg)
        result = r.decide("refactor the auth module to remove duplication", "refactor")
        assert result.decision != RoutingDecision.FINE

    def test_load_from_yaml_path(self, tmp_path):
        p = tmp_path / "p.yaml"
        p.write_text("routing: {}\n")
        r = Router.from_yaml(p)
        assert isinstance(r, Router)


class TestBudgets:
    def test_all_decisions_have_budget(self):
        for d in RoutingDecision:
            assert d in TURN_BUDGETS
            assert TURN_BUDGETS[d] >= 1

    def test_fine_lt_sequential(self):
        assert TURN_BUDGETS[RoutingDecision.FINE] < TURN_BUDGETS[RoutingDecision.SEQUENTIAL]

    def test_compound_lower_than_sequential(self):
        # COMPOUND is a structured batch (read-all then write-all) — efficient by design.
        # SEQUENTIAL is open-ended investigation; needs a higher ceiling for tasks
        # like SWE-bench bug fixes that average 14+ turns.
        assert TURN_BUDGETS[RoutingDecision.COMPOUND] < TURN_BUDGETS[RoutingDecision.SEQUENTIAL]

    def test_sequential_budget_sufficient_for_swebench(self):
        # Phase 0 SWE-bench baseline: avg 14.45 turns. Budget must exceed this.
        assert TURN_BUDGETS[RoutingDecision.SEQUENTIAL] >= 20


class TestBuildCompoundPrompt:
    def test_contains_all_files(self):
        files = ["a.py", "b.py", "c.py"]
        result = Router.build_compound_prompt("do the thing", files)
        for f in files:
            assert f in result

    def test_contains_original_prompt(self):
        original = "rename foo to bar across all files"
        result = Router.build_compound_prompt(original, ["x.py"])
        assert original in result

    def test_batch_instruction_present(self):
        result = Router.build_compound_prompt("fix it", ["a.py", "b.py"])
        assert "read ALL files" in result
        assert "Batch your reads before your writes" in result

    def test_file_count_in_header(self):
        files = ["a.py", "b.py", "c.py", "d.py"]
        result = Router.build_compound_prompt("do it", files)
        assert "4 total" in result

    def test_single_file_works(self):
        result = Router.build_compound_prompt("add logging", ["handler.py"])
        assert "handler.py" in result
        assert "1 total" in result

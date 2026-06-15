"""Unit tests for controller/formula.py and controller/routing_config.py."""
import pytest
from code_capsules.controller.formula import Features, RoutingDecision, score_v1, score_v2, route_v2
from code_capsules.controller.routing_config import RoutingConfig, DEFAULT_CONFIG


def make_feat(**kwargs) -> Features:
    defaults = dict(
        overhead_ratio_est=0.85, parallelizable_ratio=0.0,
        bash_ratio=0.0, context_load_ratio=0.0, write_concentration=1.0,
        read_before_write_ratio=0.0, tool_calls_per_turn=0.04, n_tool_calls=1,
        n_distinct_files=1, reads_writes_same_file_ratio=0.0, task_type="new_feature",
    )
    defaults.update(kwargs)
    return Features(**defaults)


class TestScoreV1:
    def test_humaneval_like(self):
        # 2 agents, 0.5 tools/agent, overhead=0.83, depth=1
        comp = dict(overhead_ratio_est=0.83, agent_count=2, avg_output_tokens=150,
                    tool_calls_per_agent=0.5, dependency_depth=1)
        s = score_v1(comp)
        assert 0.40 < s < 0.55, f"Expected ~0.49, got {s}"

    def test_swebench_like(self):
        comp = dict(overhead_ratio_est=0.95, agent_count=14, avg_output_tokens=400,
                    tool_calls_per_agent=1.1, dependency_depth=13)
        s = score_v1(comp)
        assert s > 0.65

    def test_range(self):
        for overhead in [0.0, 0.5, 1.0]:
            comp = dict(overhead_ratio_est=overhead, agent_count=2,
                        avg_output_tokens=100, tool_calls_per_agent=1.0, dependency_depth=1)
            s = score_v1(comp)
            assert 0.0 <= s <= 1.0


class TestScoreV2:
    def test_write_only_scores_zero(self):
        feat = make_feat(parallelizable_ratio=0.0, context_load_ratio=0.0,
                         read_before_write_ratio=0.0, tool_calls_per_turn=0.04,
                         write_concentration=1.0)
        s = score_v2(feat)
        assert s < 0.05, f"Write-only should score near 0, got {s}"

    def test_compound_candidate_scores_high(self):
        feat = make_feat(parallelizable_ratio=0.45, context_load_ratio=0.5,
                         read_before_write_ratio=0.8, tool_calls_per_turn=0.8,
                         write_concentration=0.45)
        s = score_v2(feat)
        assert s >= 0.45, f"Compound candidate should score >= 0.45, got {s}"

    def test_score_always_in_range(self):
        for par in [0.0, 0.5, 1.0]:
            for bash in [0.0, 0.5, 1.0]:
                feat = make_feat(parallelizable_ratio=par, bash_ratio=bash)
                s = score_v2(feat)
                assert 0.0 <= s <= 1.0

    def test_overhead_ratio_excluded(self):
        # Changing overhead_ratio should not change v2 score
        feat_low = make_feat(overhead_ratio_est=0.1)
        feat_high = make_feat(overhead_ratio_est=0.99)
        assert score_v2(feat_low) == score_v2(feat_high)


class TestRouteV2:
    def test_humaneval_routes_fine(self):
        feat = make_feat(bash_ratio=0.0, write_concentration=1.0,
                         parallelizable_ratio=0.0, n_distinct_files=1)
        d, s, notes = route_v2(feat, config=DEFAULT_CONFIG)
        assert d == RoutingDecision.FINE

    def test_swebench_routes_sequential(self):
        feat = make_feat(bash_ratio=0.72, parallelizable_ratio=0.11,
                         context_load_ratio=0.19, write_concentration=0.09,
                         n_distinct_files=3, task_type="bug_fix")
        d, s, notes = route_v2(feat, config=DEFAULT_CONFIG)
        assert d == RoutingDecision.SEQUENTIAL
        assert any("bash_override" in n for n in notes)

    def test_compound_routes_compound(self):
        feat = make_feat(bash_ratio=0.05, parallelizable_ratio=0.45,
                         context_load_ratio=0.5, read_before_write_ratio=0.8,
                         tool_calls_per_turn=0.8, write_concentration=0.45,
                         n_distinct_files=10, task_type="refactor")
        d, s, notes = route_v2(feat, config=DEFAULT_CONFIG)
        assert d == RoutingDecision.COMPOUND

    def test_compound_blocked_by_single_file(self):
        feat = make_feat(bash_ratio=0.05, parallelizable_ratio=0.45,
                         context_load_ratio=0.5, read_before_write_ratio=0.8,
                         tool_calls_per_turn=0.8, write_concentration=0.45,
                         n_distinct_files=1, task_type="refactor")
        d, s, notes = route_v2(feat, config=DEFAULT_CONFIG)
        assert d == RoutingDecision.SEQUENTIAL
        assert any("compound_blocked" in n for n in notes)

    def test_compound_blocked_by_task_type(self):
        feat = make_feat(bash_ratio=0.05, parallelizable_ratio=0.45,
                         context_load_ratio=0.5, read_before_write_ratio=0.8,
                         tool_calls_per_turn=0.8, write_concentration=0.45,
                         n_distinct_files=10, task_type="bug_fix")  # not in whitelist
        d, s, notes = route_v2(feat, config=DEFAULT_CONFIG)
        assert d == RoutingDecision.SEQUENTIAL

    def test_returns_three_tuple(self):
        feat = make_feat()
        result = route_v2(feat, config=DEFAULT_CONFIG)
        assert len(result) == 3
        decision, score, notes = result
        assert isinstance(decision, RoutingDecision)
        assert isinstance(score, float)
        assert isinstance(notes, list)

    def test_notes_always_non_empty(self):
        for bash in [0.0, 0.5, 0.9]:
            feat = make_feat(bash_ratio=bash)
            _, _, notes = route_v2(feat, config=DEFAULT_CONFIG)
            assert len(notes) >= 1


class TestRoutingConfig:
    def test_default_thresholds(self):
        cfg = RoutingConfig()
        assert cfg.fine_threshold == 0.20
        assert cfg.compound_threshold == 0.45
        assert cfg.bash_sequential_threshold == 0.40
        assert cfg.compound_min_distinct_files == 2

    def test_custom_config_overrides(self):
        cfg = RoutingConfig(bash_sequential_threshold=0.30, compound_min_distinct_files=5)
        assert cfg.bash_sequential_threshold == 0.30
        assert cfg.compound_min_distinct_files == 5

    def test_serialisation_round_trip(self):
        cfg = RoutingConfig(bash_sequential_threshold=0.35, compound_min_distinct_files=3)
        d = cfg.to_dict()
        cfg2 = RoutingConfig.from_dict(d)
        assert cfg2.bash_sequential_threshold == 0.35
        assert cfg2.compound_min_distinct_files == 3

    def test_compound_gates_pass(self):
        cfg = RoutingConfig()
        passes, failed = cfg.check_compound_gates("refactor", 5, 0.5, 0.3)
        assert passes
        assert failed == []

    def test_compound_gates_fail_task_type(self):
        cfg = RoutingConfig()
        passes, failed = cfg.check_compound_gates("bug_fix", 5, 0.5, 0.3)
        assert not passes
        assert any("task_type" in f for f in failed)

    def test_compound_gates_fail_distinct_files(self):
        cfg = RoutingConfig()
        passes, failed = cfg.check_compound_gates("refactor", 1, 0.5, 0.3)
        assert not passes
        assert any("n_distinct_files" in f for f in failed)

    def test_custom_task_type_whitelist(self):
        cfg = RoutingConfig(compound_task_type_whitelist=["multi_document_summary"])
        passes, _ = cfg.check_compound_gates("multi_document_summary", 5, 0.5, 0.3)
        assert passes
        passes2, _ = cfg.check_compound_gates("refactor", 5, 0.5, 0.3)
        assert not passes2

    def test_empty_whitelist_allows_all(self):
        cfg = RoutingConfig(compound_task_type_whitelist=[])
        passes, _ = cfg.check_compound_gates("anything", 5, 0.5, 0.3)
        assert passes

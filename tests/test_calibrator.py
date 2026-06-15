"""Tests for controller/calibrator.py — Phase 7B turn-budget recommendation."""
import json
import pytest
from pathlib import Path

from code_capsules.controller.calibrator import (
    BudgetObservation,
    TurnBudgetRecommendation,
    calibrate_turn_budget,
    observations_from_sweep_jsonl,
    _pareto_frontier,
    _detect_knee,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _obs(budget, pass_rate, cost=0.1, n=30, turns=None, cap=0.0):
    return BudgetObservation(
        budget=budget,
        n=n,
        pass_rate=pass_rate,
        avg_effective_input_tokens=100_000.0,
        avg_cost_usd=cost,
        avg_actual_turns=turns if turns is not None else min(budget, budget * 0.9),
        cap_hit_rate=cap,
    )


# ── Pareto frontier ──────────────────────────────────────────────────────────

class TestParetoFrontier:
    def test_strictly_improving_curve_keeps_all(self):
        obs = [_obs(5, 0.20, cost=0.05),
               _obs(10, 0.40, cost=0.10),
               _obs(20, 0.60, cost=0.20),
               _obs(40, 0.65, cost=0.40)]
        frontier = _pareto_frontier(obs)
        assert len(frontier) == 4
        # Sorted ascending
        assert [f.budget for f in frontier] == [5, 10, 20, 40]

    def test_dominated_point_dropped(self):
        # budget=10 has both worse pass_rate AND worse cost than budget=20 → dominated
        obs = [_obs(5, 0.20, cost=0.05),
               _obs(10, 0.30, cost=0.30),    # dominated by 20
               _obs(20, 0.50, cost=0.20)]
        frontier = _pareto_frontier(obs)
        assert {f.budget for f in frontier} == {5, 20}

    def test_single_point_is_its_own_frontier(self):
        obs = [_obs(20, 0.5, cost=0.2)]
        assert _pareto_frontier(obs) == obs

    def test_equal_pass_rate_higher_cost_is_dominated(self):
        # WEAK dominance: same pass-rate + higher cost → dominated.
        # (Fixed after the Phase 7B plateau-jump-plateau bug.)
        obs = [_obs(20, 0.5, cost=0.2),
               _obs(40, 0.5, cost=0.4)]
        frontier = _pareto_frontier(obs)
        assert {f.budget for f in frontier} == {20}

    def test_equal_cost_higher_pass_dominates_lower(self):
        # WEAK dominance the other axis: same cost + higher pass-rate → wins.
        obs = [_obs(20, 0.5, cost=0.2),
               _obs(40, 0.7, cost=0.2)]
        frontier = _pareto_frontier(obs)
        assert {f.budget for f in frontier} == {40}

    def test_ties_on_both_axes_keep_both(self):
        # Identical on both axes → neither dominates → keep both.
        obs = [_obs(20, 0.5, cost=0.2), _obs(40, 0.5, cost=0.2)]
        frontier = _pareto_frontier(obs)
        assert len(frontier) == 2

    def test_plateau_jump_plateau_curve_collapses_to_two_points(self):
        """
        Regression test for the Phase 7B calibrator bug.

        Real Phase 7B SWE-bench data: budget=10 and budget=40 produced the same
        pass-rate as the points on either side, but at higher cost. With strict
        dominance both stayed on the frontier and the knee detector returned 5
        (first low-gain transition: 5→10). With weak dominance the frontier
        correctly collapses to {5, 20}, and the calibrator recommends 20.
        """
        obs = [
            _obs(5,  0.033, cost=0.19),
            _obs(10, 0.033, cost=0.36),    # same pass as 5, higher cost → dominated
            _obs(20, 0.167, cost=0.69),
            _obs(40, 0.167, cost=0.94),    # same pass as 20, higher cost → dominated
        ]
        frontier = _pareto_frontier(obs)
        assert {f.budget for f in frontier} == {5, 20}


# ── Knee detection ───────────────────────────────────────────────────────────

class TestKneeDetection:
    def test_clear_knee_picks_smaller_budget(self):
        # 5→10: +0.20/5 = 0.04/turn (steep)
        # 10→20: +0.20/10 = 0.02/turn (still gaining)
        # 20→40: +0.05/20 = 0.0025/turn (knee — below 0.005)
        frontier = [_obs(5, 0.20), _obs(10, 0.40), _obs(20, 0.60), _obs(40, 0.65)]
        rec, reasoning, fallback = _detect_knee(frontier, min_quality_gain_per_turn=0.005)
        assert rec.budget == 20
        assert fallback is False
        assert "20" in reasoning

    def test_no_knee_returns_largest_with_fallback(self):
        # All transitions exceed threshold — no knee in this sweep
        frontier = [_obs(5, 0.20), _obs(10, 0.50), _obs(20, 0.80)]
        rec, _, fallback = _detect_knee(frontier, min_quality_gain_per_turn=0.005)
        assert rec.budget == 20
        assert fallback is True

    def test_single_point_is_recommendation_with_fallback(self):
        rec, reasoning, fallback = _detect_knee([_obs(20, 0.5)], 0.005)
        assert rec.budget == 20
        assert fallback is True
        assert "no knee" in reasoning.lower() or "only one" in reasoning.lower()

    def test_threshold_changes_recommendation(self):
        # Lower threshold should pick a larger budget (squeeze more quality)
        frontier = [_obs(5, 0.20), _obs(10, 0.40), _obs(20, 0.50), _obs(40, 0.55)]
        # At threshold 0.005: 20→40 is 0.05/20 = 0.0025/turn → knee at 20
        rec, _, _ = _detect_knee(frontier, min_quality_gain_per_turn=0.005)
        assert rec.budget == 20
        # At threshold 0.001: 20→40 is 0.0025/turn → still ABOVE → no knee → fallback to 40
        rec, _, fallback = _detect_knee(frontier, min_quality_gain_per_turn=0.001)
        assert rec.budget == 40
        assert fallback is True


# ── Top-level calibrate_turn_budget ──────────────────────────────────────────

class TestCalibrate:
    def test_realistic_swebench_curve_recommends_20(self):
        # Mirrors the Phase 7 audit findings: most quality gain by budget=20,
        # diminishing returns at 40
        obs = [
            _obs(5, 0.05, cost=0.05),
            _obs(10, 0.20, cost=0.12),
            _obs(20, 0.30, cost=0.25),
            _obs(40, 0.32, cost=0.50),       # +0.02/20 = 0.001/turn → below default
        ]
        rec = calibrate_turn_budget(obs)
        assert rec.recommended_budget == 20
        assert rec.fallback_used is False
        assert rec.target_quality_gain_per_turn == 0.005

    def test_phase7b_actual_data_recommends_20(self):
        """
        End-to-end regression: feed the actual Phase 7B SWE-bench sweep results
        into the top-level calibrate_turn_budget(). With the weak-dominance fix
        the recommendation must be 20, not 5.

        Real numbers from evals/phase7b_sweep_20260511T044819_tb{5,10,20,40}.jsonl.
        """
        obs = [
            _obs(5,  0.033, cost=0.190),
            _obs(10, 0.033, cost=0.356),
            _obs(20, 0.167, cost=0.694),
            _obs(40, 0.167, cost=0.942),
        ]
        rec = calibrate_turn_budget(obs)
        assert rec.recommended_budget == 20
        # Frontier is collapsed by weak dominance to {5, 20}.
        assert {f.budget for f in rec.pareto_frontier} == {5, 20}
        # 5→20 gain is (0.167-0.033)/15 = 0.0089/turn ≈ 0.89%/turn — above the
        # 0.5%/turn default, so no knee found → fallback to largest in frontier.
        assert rec.fallback_used is True

    def test_returns_full_frontier_on_recommendation(self):
        obs = [_obs(5, 0.10), _obs(10, 0.30), _obs(20, 0.50), _obs(40, 0.55)]
        rec = calibrate_turn_budget(obs)
        assert isinstance(rec.pareto_frontier, list)
        assert len(rec.pareto_frontier) >= 1
        assert all(isinstance(o, BudgetObservation) for o in rec.pareto_frontier)

    def test_empty_observations_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            calibrate_turn_budget([])

    def test_recommendation_is_immutable(self):
        rec = calibrate_turn_budget([_obs(20, 0.5)])
        with pytest.raises((AttributeError, Exception)):
            rec.recommended_budget = 999    # frozen dataclass


# ── observations_from_sweep_jsonl ────────────────────────────────────────────

class TestObservationsFromJsonl:
    def _row(self, budget, resolved, turns=None, in_tok=100_000, cost=0.1):
        return {
            "instance_id": f"foo__bar-{budget}",
            "turn_budget": budget,
            "actual_turns": turns if turns is not None else int(budget * 0.7),
            "resolved": resolved,
            "effective_input_tokens": in_tok,
            "cost_usd": cost,
        }

    def test_aggregates_one_observation_per_budget(self, tmp_path):
        f1 = tmp_path / "tb5.jsonl"
        f2 = tmp_path / "tb20.jsonl"
        f1.write_text("\n".join(json.dumps(self._row(5, r)) for r in [True, False, False]) + "\n")
        f2.write_text("\n".join(json.dumps(self._row(20, r)) for r in [True, True, True, False]) + "\n")

        obs = observations_from_sweep_jsonl([f1, f2])
        assert len(obs) == 2
        by_budget = {o.budget: o for o in obs}
        assert by_budget[5].pass_rate == pytest.approx(1/3)
        assert by_budget[20].pass_rate == pytest.approx(3/4)

    def test_skips_rows_with_resolved_none(self, tmp_path):
        f = tmp_path / "x.jsonl"
        rows = [self._row(20, True), self._row(20, None), self._row(20, False)]
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        obs = observations_from_sweep_jsonl([f])
        assert obs[0].n == 2     # only 2 evaluated (the None was skipped)
        assert obs[0].pass_rate == 0.5

    def test_cap_hit_rate_computed(self, tmp_path):
        f = tmp_path / "x.jsonl"
        # budget=10, half hit cap, half don't
        rows = [self._row(10, True, turns=10), self._row(10, True, turns=10),
                self._row(10, False, turns=5), self._row(10, False, turns=8)]
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        obs = observations_from_sweep_jsonl([f])
        assert obs[0].cap_hit_rate == 0.5

    def test_skips_unparseable_lines(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text(json.dumps(self._row(20, True)) + "\nNOT JSON\n" +
                     json.dumps(self._row(20, False)) + "\n")
        obs = observations_from_sweep_jsonl([f])
        assert obs[0].n == 2

    def test_skips_rows_missing_turn_budget(self, tmp_path):
        f = tmp_path / "x.jsonl"
        bad = {"resolved": True, "actual_turns": 5}     # no turn_budget
        good = self._row(20, True)
        f.write_text(json.dumps(bad) + "\n" + json.dumps(good) + "\n")
        obs = observations_from_sweep_jsonl([f])
        assert len(obs) == 1
        assert obs[0].budget == 20

"""
Turn-budget calibrator.

Coding-agent analog of Agentic-Capsules' `Pipeline.calibrate()`, which
recommends `compose_at` / `quality_floor` thresholds from observed pipeline
data. For coding agents, the dimension that matters is the turn budget: our
audit showed it dominates mode selection on monomorphic workloads.

Given a turn-budget sweep on a representative task sample, this module:

  1. Computes the observed (cost, quality) point at each budget.
  2. Fits the cost-quality Pareto frontier.
  3. Recommends the "knee" budget: the smallest budget where additional
     turns no longer materially improve quality (configurable threshold).

The recommendation is a `TurnBudgetRecommendation` dataclass, read-only.
The operator applies it explicitly (mirrors AC's `dataclasses.replace()` pattern).

Usage:
    from code_capsules.controller.calibrator import calibrate_turn_budget

    # observations = list of (budget, pass_rate, avg_cost) tuples from the sweep
    rec = calibrate_turn_budget(observations,
                                min_quality_gain_per_extra_turn=0.005)
    print(rec.recommended_budget)        # e.g. 20
    print(rec.knee_reasoning)            # e.g. "20→40 added 0.003/turn (< 0.005)"
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BudgetObservation:
    """One row of the turn-budget sweep: aggregate stats at one budget."""
    budget: int                  # max-turns value used in the run
    n: int                       # number of tasks evaluated
    pass_rate: float             # fraction of tasks passed (0-1)
    avg_effective_input_tokens: float
    avg_cost_usd: float
    avg_actual_turns: float      # how many turns the model actually used on average
    cap_hit_rate: float          # fraction of tasks that hit the cap (actual==budget)


@dataclass(frozen=True)
class TurnBudgetRecommendation:
    """Output of the calibrator. Read-only; operator applies explicitly."""
    recommended_budget: int
    pareto_frontier: list[BudgetObservation]   # sorted ascending by budget
    knee_reasoning: str
    target_quality_gain_per_turn: float        # threshold used for knee detection
    fallback_used: bool = False                # True when no clear knee found


# ── Pareto frontier ──────────────────────────────────────────────────────────

def _pareto_frontier(obs: list[BudgetObservation]) -> list[BudgetObservation]:
    """
    Returns observations on the cost-vs-quality Pareto frontier under WEAK
    dominance: a point is dominated if another point is at least as good on
    BOTH axes (>= pass_rate AND <= cost) and strictly better on at least one.

    Why weak: strict-dominance kept points like (budget=10, pass=3.3%, cost=$0.36)
    on the frontier when (budget=5, pass=3.3%, cost=$0.19) was identically good
    on quality but cheaper. That cluttered the curve with "same quality, higher
    cost" siblings and confused the knee detector, we'd return a low budget
    as the knee just because the next budget had the same pass-rate.

    Two points with identical pass_rate and identical cost are kept; ties on
    only one axis are resolved in favour of the strict winner.
    """
    frontier = []
    for o in obs:
        dominated = False
        for other in obs:
            if other is o:
                continue
            # Other is at least as good on both axes
            as_good = (other.pass_rate >= o.pass_rate and
                       other.avg_cost_usd <= o.avg_cost_usd)
            # And strictly better on at least one
            strictly_better = (other.pass_rate > o.pass_rate or
                               other.avg_cost_usd < o.avg_cost_usd)
            if as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(o)
    return sorted(frontier, key=lambda x: x.budget)


# ── Knee detection ───────────────────────────────────────────────────────────

def _detect_knee(
    frontier: list[BudgetObservation],
    min_quality_gain_per_turn: float,
) -> tuple[BudgetObservation, str, bool]:
    """
    Find the knee of the Pareto curve: the smallest budget where adding more
    turns no longer buys >= `min_quality_gain_per_turn` per extra turn.

    Returns (recommended_observation, reasoning_string, fallback_used).

    Algorithm:
      Walk frontier ascending by budget. For each adjacent pair (b1, b2),
      compute (pass_rate_2 - pass_rate_1) / (b2 - b1), quality gain per turn.
      The first pair where this drops below the threshold marks the knee:
      pick b1 (the smaller budget; b2's extra turns aren't paying off).

    Fallback: if no pair drops below threshold, return the largest-budget
    observation (no knee found, quality is still improving). Caller can
    interpret fallback_used=True as "you're sweeping a range that's still
    on the steep part of the curve; extend the sweep upward."
    """
    if not frontier:
        raise ValueError("Empty Pareto frontier: need at least one observation")
    if len(frontier) == 1:
        only = frontier[0]
        return only, f"only one budget observed ({only.budget}); no knee to detect", True

    for i in range(len(frontier) - 1):
        a, b = frontier[i], frontier[i + 1]
        delta_turns = b.budget - a.budget
        if delta_turns <= 0:
            continue
        gain_per_turn = (b.pass_rate - a.pass_rate) / delta_turns
        if gain_per_turn < min_quality_gain_per_turn:
            return a, (
                f"knee at budget={a.budget}: going {a.budget}→{b.budget} "
                f"added {gain_per_turn*100:.2f}%/turn pass-rate "
                f"(< threshold {min_quality_gain_per_turn*100:.2f}%/turn)"
            ), False

    # No knee found - quality still improving at the top of the sweep
    last = frontier[-1]
    return last, (
        f"no knee detected within sweep range; quality still improving at "
        f"budget={last.budget}. Consider extending the sweep upward."
    ), True


# ── Main entry point ─────────────────────────────────────────────────────────

def calibrate_turn_budget(
    observations: list[BudgetObservation],
    min_quality_gain_per_turn: float = 0.005,   # 0.5% pass-rate per extra turn
) -> TurnBudgetRecommendation:
    """
    Given a turn-budget sweep on a representative sample, recommend the
    Pareto-knee budget.

    `min_quality_gain_per_turn` is the threshold below which extra turns are
    deemed "not worth it." Default 0.005 = 0.5% pass-rate per extra turn.
    Operators with cheaper compute can lower it (squeeze more quality);
    operators with expensive compute can raise it (accept more quality cost
    in exchange for shorter runs).
    """
    if not observations:
        raise ValueError("Need at least one BudgetObservation to calibrate")

    frontier = _pareto_frontier(observations)
    rec_obs, reasoning, fallback = _detect_knee(frontier, min_quality_gain_per_turn)

    return TurnBudgetRecommendation(
        recommended_budget=rec_obs.budget,
        pareto_frontier=frontier,
        knee_reasoning=reasoning,
        target_quality_gain_per_turn=min_quality_gain_per_turn,
        fallback_used=fallback,
    )


# ── JSONL → observations helper (for use with run_swebench_docker.py output) ──

def observations_from_sweep_jsonl(
    jsonl_paths: list[Path],
) -> list[BudgetObservation]:
    """
    Aggregate per-instance JSONL rows from a turn-budget sweep into one
    BudgetObservation per distinct turn_budget value.

    Each input JSONL is expected to follow the run_swebench_docker.py schema:
    rows with fields {turn_budget, actual_turns, resolved, effective_input_tokens,
    cost_usd}.
    """
    by_budget: dict[int, list[dict]] = {}
    for path in jsonl_paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                budget = row.get("turn_budget")
                if budget is None:
                    continue
                by_budget.setdefault(budget, []).append(row)

    observations: list[BudgetObservation] = []
    for budget, rows in sorted(by_budget.items()):
        evaluated = [r for r in rows if r.get("resolved") is not None]
        if not evaluated:
            continue
        n = len(evaluated)
        pass_rate = sum(1 for r in evaluated if r.get("resolved")) / n
        avg_in = sum(r.get("effective_input_tokens", 0) for r in evaluated) / n
        avg_cost = sum(r.get("cost_usd", 0) for r in evaluated) / n
        avg_turns = sum(r.get("actual_turns", 0) for r in evaluated) / n
        cap_hits = sum(1 for r in evaluated
                       if r.get("actual_turns", 0) >= budget) / n
        observations.append(BudgetObservation(
            budget=budget,
            n=n,
            pass_rate=pass_rate,
            avg_effective_input_tokens=avg_in,
            avg_cost_usd=avg_cost,
            avg_actual_turns=avg_turns,
            cap_hit_rate=cap_hits,
        ))
    return observations


# ── CLI for ad-hoc analysis ──────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Turn-budget calibrator")
    parser.add_argument("jsonls", nargs="+", type=Path,
                        help="Sweep JSONL files (one per turn-budget value)")
    parser.add_argument("--min-gain-per-turn", type=float, default=0.005,
                        help="Threshold pass-rate gain per extra turn for knee detection")
    args = parser.parse_args()

    obs = observations_from_sweep_jsonl(args.jsonls)
    if not obs:
        print("No usable observations found in input files.")
        return

    print(f"\nObserved budgets:")
    print(f"  {'budget':>6} {'n':>3} {'pass%':>6} {'avg_t':>5} {'cap%':>5} {'cost$':>7}")
    for o in obs:
        print(f"  {o.budget:>6} {o.n:>3} {o.pass_rate*100:>5.1f}% {o.avg_actual_turns:>5.1f} "
              f"{o.cap_hit_rate*100:>4.1f}% {o.avg_cost_usd:>7.3f}")

    rec = calibrate_turn_budget(obs, args.min_gain_per_turn)
    print(f"\nRecommended turn budget: {rec.recommended_budget}")
    print(f"  Reasoning: {rec.knee_reasoning}")
    if rec.fallback_used:
        print("  Fallback used: see reasoning")


if __name__ == "__main__":
    _cli()

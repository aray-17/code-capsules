"""
Code-Capsules composition score formula: v1 (inherited) and v2 (recalibrated).

v1: inherited from Agentic-Capsules.
v2: redesigned for coding agents using the baseline and instrumentation findings.

Routing decisions:
  FINE       - one LLM call, one write, done. score < FINE_THRESHOLD.
  SEQUENTIAL - long dependency chain (bash loops, read→patch→verify).
               FINE_THRESHOLD ≤ score < COMPOUND_THRESHOLD, OR bash_ratio high.
  COMPOUND   - batchable reads + parallel edits. score ≥ COMPOUND_THRESHOLD
               AND par_ratio > PAR_THRESHOLD.
"""
from __future__ import annotations

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from code_capsules.controller.routing_config import RoutingConfig


class RoutingDecision(str, Enum):
    FINE = "fine"
    SEQUENTIAL = "sequential"
    COMPOUND = "compound"


FINE_THRESHOLD = 0.20
COMPOUND_THRESHOLD = 0.45
PAR_THRESHOLD = 0.15        # par_ratio above this → COMPOUND candidate
BASH_SEQUENTIAL_THRESHOLD = 0.40  # bash_ratio above this → force SEQUENTIAL


@dataclass
class Features:
    """Normalised feature vector for scoring. All values in [0, 1]."""
    overhead_ratio_est: float       # cache_read / effective_input (unreliable in CC)
    parallelizable_ratio: float     # width / (width + depth)
    bash_ratio: float               # bash / n_total
    context_load_ratio: float       # reads / n_total
    write_concentration: float      # writes / n_total
    read_before_write_ratio: float  # reads→writes / n_writes
    tool_calls_per_turn: float      # n_total / n_turns (normalised to [0,1] at 5+)
    n_tool_calls: int               # raw count
    n_distinct_files: int = 0       # unique files touched (COMPOUND gate)
    reads_writes_same_file_ratio: float = 0.0  # overlap of read/write file sets
    task_type: str = "unknown"      # from task_classifier (COMPOUND gate)
    multi_file_scope: bool = False  # scope keywords detected in prompt

    @classmethod
    def from_dag_and_session(cls, dag: dict, session: dict, comp: dict) -> "Features":
        n = dag.get("n_tool_calls", 0)
        n_turns = session.get("num_turns", dag.get("estimated_depth", 1))
        tpt = (n / max(n_turns, 1)) / 5.0   # normalise: 5+ tools/turn → 1.0

        # bash_ratio not in the baseline comp dict - derive from dag
        n_bash = dag.get("n_bash", 0)
        bash_ratio = (n_bash / n) if n > 0 else 0.0

        rbw = dag.get("read_before_write_ratio", comp.get("read_before_write_ratio", 0.0))

        return cls(
            overhead_ratio_est=comp.get("overhead_ratio_est", 0.0),
            parallelizable_ratio=dag.get("parallelizable_ratio", comp.get("parallelizable_ratio", 0.0)),
            bash_ratio=bash_ratio,
            context_load_ratio=dag.get("context_load_ratio", comp.get("context_load_ratio", 0.0)),
            write_concentration=dag.get("write_concentration", comp.get("write_concentration", 0.0)),
            read_before_write_ratio=rbw,
            tool_calls_per_turn=min(tpt, 1.0),
            n_tool_calls=n,
            n_distinct_files=dag.get("n_distinct_files", 0),
            reads_writes_same_file_ratio=dag.get("reads_writes_same_file_ratio", 0.0),
            task_type=session.get("task_type", "unknown"),
        )


# ── v1: inherited formula ─────────────────────────────────────────────────────

def score_v1(comp: dict) -> float:
    """Exact replica of the Agentic-Capsules formula."""
    overhead = comp.get("overhead_ratio_est", 0.0)
    agents = comp.get("agent_count", 1)
    avg_out = comp.get("avg_output_tokens", 0.0)
    tools_per_agent = comp.get("tool_calls_per_agent", 0.0)
    depth = comp.get("dependency_depth", 0)
    return (
        0.45 * overhead
        + 0.25 * min(agents / 4, 1.0)
        + 0.00 * min(avg_out / 300, 1.0)
        + 0.25 * min(tools_per_agent / 3, 1.0)
        - 0.05 * min(depth / max(agents - 1, 1), 1.0)
    )


# ── v2: recalibrated formula ──────────────────────────────────────────────────

_V2_WEIGHTS: dict[str, float] = {
    # Set during the Pareto sweep - defaults below are the sweep winners.
    # Features that INCREASE compound/sequential score:
    "parallelizable_ratio":    0.40,  # main signal: wide DAG → compound
    "context_load_ratio":      0.20,  # lots of reads → batching opportunity
    "read_before_write_ratio": 0.20,  # read-then-write pattern → compound
    "tool_calls_per_turn":     0.10,  # many tools/turn → parallel activity
    # Features that DECREASE score (write-only / simple tasks):
    "write_concentration":    -0.10,  # all writes → already fine-grained
}


def score_v2(feat: Features, weights: dict[str, float] | None = None) -> float:
    """
    Recalibrated formula for coding agents.
    overhead_ratio_est intentionally excluded - unreliable in Claude Code.
    bash_ratio handled separately in route_v2 (forces SEQUENTIAL, not score).
    """
    w = weights or _V2_WEIGHTS
    raw = (
        w.get("parallelizable_ratio", 0)    * feat.parallelizable_ratio
        + w.get("context_load_ratio", 0)    * feat.context_load_ratio
        + w.get("read_before_write_ratio", 0) * feat.read_before_write_ratio
        + w.get("tool_calls_per_turn", 0)   * feat.tool_calls_per_turn
        + w.get("write_concentration", 0)   * feat.write_concentration
    )
    return max(0.0, min(1.0, raw))


def route_v2(
    feat: Features,
    weights: dict[str, float] | None = None,
    config: Optional["RoutingConfig"] = None,
    # Legacy keyword args for backward compat with pareto.py sweep
    fine_threshold: float = FINE_THRESHOLD,
    compound_threshold: float = COMPOUND_THRESHOLD,
    par_threshold: float = PAR_THRESHOLD,
    bash_sequential_threshold: float = BASH_SEQUENTIAL_THRESHOLD,
) -> tuple[RoutingDecision, float, list[str]]:
    """
    Returns (routing_decision, score_v2, routing_notes).

    routing_notes: list of strings explaining the decision - useful for logging
    and for the paper's ablation analysis.

    Rules applied in order:
    1. bash_ratio > threshold → SEQUENTIAL (Bash exploration loop, not batchable)
    2. score < fine_threshold → FINE (trivial write-from-scratch task)
    3. COMPOUND gates (all must pass):
       - task_type in whitelist
       - n_distinct_files ≥ minimum
       - read_before_write_ratio ≥ minimum
       - par_ratio ≥ threshold
       - score ≥ compound_threshold
       → COMPOUND if all pass
    4. Otherwise → SEQUENTIAL
    """
    if config is not None:
        ft = config.fine_threshold
        ct = config.compound_threshold
        pt = config.par_ratio_threshold
        bt = config.bash_sequential_threshold
        w = config.formula_weights
    else:
        ft, ct, pt, bt, w = fine_threshold, compound_threshold, par_threshold, bash_sequential_threshold, weights

    score = score_v2(feat, w)
    notes: list[str] = []

    # Rule 1: bash override
    if feat.bash_ratio >= bt:
        notes.append(f"bash_override: bash_ratio={feat.bash_ratio:.2f} >= {bt}")
        return RoutingDecision.SEQUENTIAL, score, notes

    # Rule 2: trivial task - but elevate to SEQUENTIAL for task types that
    # structurally require iteration (bug_fix: read→investigate→patch→verify).
    if score < ft:
        if config is not None and feat.task_type in config.sequential_min_task_types:
            notes.append(
                f"sequential_floor: {feat.task_type} requires sequential "
                f"(score={score:.3f} would be fine)"
            )
            return RoutingDecision.SEQUENTIAL, score, notes
        notes.append(f"fine: score={score:.3f} < {ft}")
        return RoutingDecision.FINE, score, notes

    # Rule 3: COMPOUND gating
    if score >= ct:
        if config is not None:
            passes, failed_gates = config.check_compound_gates(
                task_type=feat.task_type,
                n_distinct_files=feat.n_distinct_files,
                read_before_write_ratio=feat.read_before_write_ratio,
                par_ratio=feat.parallelizable_ratio,
            )
        else:
            passes = feat.parallelizable_ratio >= pt
            failed_gates = [] if passes else [f"par_ratio={feat.parallelizable_ratio:.2f} < {pt}"]

        if passes:
            notes.append(f"compound: score={score:.3f} >= {ct}, all gates passed")
            return RoutingDecision.COMPOUND, score, notes
        else:
            notes.append(f"compound_blocked: {'; '.join(failed_gates)}")

    notes.append(f"sequential: default (score={score:.3f})")
    return RoutingDecision.SEQUENTIAL, score, notes


def get_v2_weights() -> dict[str, float]:
    return dict(_V2_WEIGHTS)


def set_v2_weights(weights: dict[str, float]) -> None:
    _V2_WEIGHTS.update(weights)

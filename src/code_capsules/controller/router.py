"""
Live router — the policy execution layer.

Takes a prompt + optional observed session features, produces a routing
decision and turn budget. This is the component that fires at the start
of every session to determine how Claude should be invoked.

Usage:
    router = Router(config=DEFAULT_CONFIG)
    result = router.decide(prompt, task_type)
    subprocess.run(["claude", "-p", prompt, "--max-turns", str(result.turn_budget)])

Or with a YAML policy:
    router = Router.from_yaml("policy.yaml")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from code_capsules.controller.formula import Features, RoutingDecision, route_v2
from code_capsules.controller.routing_config import DEFAULT_CONFIG, RoutingConfig

# NOTE: the prompt predictor + task classifier live in the optional private `hooks/`
# package (superseded for the live framework by controller/routing.py). They are
# imported lazily inside decide() so this module stays standalone-importable without
# hooks/ present; calling decide() still requires hooks/ (legacy path).


# Turn budgets per routing decision.
# FINE: tight — forces single-shot; catches most simple tasks in 1-2 turns.
# SEQUENTIAL: generous — SWE-bench bug fixes average 14.45 turns (Phase 0).
#   Models self-terminate early for simple tasks, so a high ceiling is safe.
# COMPOUND: lower than SEQUENTIAL — structured batch prompt (read-all then write-all)
#   is efficient; doesn't need an open-ended investigation budget.
TURN_BUDGETS: dict[RoutingDecision, int] = {
    RoutingDecision.FINE: 2,
    RoutingDecision.SEQUENTIAL: 20,
    RoutingDecision.COMPOUND: 8,
}


@dataclass
class RouteResult:
    decision: RoutingDecision
    score: float
    notes: list[str]
    turn_budget: int
    prediction: PromptPrediction
    task_type: str
    observed_features_used: bool = False   # True when partial session data overrode prediction


class Router:
    """
    Routing policy executor.

    decide() is the hot path — called once per session before the first tool call.
    It combines the prompt prediction (always available) with any observed session
    features (available mid-session from hook data) to produce a routing decision.
    """

    def __init__(self, config: RoutingConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Router":
        from code_capsules.controller.yaml_dsl import load_policy
        return cls(config=load_policy(Path(path)))

    def decide(
        self,
        prompt: str,
        task_type: Optional[str] = None,
        *,
        observed_bash_ratio: float = 0.0,
        observed_par_ratio: Optional[float] = None,
        observed_n_distinct_files: Optional[int] = None,
        observed_context_load_ratio: float = 0.0,
        observed_rbw_ratio: float = 0.0,
    ) -> RouteResult:
        """
        Produce a routing decision for the given prompt.

        Observed features override predictions when provided (non-zero / non-None).
        This supports mid-session re-routing after a few tool calls have fired.

        Args:
            prompt: The user's raw task prompt.
            task_type: Pre-classified task type (e.g. "bug_fix"). If None,
                       classify() is called automatically.
            observed_bash_ratio: Actual bash_ratio from hook data so far.
            observed_par_ratio: Actual par_ratio from hook data (None = use prediction).
            observed_n_distinct_files: Actual distinct files from hook data.
            observed_context_load_ratio: Actual context_load_ratio from hook data.
            observed_rbw_ratio: Actual read-before-write ratio from hook data.
        """
        from code_capsules.controller.prompt_predictor import predict_from_prompt
        from code_capsules.controller.task_classifier import classify

        if task_type is None:
            clf = classify(prompt)
            task_type = clf.task_type.value

        prediction = predict_from_prompt(prompt, task_type)

        # Build Features — merge prediction with any observed data
        observed_used = False
        bash_ratio = observed_bash_ratio  # always prefer observed (bash is runtime-only)
        if observed_bash_ratio > 0:
            observed_used = True

        par_ratio = (
            observed_par_ratio
            if observed_par_ratio is not None
            else prediction.predicted_par_ratio
        )
        n_distinct_files = (
            observed_n_distinct_files
            if observed_n_distinct_files is not None
            else prediction.predicted_n_distinct_files
        )
        context_load = (
            observed_context_load_ratio
            if observed_context_load_ratio > 0
            else 0.0
        )
        rbw_ratio = (
            observed_rbw_ratio
            if observed_rbw_ratio > 0
            else (0.80 if prediction.predicted_independent_edits else 0.0)
        )

        if any([observed_par_ratio is not None,
                observed_n_distinct_files is not None,
                observed_context_load_ratio > 0,
                observed_rbw_ratio > 0]):
            observed_used = True

        # write_concentration: inverse of n_distinct_files (many files = less concentrated)
        write_concentration = max(1.0 / max(n_distinct_files, 1), 0.10)

        features = Features(
            overhead_ratio_est=0.85,
            parallelizable_ratio=par_ratio,
            bash_ratio=bash_ratio,
            context_load_ratio=context_load,
            write_concentration=write_concentration,
            read_before_write_ratio=rbw_ratio,
            tool_calls_per_turn=0.04,      # conservative pre-routing estimate
            n_tool_calls=max(n_distinct_files, 1),
            n_distinct_files=n_distinct_files,
            reads_writes_same_file_ratio=0.0,
            task_type=task_type,
        )

        decision, score, notes = route_v2(features, config=self.config)
        return RouteResult(
            decision=decision,
            score=score,
            notes=notes,
            turn_budget=TURN_BUDGETS[decision],
            prediction=prediction,
            task_type=task_type,
            observed_features_used=observed_used,
        )

    @staticmethod
    def build_compound_prompt(original_prompt: str, file_list: list[str]) -> str:
        """
        Wrap a prompt with batching instructions for COMPOUND routing.

        COMPOUND execution is a single Claude session — the savings come from
        reading all files in one batched pass before writing, so the system-prompt
        cache is paid once and Claude has full context before any edit.

        The structured prefix below works with any LLM that follows instructions;
        it is not Claude-specific.
        """
        files_block = "\n".join(f"  - {f}" for f in file_list)
        return (
            f"Files to modify ({len(file_list)} total):\n{files_block}\n\n"
            f"Strategy: read ALL files above first to understand the full scope, "
            f"then apply the required change to each file. "
            f"Batch your reads before your writes — do not interleave "
            f"read-then-write per file.\n\n"
            f"{original_prompt}"
        )

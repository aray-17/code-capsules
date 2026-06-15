"""
RoutingConfig — all thresholds, weights, and gating criteria in one place.

Developers tune this without touching formula logic. Phase 5 will add
YAML loading so policy files can override defaults at deploy time.

Design: mirrors Agentic-Capsules' policy layer — mechanism (formula.py)
is separate from policy (routing_config.py / Phase 5 YAML DSL).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class RoutingConfig:
    # ── Routing thresholds ────────────────────────────────────────────────────
    fine_threshold: float = 0.20
    compound_threshold: float = 0.45
    par_ratio_threshold: float = 0.15       # min par_ratio to consider COMPOUND
    bash_sequential_threshold: float = 0.40 # bash_ratio above → force SEQUENTIAL

    # ── SEQUENTIAL floor by task type ────────────────────────────────────────
    # Task types that always route to SEQUENTIAL, never FINE — even when the
    # prompt score is low. Bug fixes require read→investigate→patch→verify cycles
    # regardless of how simple the prompt looks; FINE's 2-turn cap cuts them off
    # before a patch can be generated.
    sequential_min_task_types: list[str] = field(
        default_factory=lambda: ["bug_fix"]
    )

    # ── COMPOUND gating criteria (all must pass) ──────────────────────────────
    # task_type whitelist: only these types are considered for COMPOUND routing.
    # Empty list = no gating (any task_type may COMPOUND).
    compound_task_type_whitelist: list[str] = field(
        default_factory=lambda: ["refactor", "new_feature"]
    )
    # Minimum distinct files touched to consider COMPOUND.
    # 0 = disabled. Set to 2+ to require multi-file scope.
    compound_min_distinct_files: int = 2
    # Minimum read_before_write_ratio to consider COMPOUND.
    compound_min_rbw_ratio: float = 0.10

    # ── Formula weights (v2) ──────────────────────────────────────────────────
    # Positive weights increase COMPOUND/SEQUENTIAL score.
    # Negative weights decrease score (push toward FINE).
    formula_weights: dict[str, float] = field(default_factory=lambda: {
        "parallelizable_ratio":    0.40,
        "context_load_ratio":      0.20,
        "read_before_write_ratio": 0.20,
        "tool_calls_per_turn":     0.10,
        "write_concentration":    -0.10,
    })

    # ── Task classifier scope keywords ────────────────────────────────────────
    # Prompt keywords that signal multi-file / codebase-wide scope.
    # When matched, score a COMPOUND bonus (not currently in formula — Phase 5).
    scope_keywords: list[str] = field(default_factory=lambda: [
        "across all", "every occurrence", "all callers", "all usages",
        "all files", "entire codebase", "throughout the", "update all",
        "rename all", "replace all", "in all", "across the project",
    ])

    # ── Workload routing (extension primitives) ───────────────────────────────
    # The WorkloadClassifier + RoutingStrategy that map a task to a variant.
    # `classifier`/`routing_strategy` are registry names; `routes` is a list of
    # {when: <class-label>, use: <variant-name>} rules consumed by the rule_based
    # strategy; `route_default` is the fallback variant. None/empty => the
    # framework's default routing (no per-class override). build_routing() in
    # controller.routing_strategies materialises these from the registry.
    classifier: Optional[str] = None
    routing_strategy: str = "rule_based"
    routes: list[dict] = field(default_factory=list)
    route_default: Optional[str] = None

    # ── Quality gate thresholds ───────────────────────────────────────────────
    # Used in Phase 3 quality gate. test_pass_rate + lint_weight×lint_clean.
    quality_lint_weight: float = 0.10
    quality_min_pass_rate: float = 1.0      # must pass all tests to accept routing
    # "code" = CodeQualityGate (default), "binary" = BinaryQualityGate
    quality_gate_mode: str = "code"

    # ── Observability ─────────────────────────────────────────────────────────
    # Log routing decisions to logs/routing_decisions.jsonl for Phase 6 analysis.
    log_routing_decisions: bool = True
    routing_log_path: str = "logs/routing_decisions.jsonl"

    # ── Open questions (tracked here for visibility) ──────────────────────────
    # These fields document what is NOT yet validated and should be tuned in
    # Phase 6. They are not used in routing logic — they are documentation.
    _open: dict[str, str] = field(default_factory=lambda: {
        "bash_sequential_threshold":
            "0.40 is empirically chosen from Phase 2 sweep. Validate in Phase 6 "
            "by measuring token savings at different threshold values.",
        "compound_min_distinct_files":
            "2 is a conservative guess. Real COMPOUND benefit may require 5+. "
            "Validate with synthetic multi-file refactoring tasks in Phase 6.",
        "compound_task_type_whitelist":
            "refactor/new_feature are hypothesised COMPOUND candidates. "
            "bug_fix and explain are assumed SEQUENTIAL/FINE. Validate in Phase 6.",
        "par_ratio_threshold":
            "0.15 is a placeholder — no empirical COMPOUND data yet (Phase 0 had "
            "zero COMPOUND examples). Set based on Phase 6 results.",
    }, repr=False)

    def check_compound_gates(
        self,
        task_type: str,
        n_distinct_files: int,
        read_before_write_ratio: float,
        par_ratio: float,
    ) -> tuple[bool, list[str]]:
        """
        Returns (passes_all_gates, list_of_failed_gate_names).
        All gates must pass for COMPOUND to be considered.
        """
        failed = []

        if self.compound_task_type_whitelist:
            if task_type not in self.compound_task_type_whitelist:
                failed.append(f"task_type={task_type} not in whitelist")

        if self.compound_min_distinct_files > 0:
            if n_distinct_files < self.compound_min_distinct_files:
                failed.append(
                    f"n_distinct_files={n_distinct_files} < {self.compound_min_distinct_files}"
                )

        if read_before_write_ratio < self.compound_min_rbw_ratio:
            failed.append(
                f"rbw_ratio={read_before_write_ratio:.2f} < {self.compound_min_rbw_ratio}"
            )

        if par_ratio < self.par_ratio_threshold:
            failed.append(f"par_ratio={par_ratio:.2f} < {self.par_ratio_threshold}")

        return len(failed) == 0, failed

    def to_dict(self) -> dict:
        return {
            "fine_threshold": self.fine_threshold,
            "compound_threshold": self.compound_threshold,
            "par_ratio_threshold": self.par_ratio_threshold,
            "bash_sequential_threshold": self.bash_sequential_threshold,
            "sequential_min_task_types": self.sequential_min_task_types,
            "compound_task_type_whitelist": self.compound_task_type_whitelist,
            "compound_min_distinct_files": self.compound_min_distinct_files,
            "compound_min_rbw_ratio": self.compound_min_rbw_ratio,
            "formula_weights": self.formula_weights,
            "scope_keywords": self.scope_keywords,
            "classifier": self.classifier,
            "routing_strategy": self.routing_strategy,
            "routes": self.routes,
            "route_default": self.route_default,
            "quality_lint_weight": self.quality_lint_weight,
            "quality_min_pass_rate": self.quality_min_pass_rate,
            "quality_gate_mode": self.quality_gate_mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RoutingConfig":
        return cls(**{k: v for k, v in d.items() if not k.startswith("_")})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "RoutingConfig":
        return cls.from_dict(json.loads(path.read_text()))


# Default singleton — import this directly for non-configurable use.
DEFAULT_CONFIG = RoutingConfig()

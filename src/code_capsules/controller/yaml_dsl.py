"""
YAML policy DSL for RoutingConfig.

Developers configure routing thresholds, weights, and gating criteria in a
plain YAML file without touching Python. Same model as Agentic-Capsules'
policy layer: mechanism (formula.py) is separate from policy (here).

Schema (all keys optional; omitted keys keep RoutingConfig defaults):

    routing:
      fine_threshold: 0.20
      compound_threshold: 0.45
      bash_sequential_threshold: 0.40
      par_ratio_threshold: 0.15

      sequential_min_task_types: [bug_fix]   # always SEQUENTIAL, never FINE

      compound_gate:
        task_type_whitelist: [refactor, new_feature]
        min_distinct_files: 2
        min_rbw_ratio: 0.10

      formula_weights:
        parallelizable_ratio: 0.40
        context_load_ratio: 0.20
        read_before_write_ratio: 0.20
        tool_calls_per_turn: 0.10
        write_concentration: -0.10

      quality_gate:
        mode: python          # python | binary
        lint_weight: 0.10

      # Workload routing (extension primitives): classifier emits a class label,
      # routes map labels -> shipped variant names, default is the fallback.
      classifier: yaml_mapping        # WorkloadClassifier registry name
      routing_strategy: rule_based    # RoutingStrategy registry name
      routes:
        - when: scientific_workload
          use: stuck_signal_injection
      default: two_pass_critique      # fallback variant when no route matches

      scope_keywords:         # replaces defaults
        - across all
        - entire codebase
      scope_keywords_extra:   # merged with defaults
        - all documents
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML is required for YAML DSL support. Install with: pip install pyyaml"
    ) from e

from code_capsules.controller.routing_config import DEFAULT_CONFIG, RoutingConfig


_VALID_GATE_MODES = {"python", "binary", "code"}
_VALID_WEIGHT_KEYS = set(DEFAULT_CONFIG.formula_weights.keys())


class PolicyError(ValueError):
    """Raised when a policy file has invalid content."""


def load_policy(path: Path) -> RoutingConfig:
    """Load a YAML policy file and return a RoutingConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise PolicyError(f"Invalid YAML in {path}: {exc}") from exc
    return _parse(raw, source=str(path))


def load_policy_string(text: str) -> RoutingConfig:
    """Load a YAML policy from a string (for testing and embedding)."""
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise PolicyError(f"Invalid YAML: {exc}") from exc
    return _parse(raw, source="<string>")


def config_to_yaml(cfg: RoutingConfig, path: Path) -> None:
    """Serialise a RoutingConfig to a YAML policy file."""
    doc = {
        "routing": {
            "fine_threshold": cfg.fine_threshold,
            "compound_threshold": cfg.compound_threshold,
            "bash_sequential_threshold": cfg.bash_sequential_threshold,
            "par_ratio_threshold": cfg.par_ratio_threshold,
            "sequential_min_task_types": list(cfg.sequential_min_task_types),
            "compound_gate": {
                "task_type_whitelist": list(cfg.compound_task_type_whitelist),
                "min_distinct_files": cfg.compound_min_distinct_files,
                "min_rbw_ratio": cfg.compound_min_rbw_ratio,
            },
            "formula_weights": dict(cfg.formula_weights),
            "quality_gate": {
                "mode": cfg.quality_gate_mode,
                "lint_weight": cfg.quality_lint_weight,
            },
            "scope_keywords": list(cfg.scope_keywords),
        }
    }
    Path(path).write_text(yaml.dump(doc, default_flow_style=False, sort_keys=False))


# ── Internal parser ───────────────────────────────────────────────────────────

def _parse(raw: dict, source: str) -> RoutingConfig:
    if "routing" not in raw:
        raise PolicyError(f"{source}: top-level 'routing' key is required")

    r: dict[str, Any] = raw["routing"] or {}

    # Start from a full copy of defaults so every unspecified field keeps its value
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # ── Scalar thresholds ─────────────────────────────────────────────────────
    for key in ("fine_threshold", "compound_threshold",
                "bash_sequential_threshold", "par_ratio_threshold"):
        if key in r:
            val = r[key]
            _require_float_in_range(val, 0.0, 1.0, key, source)
            setattr(cfg, key, float(val))

    # Cross-threshold consistency
    if cfg.fine_threshold >= cfg.compound_threshold:
        raise PolicyError(
            f"{source}: fine_threshold ({cfg.fine_threshold}) must be "
            f"< compound_threshold ({cfg.compound_threshold})"
        )

    # ── Sequential floor ──────────────────────────────────────────────────────
    if "sequential_min_task_types" in r:
        cfg.sequential_min_task_types = list(r["sequential_min_task_types"])

    # ── Compound gate ─────────────────────────────────────────────────────────
    if "compound_gate" in r:
        gate = r["compound_gate"] or {}
        if "task_type_whitelist" in gate:
            cfg.compound_task_type_whitelist = list(gate["task_type_whitelist"])
        if "min_distinct_files" in gate:
            cfg.compound_min_distinct_files = int(gate["min_distinct_files"])
        if "min_rbw_ratio" in gate:
            _require_float_in_range(gate["min_rbw_ratio"], 0.0, 1.0,
                                    "compound_gate.min_rbw_ratio", source)
            cfg.compound_min_rbw_ratio = float(gate["min_rbw_ratio"])

    # ── Formula weights ───────────────────────────────────────────────────────
    if "formula_weights" in r:
        weights = r["formula_weights"] or {}
        for k, v in weights.items():
            if k not in _VALID_WEIGHT_KEYS:
                raise PolicyError(
                    f"{source}: formula_weights contains unknown weight key '{k}'. "
                    f"Valid keys: {sorted(_VALID_WEIGHT_KEYS)}"
                )
            cfg.formula_weights[k] = float(v)

    # ── Quality gate ──────────────────────────────────────────────────────────
    if "quality_gate" in r:
        qg = r["quality_gate"] or {}
        if "mode" in qg:
            mode = str(qg["mode"])
            if mode not in _VALID_GATE_MODES:
                raise PolicyError(
                    f"{source}: quality_gate.mode '{mode}' is not valid. "
                    f"Choose from: {sorted(_VALID_GATE_MODES)}"
                )
            cfg.quality_gate_mode = mode
        if "lint_weight" in qg:
            _require_float_in_range(qg["lint_weight"], 0.0, 1.0,
                                    "quality_gate.lint_weight", source)
            cfg.quality_lint_weight = float(qg["lint_weight"])

    # ── Workload routing (extension primitives) ───────────────────────────────
    # classifier / routing_strategy must resolve in the registry; routes must be
    # a list of {when, use} rules. Validated here so a bad name fails at load,
    # not mid-run. Registry import is lazy (only when these keys are present).
    if any(k in r for k in ("classifier", "routing_strategy", "routes", "default")):
        from code_capsules.api import registry as _registry

        if "classifier" in r and r["classifier"] is not None:
            name = str(r["classifier"])
            if name not in _registry.list_registered("classifier"):
                raise PolicyError(
                    f"{source}: routing.classifier '{name}' is not registered. "
                    f"Available: {_registry.list_registered('classifier')}"
                )
            cfg.classifier = name
        if "routing_strategy" in r and r["routing_strategy"] is not None:
            name = str(r["routing_strategy"])
            if name not in _registry.list_registered("routing_strategy"):
                raise PolicyError(
                    f"{source}: routing.routing_strategy '{name}' is not registered. "
                    f"Available: {_registry.list_registered('routing_strategy')}"
                )
            cfg.routing_strategy = name
        if "routes" in r and r["routes"] is not None:
            routes = r["routes"]
            if not isinstance(routes, list) or not all(
                isinstance(x, dict) and "when" in x and "use" in x for x in routes
            ):
                raise PolicyError(
                    f"{source}: routing.routes must be a list of "
                    f"{{when: <class-label>, use: <variant-name>}} rules"
                )
            valid_variants = set(_registry.list_registered("variant"))
            for rule in routes:
                if rule["use"] not in valid_variants:
                    raise PolicyError(
                        f"{source}: routing route use '{rule['use']}' is not a "
                        f"registered variant. Available: {sorted(valid_variants)}"
                    )
            cfg.routes = [dict(x) for x in routes]
        if "default" in r and r["default"] is not None:
            cfg.route_default = str(r["default"])

    # ── Scope keywords ────────────────────────────────────────────────────────
    if "scope_keywords" in r:
        # Full replacement - caller owns the list
        cfg.scope_keywords = list(r["scope_keywords"])
    if "scope_keywords_extra" in r:
        # Additive - merges with whatever scope_keywords currently holds
        cfg.scope_keywords = list(cfg.scope_keywords) + list(r["scope_keywords_extra"])

    return cfg


def _require_float_in_range(
    val: Any, lo: float, hi: float, field: str, source: str
) -> None:
    try:
        fval = float(val)
    except (TypeError, ValueError):
        raise PolicyError(f"{source}: {field} must be a number, got {val!r}")
    if not (lo <= fval <= hi):
        raise PolicyError(
            f"{source}: {field} must be in [{lo}, {hi}], got {fval}"
        )

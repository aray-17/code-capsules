"""
Built-in WorkloadClassifier implementations.

Three shipped defaults:

- GenericPromptClassifier: domain-agnostic heuristics on task text.
  Identifies find-difficulty (named vs symptom), task-type (bugfix vs
  feature vs refactor), iteration-hint (high test count → high iter).
  Works on any text-based task without domain knowledge.

- YamlMappingClassifier: reads a YAML file mapping context-derived keys
  (e.g. repo names) → class labels. Used for benchmark-style workloads
  where the user has a known per-instance mapping. Constructed with no
  argument it loads the shipped SWE-bench mapping
  (code_capsules/config/swe_bench_repo_classes.yaml) and registers under
  the name "yaml_mapping"; users for other domains pass their own table.

- BenchmarkSaturatedClassifier: trivial — always returns
  "saturated_workload". Used for HumanEval / MBPP / simple bug-fix
  workloads where the recommendation is uniformly the cheapest tier.

User-defined classifiers conform to the WorkloadClassifier Protocol.
Domain logic (specific repo names, custom keywords) stays in user code
or config files — NOT in this module.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from code_capsules.api import TaskDescriptor


# ── Regex heuristics (domain-agnostic; work on any text-based task) ──────────

_PY_PATH_RE = re.compile(
    r"\b[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*\.(?:py|js|ts|tsx|jsx|go|rs|java|c|cpp|h|hpp)\b"
)
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]*)+\b")
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)|File \"[^\"]+\", line \d+", re.IGNORECASE)
_BUGFIX_RE = re.compile(r"\bfix(?:es|ed|ing)?\b|\bbug\b|\bbroken?\b|\bcrash(?:es|ed)?\b|\berror\b",
                         re.IGNORECASE)
_REFACTOR_RE = re.compile(r"\brefactor(?:s|ed|ing)?\b|\brestructure|\brename\b|\bextract\b",
                           re.IGNORECASE)
_FEATURE_RE = re.compile(r"\bimplement\b|\badd (?:a |an )?(?:new |support )|"
                           r"\bfeature\b|\bsupport for\b", re.IGNORECASE)


class GenericPromptClassifier:
    """Domain-agnostic classification from task text alone."""

    name = "generic_prompt"

    def classify(self, task: TaskDescriptor) -> str:
        """Returns one of:
          find_named         — issue text names specific files/classes/traceback
          find_symptom       — issue text describes symptoms without naming code
          task_refactor      — task verbs suggest refactor
          task_feature       — task verbs suggest new feature
          task_bugfix        — task verbs suggest bug fix (default fallback)
        """
        text = task.text or ""

        # find_difficulty axis
        has_file_mention = bool(_PY_PATH_RE.search(text))
        has_traceback = bool(_TRACEBACK_RE.search(text))
        camel_hits = len(_CAMEL_RE.findall(text))
        named = has_file_mention or has_traceback or camel_hits >= 4

        # task-type axis (used as secondary label when find is named)
        if _REFACTOR_RE.search(text):
            task_label = "task_refactor"
        elif _FEATURE_RE.search(text):
            task_label = "task_feature"
        elif _BUGFIX_RE.search(text):
            task_label = "task_bugfix"
        else:
            task_label = "task_bugfix"  # default fallback

        # Find difficulty is the primary signal for routing
        if named:
            return "find_named"
        return "find_symptom"


#: The shipped SWE-bench repo→workload-class table (calibration DATA, not code).
#: Lives inside the package so the default classifier works standalone.
SHIPPED_REPO_CLASSES = Path(__file__).resolve().parents[1] / "config" / "swe_bench_repo_classes.yaml"


class YamlMappingClassifier:
    """Reads a YAML file mapping a context key → class label.

    Used for benchmark-style workloads (e.g. SWE-bench) where the user
    knows ahead of time which class each instance belongs to. The YAML
    file shape:

        # code_capsules/config/swe_bench_repo_classes.yaml
        key_field: repo            # which TaskDescriptor.context field to look up
        default: hard_workload     # fallback when key not found
        mapping:
          matplotlib:    visualization_workload
          sympy:         scientific_workload
          ...

    Constructed with no argument it loads the shipped SWE-bench table and
    takes the name ``yaml_mapping`` (this is the registry default that makes
    the ``yaml_mapping`` name from the paper's primitive table resolvable).
    Pass ``yaml_path`` for a custom domain table; ``name`` to register it
    under a distinct key.
    """

    def __init__(self, yaml_path: str | Path | None = None, name: Optional[str] = None):
        import yaml
        path = Path(yaml_path) if yaml_path is not None else SHIPPED_REPO_CLASSES
        if not path.exists():
            raise FileNotFoundError(f"YamlMappingClassifier: {path} not found")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        self.key_field = data.get("key_field", "repo")
        self.default = data.get("default", "hard_workload")
        self.mapping = data.get("mapping", {})
        # Default (shipped) table registers under the literal "yaml_mapping"
        # (the paper's Table-2 selector name); a custom table keeps its stem so
        # multiple domain tables can coexist in the registry.
        if name is not None:
            self.name = name
        elif yaml_path is None:
            self.name = "yaml_mapping"
        else:
            self.name = f"yaml_mapping:{path.stem}"

    def classify(self, task: TaskDescriptor) -> str:
        key = task.context.get(self.key_field)
        if key is None:
            return self.default
        return self.mapping.get(key, self.default)


class BenchmarkSaturatedClassifier:
    """Trivial classifier: always returns 'saturated_workload'.

    Use for benchmarks where the recommended config is uniform across
    all instances (HumanEval, MBPP). Eliminates per-instance classifier
    overhead.
    """

    name = "benchmark_saturated"

    def classify(self, task: TaskDescriptor) -> str:
        return "saturated_workload"


__all__ = [
    "GenericPromptClassifier",
    "YamlMappingClassifier",
    "BenchmarkSaturatedClassifier",
]

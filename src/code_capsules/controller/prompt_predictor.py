"""
Prompt-based DAG topology predictor.

Predicts DAG shape (par_ratio, file count, scope, independent edits) from
the user prompt alone — before any tool calls fire. This is the "predictive"
half of the routing pipeline (Phase 4).

The "reactive" half (observed runtime features) is supplied by the caller as
observed_* arguments; this predictor fills in the gaps before any tool data is
available.

Target: <5ms per prediction, no LLM calls.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum


class Scope(str, Enum):
    LOCAL = "local"        # single file / focused area
    MODULE = "module"      # a handful of related files
    CODEBASE = "codebase"  # project-wide ("all files", "entire codebase")


# ── Compiled pattern sets ─────────────────────────────────────────────────────

# Extracts an explicit file/module/class count:
# "10 files", "5 modules", "8 microservice files", "3 core modules"
_COUNT_PATTERN = re.compile(
    r"\b(\d+)\s+(?:[\w-]+\s+)?(?:files?|modules?|classes?|functions?|services?|components?|tests?)\b",
    re.IGNORECASE,
)

# Single-file signals
_SINGLE_FILE_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bthis file\b", r"\bthe file\b", r"\bthis module\b",
    # Explicit filename references — language-neutral list
    r"\bin [\w./]+" + r"\.(?:py|ts|tsx|js|jsx|go|rs|java|cpp|cc|c|h|hpp|rb|swift|kt|cs|php|scala|ex|exs)\b",
]]

# Codebase-wide scope signals
_CODEBASE_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bacross all\b", r"\ball files\b", r"\bentire codebase\b",
    r"\bthroughout the\b", r"\ball callers\b", r"\ball usages\b",
    r"\ball instances\b", r"\bevery occurrence\b", r"\ball [\w]+ files\b",
    r"\bacross the project\b", r"\bin all\b", r"\bproject.wide\b",
    r"\bupdate all\b", r"\brename all\b", r"\breplace all\b",
]]

# Independent-edit signals (same change applied to N targets)
_INDEPENDENT_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\beach (?:file|module|service|class|function)\b",
    r"\bsame (?:change|transformation|update|pattern|fix)\b",
    r"\bapply (?:the same|this) .{0,30} to (?:each|all|every)\b",
    r"\brename .{0,50} (?:across|in all|throughout)\b",
    r"\bin each\b",
    r"\bto every\b",
]]

# Sequential-dependency signals (edits depend on each other)
_SEQUENTIAL_DEP_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bthen (?:update|change|fix|modify)\b",
    r"\bafter (?:refactor|update|chang)\b",
    r"\bbase class.{0,30}subclass\b",
    r"\bsubclass.{0,30}base class\b",
    r"\bdepend\b",
    r"\bso that\b",
]]

# Module-scope signals
_MODULE_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bthe [\w]+ module\b", r"\bthe [\w]+ package\b",
    r"\bthe [\w]+ service\b",
    r"\bacross [\w]+ (?:files|modules)\b",
    # "each service file", "each module", "each service" → multi-target
    r"\beach (?:file|module|service|class|component|test)\b",
    r"\beach [\w]+ file\b",
]]

# Task types where par_ratio is structurally 0 (always sequential)
_SEQUENTIAL_TASK_TYPES = {"bug_fix", "explain", "test_write", "unknown"}
# Task types that are COMPOUND candidates if scope/independent signals fire
_COMPOUND_TASK_TYPES = {"refactor", "new_feature"}

# Codebase-wide default file count when we can't find an explicit number
_CODEBASE_DEFAULT_FILE_COUNT = 10
# Multi-file (module) default when we detect a handful of files
_MODULE_DEFAULT_FILE_COUNT = 4


@dataclass
class PromptPrediction:
    predicted_file_count: int          # best-effort estimate; 1 = single file
    predicted_scope: Scope             # LOCAL / MODULE / CODEBASE
    predicted_independent_edits: bool  # same transformation applied to N targets
    predicted_par_ratio: float         # estimated [0.0, 1.0]
    predicted_n_distinct_files: int    # same as file_count for pre-routing
    confidence: int                    # number of distinct signals that fired
    signals: list[str] = field(default_factory=list)
    prediction_ms: float = 0.0


def predict_from_prompt(prompt: str, task_type: str) -> PromptPrediction:
    """
    Predict DAG topology from prompt and task_type classifier output.
    No LLM. Target: <5ms.
    """
    t0 = time.monotonic()
    signals: list[str] = []
    confidence = 0

    # ── File count ────────────────────────────────────────────────────────────
    count_match = _COUNT_PATTERN.search(prompt)
    explicit_count: int | None = int(count_match.group(1)) if count_match else None
    if explicit_count is not None:
        signals.append(f"explicit_count:{explicit_count}")
        confidence += 1

    # ── Scope ────────────────────────────────────────────────────────────────
    # Single-file signal is checked first — it has highest specificity.
    is_single = any(pat.search(prompt) for pat in _SINGLE_FILE_PATTERNS)
    is_codebase = not is_single and any(pat.search(prompt) for pat in _CODEBASE_PATTERNS)
    is_module = not is_codebase and not is_single and any(pat.search(prompt) for pat in _MODULE_PATTERNS)

    if is_single or task_type in _SEQUENTIAL_TASK_TYPES:
        scope = Scope.LOCAL
    elif is_codebase:
        scope = Scope.CODEBASE
        signals.append("scope:codebase")
        confidence += 1
    elif is_module:
        scope = Scope.MODULE
        signals.append("scope:module")
        confidence += 1
    elif explicit_count is not None and explicit_count >= 8:
        scope = Scope.CODEBASE
        signals.append("scope:codebase_by_count")
        confidence += 1
    elif explicit_count is not None and explicit_count >= 3:
        scope = Scope.MODULE
        signals.append("scope:module_by_count")
        confidence += 1
    else:
        scope = Scope.LOCAL

    # ── Independent edits ─────────────────────────────────────────────────────
    has_independent = any(pat.search(prompt) for pat in _INDEPENDENT_PATTERNS)
    has_seq_dep = any(pat.search(prompt) for pat in _SEQUENTIAL_DEP_PATTERNS)

    if has_independent and not has_seq_dep and scope != Scope.LOCAL:
        predicted_independent = True
        signals.append("independent_edits")
        confidence += 1
    else:
        predicted_independent = False
        if has_seq_dep:
            signals.append("sequential_dependency")

    # ── File count final estimate ─────────────────────────────────────────────
    if explicit_count is not None:
        file_count = explicit_count
    elif scope == Scope.CODEBASE:
        file_count = _CODEBASE_DEFAULT_FILE_COUNT
    elif scope == Scope.MODULE:
        file_count = _MODULE_DEFAULT_FILE_COUNT
    else:
        file_count = 1

    # ── Par ratio estimate ────────────────────────────────────────────────────
    par_ratio = _estimate_par_ratio(
        task_type, scope, predicted_independent, file_count, signals
    )

    ms = (time.monotonic() - t0) * 1000
    return PromptPrediction(
        predicted_file_count=file_count,
        predicted_scope=scope,
        predicted_independent_edits=predicted_independent,
        predicted_par_ratio=par_ratio,
        predicted_n_distinct_files=file_count,
        confidence=confidence,
        signals=signals,
        prediction_ms=ms,
    )


def _estimate_par_ratio(
    task_type: str,
    scope: Scope,
    independent: bool,
    file_count: int,
    signals: list[str],
) -> float:
    """
    par_ratio = width / (width + depth) from the predicted DAG.
    Conservative estimates — par_ratio is reactive; this is a prior.

    Design logic:
    - Sequential task types (bug_fix, explain): par_ratio = 0.0
    - Single-file (LOCAL): par_ratio = 0.0 (all tools on one file = serial chain)
    - Multi-file + independent edits: par_ratio = file_count/(file_count+1)
      capped at 0.75 (we don't know if Claude will actually batch them)
    - Multi-file + no independent signal: par_ratio = 0.15 (modest parallelism)
    """
    if task_type in _SEQUENTIAL_TASK_TYPES:
        return 0.0

    if scope == Scope.LOCAL:
        return 0.0

    if independent and scope in (Scope.MODULE, Scope.CODEBASE):
        # Estimate: if N files are edited independently, width≈N, depth≈1
        ratio = file_count / (file_count + 1)
        return min(ratio, 0.75)

    if scope == Scope.CODEBASE and not independent:
        return 0.15

    if scope == Scope.MODULE and not independent:
        return 0.10

    return 0.0


def features_from_prediction(
    prediction: PromptPrediction,
    task_type: str,
    overhead_ratio_est: float = 0.85,
    bash_ratio: float = 0.0,
    context_load_ratio: float = 0.0,
    write_concentration: float = 1.0,
    tool_calls_per_turn: float = 0.04,
    n_tool_calls: int = 1,
    read_before_write_ratio: float = 0.0,
    reads_writes_same_file_ratio: float = 0.0,
) -> "Features":
    """
    Build a Features object from a PromptPrediction for pre-routing decisions.

    Callers pass any observed partial features (e.g., bash_ratio known from
    session history); unobserved fields fall back to safe defaults.
    """
    from code_capsules.controller.formula import Features

    # Estimate context_load_ratio from file count: more files = more reads
    if context_load_ratio == 0.0 and prediction.predicted_n_distinct_files > 1:
        n = prediction.predicted_n_distinct_files
        context_load_ratio = min(n / (n + 4), 0.60)

    # Estimate read_before_write from scope and independence
    if read_before_write_ratio == 0.0 and prediction.predicted_independent_edits:
        read_before_write_ratio = 0.80

    # Estimate write_concentration: many files → less concentrated on one file
    if prediction.predicted_n_distinct_files > 1:
        write_concentration = max(
            1.0 / prediction.predicted_n_distinct_files, 0.10
        )

    return Features(
        overhead_ratio_est=overhead_ratio_est,
        parallelizable_ratio=prediction.predicted_par_ratio,
        bash_ratio=bash_ratio,
        context_load_ratio=context_load_ratio,
        write_concentration=write_concentration,
        read_before_write_ratio=read_before_write_ratio,
        tool_calls_per_turn=tool_calls_per_turn,
        n_tool_calls=n_tool_calls,
        n_distinct_files=prediction.predicted_n_distinct_files,
        reads_writes_same_file_ratio=reads_writes_same_file_ratio,
        task_type=task_type,
    )

"""
Keyword-based task type classifier.
No LLM calls. Target: <10ms per classification.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    BUG_FIX = "bug_fix"
    NEW_FEATURE = "new_feature"
    REFACTOR = "refactor"
    TEST_WRITE = "test_write"
    EXPLAIN = "explain"
    UNKNOWN = "unknown"


# Default scope keywords — loaded from RoutingConfig.scope_keywords at classify() time.
# Kept here as fallback for callers that don't pass a config.
_DEFAULT_SCOPE_KEYWORDS: list[str] = [
    r"across all", r"every occurrence", r"all callers", r"all usages",
    r"all files", r"entire codebase", r"throughout the", r"update all",
    r"rename all", r"replace all", r"\bin all\b", r"across the project",
    r"all instances", r"\d+\s+files",
]
_SCOPE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _DEFAULT_SCOPE_KEYWORDS
]


_PATTERNS: list[tuple[TaskType, list[str]]] = [
    (TaskType.BUG_FIX, [
        r"\bfix\b", r"\bbug\b", r"\berror\b", r"\bcrash\b", r"\bfail",
        r"\bbroken\b", r"\bissue\b", r"\bregression\b", r"\bpatch\b",
        r"\bdebug\b", r"\btraceback\b", r"\bexception\b", r"\bwrong\b",
    ]),
    (TaskType.TEST_WRITE, [
        r"\btest\b", r"\btests\b", r"\bunit test", r"\bpytest\b",
        r"\bassert\b", r"\bcoverage\b", r"\bspec\b", r"\btest case",
    ]),
    (TaskType.REFACTOR, [
        r"\brefactor\b", r"\bclean\b", r"\bcleanup\b", r"\bextract\b",
        r"\bmove\b", r"\brename\b", r"\brestructure\b", r"\bsimplif",
        r"\bdedupl", r"\bDRY\b",
    ]),
    (TaskType.EXPLAIN, [
        r"\bexplain\b", r"\bwhat does\b", r"\bhow does\b", r"\bwhat is\b",
        r"\bdescribe\b", r"\bsummariz", r"\bunderstand\b", r"\bwalk me",
        r"\btell me\b", r"\bshow me\b",
    ]),
    (TaskType.NEW_FEATURE, [
        r"\badd\b", r"\bimplement\b", r"\bcreate\b", r"\bbuild\b",
        r"\bnew\b", r"\bfeature\b", r"\bsupport\b", r"\bintegrat",
        r"\bextend\b", r"\benable\b", r"\bwrite a\b",
    ]),
]

_COMPILED: list[tuple[TaskType, list[re.Pattern]]] = [
    (t, [re.compile(p, re.IGNORECASE) for p in patterns])
    for t, patterns in _PATTERNS
]


@dataclass
class ClassificationResult:
    task_type: TaskType
    confidence: int       # number of keyword signals that fired
    classification_ms: float
    multi_file_scope: bool = False  # True if scope keywords detected (COMPOUND signal)


def classify(prompt: str, scope_keywords: list[str] | None = None) -> ClassificationResult:
    """
    scope_keywords: override default scope patterns (from RoutingConfig.scope_keywords).
    Pass None to use the built-in coding-domain defaults.
    """
    t0 = time.monotonic()
    scores: dict[TaskType, int] = {t: 0 for t in TaskType}

    for task_type, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(prompt):
                scores[task_type] += 1

    best_type = max(
        (t for t in TaskType if t != TaskType.UNKNOWN),
        key=lambda t: scores[t],
    )
    best_score = scores[best_type]
    result_type = best_type if best_score > 0 else TaskType.UNKNOWN

    scope_pats = (
        [re.compile(p, re.IGNORECASE) for p in scope_keywords]
        if scope_keywords is not None
        else _SCOPE_PATTERNS
    )
    multi_file_scope = any(pat.search(prompt) for pat in scope_pats)

    ms = (time.monotonic() - t0) * 1000
    return ClassificationResult(
        task_type=result_type,
        confidence=best_score,
        classification_ms=ms,
        multi_file_scope=multi_file_scope,
    )

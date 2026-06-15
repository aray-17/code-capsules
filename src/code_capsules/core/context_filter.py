"""
Phase 8 P8-1: tool-call dependency analyzer + theoretical context-filter savings.

The Phase 7B-Supp result showed prompt-budget signaling cuts cost 44% at parity
quality. Phase 8 explores the second axis of cost savings: cache-token reduction
via topology-aware context filtering (port of Agentic-Capsules' C-7).

For research-task pipelines, AC could declare dependency edges at compile time.
For coding agents the dependency graph is dynamic — the model decides what to
read next at runtime. P8-1 (this module) builds the *retrospective* analyzer:
given a completed session, compute which prior tool outputs were actually
needed for each subsequent action, and how many cache tokens we *could have*
saved by injecting only the dependency closure rather than the full accumulated
context.

The retrospective number is the upper bound on savings any predictive filter
(P8-2) could achieve.

Design choices:
- Rule-based heuristic for v1, no LLM-in-the-loop. Reviewed against real
  session data; tuned thresholds documented.
- Conservative bias: keep more than strictly necessary (false positives are
  cheap; false negatives lose context the model needed).
- Pure functions on `ToolCallRecord` lists. Easy to test on synthetic inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from code_capsules.runtime.stream_parser import ToolCallRecord


# ── Dependency rules ─────────────────────────────────────────────────────────

def compute_dependencies(
    tool_calls: list[ToolCallRecord],
    *,
    recent_context_window: int = 3,
    bash_lookback: int = 10,
) -> dict[int, set[int]]:
    """
    For each tool call index i, return the set of prior indices that tool call
    *depends on* under the v1 heuristic.

    The heuristic combines three rules:

    1. **File locality** — if call i touches file F, it depends on every prior
       call j (j < i) that also touched F. Captures the common pattern of
       Read → Edit → Read (re-read after edit) and Edit → Edit (incremental
       changes to the same file).

    2. **Bash sequencing** — Bash calls (which usually run tests / scripts)
       depend on recent prior writes within `bash_lookback`. The model needs
       to see what was just changed to interpret the test output.

    3. **Recent context window** — the last `recent_context_window` tool calls
       before i are always included as "fresh working memory," even without an
       explicit file or bash dependency.

    The dependency set always excludes i itself.
    """
    deps: dict[int, set[int]] = {}
    for i, tc in enumerate(tool_calls):
        s: set[int] = set()
        # Rule 1: file locality
        if tc.file_path:
            for j, tcj in enumerate(tool_calls[:i]):
                if tcj.file_path == tc.file_path:
                    s.add(j)
        # Rule 2: bash sequencing
        if tc.is_bash:
            lookback_start = max(0, i - bash_lookback)
            for j in range(i - 1, lookback_start - 1, -1):
                if tool_calls[j].is_write:
                    s.add(j)
        # Rule 3: recent context window
        for j in range(max(0, i - recent_context_window), i):
            s.add(j)
        deps[i] = s
    return deps


# ── Theoretical savings ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class FilterSavings:
    """Aggregate metrics over a single session under retrospective context filtering."""
    n_calls: int
    baseline_context_tokens: int   # total tokens fed across all turns if all prior outputs are injected
    filtered_context_tokens: int   # total tokens fed if only dependency-closure outputs are injected
    savings_ratio: float            # 1 - filtered/baseline; 0 = no savings, 1 = total
    per_call_dependency_density: float  # avg fraction of prior calls each call depends on
    longest_chain_savings: int     # max single-call savings (tokens dropped from one call's context)


def compute_theoretical_savings(
    tool_calls: list[ToolCallRecord],
    *,
    recent_context_window: int = 3,
    bash_lookback: int = 10,
) -> FilterSavings:
    """
    Compute theoretical cache-token savings if each turn's prior-output context
    were restricted to the dependency closure.

    The baseline (no filtering) is each call seeing every prior tool output.
    The filtered version is each call seeing only its dependency-closure prior
    outputs. We use `output_size` (in characters) as the token proxy — exact
    token counts would require the model's tokenizer, but character-count
    ratios track token ratios closely for typical English/code outputs.

    For a session with N calls, the baseline context at call i is the sum of
    output_size for calls 0..i-1. The filtered context is the sum over the
    dependency closure of i.
    """
    n = len(tool_calls)
    if n == 0:
        return FilterSavings(0, 0, 0, 0.0, 0.0, 0)

    deps = compute_dependencies(
        tool_calls,
        recent_context_window=recent_context_window,
        bash_lookback=bash_lookback,
    )

    baseline_total = 0
    filtered_total = 0
    longest_chain = 0
    dep_counts: list[int] = []

    for i in range(n):
        baseline_i = sum(tool_calls[j].output_size for j in range(i))
        filtered_i = sum(tool_calls[j].output_size for j in deps[i])
        baseline_total += baseline_i
        filtered_total += filtered_i
        diff = baseline_i - filtered_i
        if diff > longest_chain:
            longest_chain = diff
        dep_counts.append(len(deps[i]))

    savings = (baseline_total - filtered_total) / baseline_total if baseline_total else 0.0
    n_prior_total = sum(i for i in range(n))  # 0 + 1 + ... + n-1
    density = (sum(dep_counts) / n_prior_total) if n_prior_total else 0.0

    return FilterSavings(
        n_calls=n,
        baseline_context_tokens=baseline_total,
        filtered_context_tokens=filtered_total,
        savings_ratio=savings,
        per_call_dependency_density=density,
        longest_chain_savings=longest_chain,
    )


# ── Convenience: aggregate across a list of sessions ─────────────────────────

def aggregate_savings(savings_list: list[FilterSavings]) -> dict:
    """Aggregate FilterSavings across many sessions for headline-number reporting."""
    if not savings_list:
        return {"n_sessions": 0}
    total_baseline = sum(s.baseline_context_tokens for s in savings_list)
    total_filtered = sum(s.filtered_context_tokens for s in savings_list)
    return {
        "n_sessions": len(savings_list),
        "total_baseline_context_tokens": total_baseline,
        "total_filtered_context_tokens": total_filtered,
        "aggregate_savings_ratio": (
            (total_baseline - total_filtered) / total_baseline if total_baseline else 0.0
        ),
        "avg_per_session_savings_ratio": (
            sum(s.savings_ratio for s in savings_list) / len(savings_list)
        ),
        "avg_dependency_density": (
            sum(s.per_call_dependency_density for s in savings_list) / len(savings_list)
        ),
        "max_single_call_savings_chars": max(s.longest_chain_savings for s in savings_list),
    }

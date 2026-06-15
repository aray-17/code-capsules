"""
Phase 7C: runtime quality signals for coding sessions.

These five signals detect when a Claude Code session is *stalling* — making
tool calls but not converging on a fix. They feed the escalation controller
(`controller.escalation`) which decides whether to grant additional turns to
a session that hit its initial budget.

Coding-side analog of Agentic-Capsules' C-2 quality gate and E-1 rolling-mean
quality signal. AC's signals fired on completed-agent quality scores; ours
fire on within-session tool-call patterns because coding sessions are single
LLM calls with multiple turns, not multi-agent pipelines.

All signal functions are pure (no I/O, no global state). They take a list
of `ToolCallRecord` (from `code_capsules.runtime.stream_parser.parse_stream`) and return a
single `QualitySignals` aggregate dataclass.

Design choices:
- Signals are post-hoc (computed on a completed session's tool calls). For
  Phase 7C the harness uses 2-stage escalation: run stage 1 to completion,
  compute signals, decide stage 2. A truly mid-stream variant would need
  per-tool-call hook integration and is deferred.
- Each signal returns a bool, not a score. This keeps the escalation rule
  interpretable ("file_thrash fired") and avoids tuning weights without data.
- Pattern thresholds (e.g. "≥3 reads of the same file in last 5 calls") are
  configurable via function arguments — defaults come from the Phase 7C
  design doc but can be tuned per-deployment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from code_capsules.runtime.stream_parser import ToolCallRecord


@dataclass(frozen=True)
class QualitySignals:
    """Aggregate signal state for a completed session."""
    file_thrash: bool = False           # ≥3 reads of same file in last N tool calls
    test_failure: bool = False          # Bash output contains test-failure markers
    traceback: bool = False             # Tool output contains a Python traceback
    no_progress: bool = False           # ≥N consecutive turns with reads but no writes
    patch_attempt_failed: bool = False  # Write followed by failing test invocation

    @property
    def any(self) -> bool:
        return (self.file_thrash or self.test_failure or self.traceback
                or self.no_progress or self.patch_attempt_failed)

    @property
    def names_fired(self) -> list[str]:
        out = []
        if self.file_thrash:           out.append("file_thrash")
        if self.test_failure:          out.append("test_failure")
        if self.traceback:             out.append("traceback")
        if self.no_progress:           out.append("no_progress")
        if self.patch_attempt_failed:  out.append("patch_attempt_failed")
        return out


# ── Individual signal functions ──────────────────────────────────────────────

def detect_file_thrash(
    tool_calls: list[ToolCallRecord],
    window: int = 5,
    min_reads_same_file: int = 3,
) -> bool:
    """
    Fires when the model re-reads the same file ≥`min_reads_same_file` times
    within the most recent `window` tool calls — indicating it's looping on
    a file it can't make sense of, rather than making progress.
    """
    reads = [tc for tc in tool_calls if tc.is_read and tc.file_path]
    if len(reads) < min_reads_same_file:
        return False
    recent = reads[-window:]
    counts: dict[str, int] = {}
    for tc in recent:
        counts[tc.file_path] = counts.get(tc.file_path, 0) + 1
    return any(c >= min_reads_same_file for c in counts.values())


_TEST_FAIL_PATTERNS = [
    re.compile(r"\bFAILED\b"),
    re.compile(r"\bFAIL\b\s+(test_|tests/)"),    # pytest summary line
    re.compile(r"\bAssertionError\b"),
    re.compile(r"FAILED \([^)]*errors=\d+", re.IGNORECASE),   # django runtests
    re.compile(r"FAILED \([^)]*failures=\d+", re.IGNORECASE),
]


def detect_test_failure(tool_calls: list[ToolCallRecord]) -> bool:
    """
    Fires when a Bash tool call's output indicates a test failure. We look
    only at bash outputs (not Read/Edit), since test runs happen through
    Bash. We match on common failure markers — pytest's `FAILED`, django's
    `FAILED (failures=N)`, raw `AssertionError`.
    """
    for tc in tool_calls:
        if not tc.is_bash or not tc.output_text:
            continue
        for pat in _TEST_FAIL_PATTERNS:
            if pat.search(tc.output_text):
                return True
    return False


_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")


def detect_traceback(tool_calls: list[ToolCallRecord]) -> bool:
    """
    Fires when any tool output contains a Python traceback. This is a
    stronger signal than `test_failure` — tracebacks usually indicate the
    model's patch broke imports or runtime behaviour, not just a failing
    assertion.
    """
    for tc in tool_calls:
        if tc.output_text and _TRACEBACK_RE.search(tc.output_text):
            return True
    return False


def detect_no_progress(
    tool_calls: list[ToolCallRecord],
    min_consecutive_read_turns: int = 4,
) -> bool:
    """
    Fires when the model has spent ≥`min_consecutive_read_turns` consecutive
    turns issuing read-only tool calls (no writes, no bash). Indicates the
    model is stuck exploring without converging on a fix.

    We group tool calls by `turn_index` and check whether the most recent
    consecutive read-only turns exceed the threshold. A turn counts as
    "read-only" if it contains at least one read and zero writes/bash.
    """
    if not tool_calls:
        return False

    # Build per-turn signatures: 'r' (read-only), 'w' (had a write), 'b' (had bash), 'm' (mixed)
    by_turn: dict[int, list[ToolCallRecord]] = {}
    for tc in tool_calls:
        by_turn.setdefault(tc.turn_index, []).append(tc)

    sigs: list[str] = []
    for ti in sorted(by_turn):
        calls = by_turn[ti]
        has_write = any(c.is_write for c in calls)
        has_bash  = any(c.is_bash for c in calls)
        has_read  = any(c.is_read for c in calls)
        if has_write:
            sigs.append("w")
        elif has_bash:
            sigs.append("b")
        elif has_read:
            sigs.append("r")
        else:
            sigs.append("o")    # other tool, not counted as progress or regress

    # Look at the tail — most recent consecutive read-only turns
    tail = 0
    for s in reversed(sigs):
        if s == "r":
            tail += 1
        else:
            break
    return tail >= min_consecutive_read_turns


def detect_patch_attempt_failed(tool_calls: list[ToolCallRecord]) -> bool:
    """
    Fires when a write tool call is followed by a bash tool call whose output
    indicates a test failure. The model attempted a fix, ran tests, and the
    tests still failed — a stronger "this isn't working" signal than a raw
    test failure (which could be the *baseline* failing tests before any fix).
    """
    last_write_index: Optional[int] = None
    for i, tc in enumerate(tool_calls):
        if tc.is_write:
            last_write_index = i
        elif tc.is_bash and last_write_index is not None and i > last_write_index:
            # First bash after a write — check for failure markers
            if tc.output_text:
                for pat in _TEST_FAIL_PATTERNS:
                    if pat.search(tc.output_text):
                        return True
            # Reset: we found a bash after this write. Next pattern needs a new write.
            last_write_index = None
    return False


# ── Top-level aggregator ─────────────────────────────────────────────────────

def compute_signals(
    tool_calls: list[ToolCallRecord],
    *,
    file_thrash_window: int = 5,
    file_thrash_min: int = 3,
    no_progress_turns: int = 4,
) -> QualitySignals:
    """
    Compute all five quality signals from a completed session's tool calls.

    Parameters mirror the individual signal functions' kwargs so operators
    can tune thresholds per-deployment via a single call.
    """
    return QualitySignals(
        file_thrash=detect_file_thrash(
            tool_calls,
            window=file_thrash_window,
            min_reads_same_file=file_thrash_min,
        ),
        test_failure=detect_test_failure(tool_calls),
        traceback=detect_traceback(tool_calls),
        no_progress=detect_no_progress(
            tool_calls,
            min_consecutive_read_turns=no_progress_turns,
        ),
        patch_attempt_failed=detect_patch_attempt_failed(tool_calls),
    )

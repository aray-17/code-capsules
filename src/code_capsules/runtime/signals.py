"""
Built-in Signal implementations.

Five signals shipped, all wrappers around the existing
`controller.runtime_quality_signal.compute_signals` function. Each exposes
a single signal's boolean as a Signal Protocol implementation, allowing
cascade triggers to consume them by name.

Why wrap rather than rewrite: the underlying compute_signals function is
proven (Phase 7C onward, called in run_one_escalating + run_one_injection_loop).
Splitting into per-signal classes is purely an API-conformance refactor —
no logic change.
"""
from __future__ import annotations

from typing import Any

from code_capsules.api import SessionState
from code_capsules.runtime.quality_signal import compute_signals


def _compute(state: SessionState) -> dict[str, bool]:
    """Run compute_signals and return a dict of {name: fired_bool}.

    SessionState.tool_calls is expected to match the shape compute_signals
    expects (list of ToolCallRecord from code_capsules.runtime.stream_parser).
    """
    if not state.tool_calls:
        return {}
    signals = compute_signals(state.tool_calls)
    # QualitySignals dataclass exposes flags + names_fired
    return {
        "cap_pressure": False,  # not derived from tool_calls — see CapPressure below
        "file_thrash": signals.file_thrash,
        "test_failure": signals.test_failure,
        "traceback": signals.traceback,
        "patch_attempt_failed": signals.patch_attempt_failed,
    }


class CapPressure:
    """Fires when the session used a high fraction of its turn budget.

    Cap pressure is a meta-signal: it indicates the model would have
    continued working given more turns. Combined with quality signals
    (file_thrash, test_failure) to decide cascade escalation.

    Threshold is set on the Signal instance (default 0.7).
    """

    name = "cap_pressure"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def compute(self, state: SessionState) -> Any:
        # We need the turn budget to compute cap pressure; conventionally
        # callers stash it in state's extra metadata. Without budget context,
        # return None (caller treats as "unknown — don't fire").
        # In the runtime, the orchestrator sets state.num_turns and the
        # variant config has turn_budget; cascade trigger combines them.
        return state.num_turns  # raw number; trigger does the threshold check


class FileThrash:
    """Fires when the model re-read the same file 3+ times in the last 5 calls.

    Indicates the model is spinning on a file rather than committing to
    an edit. From Iter-C Investigation B: file_thrash + test_failure is
    the signal-pair most predictive of "stage-1 ran out of turns without
    converging" — escalate or inject.
    """

    name = "file_thrash"

    def compute(self, state: SessionState) -> Any:
        return _compute(state).get("file_thrash", False)


class TestFailure:
    """Fires when the session emitted test output with failures.

    Indicates the model attempted a patch but tests don't pass yet —
    typical escalation candidate (more turns might let the model refine).
    """

    name = "test_failure"

    def compute(self, state: SessionState) -> Any:
        return _compute(state).get("test_failure", False)


class Traceback:
    """Fires when the session emitted a traceback in tool output.

    Indicates an exception during model exploration — useful for
    targeted injection (V2 mechanism: parse traceback file:line, inject).
    """

    name = "traceback"

    def compute(self, state: SessionState) -> Any:
        return _compute(state).get("traceback", False)


class PatchAttemptFailed:
    """Fires when the model produced a patch but git apply failed.

    Most surgical V2-injection candidate: the diff fragment can be passed
    directly to the next stage with "your patch had a conflict here; refine."
    """

    name = "patch_attempt_failed"

    def compute(self, state: SessionState) -> Any:
        return _compute(state).get("patch_attempt_failed", False)


__all__ = [
    "CapPressure",
    "FileThrash",
    "TestFailure",
    "Traceback",
    "PatchAttemptFailed",
]

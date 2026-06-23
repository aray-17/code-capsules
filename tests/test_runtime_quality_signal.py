"""Tests for controller/runtime_quality_signal.py - signal detection."""
import pytest
from code_capsules.runtime.stream_parser import ToolCallRecord
from code_capsules.runtime.quality_signal import (
    QualitySignals,
    compute_signals,
    detect_file_thrash,
    detect_test_failure,
    detect_traceback,
    detect_no_progress,
    detect_patch_attempt_failed,
)


# ── Fixture helpers ──────────────────────────────────────────────────────────

def _tc(name, file_path=None, turn=0, output=""):
    return ToolCallRecord(
        name=name,
        file_path=file_path,
        input_size=100,
        output_size=len(output),
        turn_index=turn,
        input_tokens_this_turn=1000,
        output_tokens_this_turn=500,
        output_text=output,
    )


# ── detect_file_thrash ───────────────────────────────────────────────────────

class TestFileThrash:
    def test_three_reads_same_file_fires(self):
        calls = [
            _tc("Read", file_path="src/a.py", turn=0),
            _tc("Read", file_path="src/a.py", turn=1),
            _tc("Read", file_path="src/a.py", turn=2),
        ]
        assert detect_file_thrash(calls) is True

    def test_two_reads_same_file_does_not_fire(self):
        calls = [
            _tc("Read", file_path="src/a.py", turn=0),
            _tc("Read", file_path="src/a.py", turn=1),
        ]
        assert detect_file_thrash(calls) is False

    def test_three_reads_different_files_does_not_fire(self):
        calls = [
            _tc("Read", file_path="a.py", turn=0),
            _tc("Read", file_path="b.py", turn=1),
            _tc("Read", file_path="c.py", turn=2),
        ]
        assert detect_file_thrash(calls) is False

    def test_writes_dont_count_as_reads(self):
        calls = [
            _tc("Write", file_path="src/a.py", turn=0),
            _tc("Write", file_path="src/a.py", turn=1),
            _tc("Read", file_path="src/a.py", turn=2),
        ]
        assert detect_file_thrash(calls) is False

    def test_only_recent_window_counts(self):
        # Old re-reads outside the window should not trigger
        calls = [
            _tc("Read", file_path="a.py", turn=0),     # outside window
            _tc("Read", file_path="b.py", turn=1),
            _tc("Read", file_path="c.py", turn=2),
            _tc("Read", file_path="d.py", turn=3),
            _tc("Read", file_path="e.py", turn=4),
            _tc("Read", file_path="f.py", turn=5),     # window: last 5 reads (b..f)
        ]
        # window=5, only b..f considered, no repeats
        assert detect_file_thrash(calls, window=5) is False

    def test_custom_threshold(self):
        # min_reads_same_file=2 is much more sensitive
        calls = [_tc("Read", file_path="a.py", turn=0),
                 _tc("Read", file_path="a.py", turn=1)]
        assert detect_file_thrash(calls, min_reads_same_file=2) is True
        assert detect_file_thrash(calls, min_reads_same_file=3) is False

    def test_empty_calls(self):
        assert detect_file_thrash([]) is False


# ── detect_test_failure ──────────────────────────────────────────────────────

class TestTestFailure:
    def test_pytest_failed_marker(self):
        calls = [_tc("Bash", output="test_foo.py::test_one FAILED\n")]
        assert detect_test_failure(calls) is True

    def test_django_failures(self):
        calls = [_tc("Bash", output="FAILED (failures=2)")]
        assert detect_test_failure(calls) is True

    def test_django_errors(self):
        calls = [_tc("Bash", output="FAILED (errors=1)")]
        assert detect_test_failure(calls) is True

    def test_assertion_error(self):
        calls = [_tc("Bash", output="    AssertionError: expected 1, got 2")]
        assert detect_test_failure(calls) is True

    def test_passing_test_does_not_fire(self):
        calls = [_tc("Bash", output="test_foo.py::test_one PASSED\n5 passed in 0.5s")]
        assert detect_test_failure(calls) is False

    def test_only_bash_outputs_inspected(self):
        # A Read tool returning text containing 'FAILED' should not fire - only Bash
        calls = [_tc("Read", file_path="x.py", output="FAILED\nAssertionError")]
        assert detect_test_failure(calls) is False

    def test_no_bash_calls(self):
        calls = [_tc("Read", file_path="a.py")]
        assert detect_test_failure(calls) is False


# ── detect_traceback ─────────────────────────────────────────────────────────

class TestTraceback:
    def test_python_traceback_fires(self):
        calls = [_tc("Bash", output="Traceback (most recent call last):\n  File ...")]
        assert detect_traceback(calls) is True

    def test_traceback_in_read_output_also_fires(self):
        # Tracebacks anywhere are a problem - even if discovered by reading a log
        calls = [_tc("Read", file_path="error.log",
                     output="Traceback (most recent call last):\n  ZeroDivisionError")]
        assert detect_traceback(calls) is True

    def test_no_traceback_does_not_fire(self):
        calls = [_tc("Bash", output="all green!")]
        assert detect_traceback(calls) is False

    def test_partial_word_does_not_match(self):
        calls = [_tc("Bash", output="traceback information requested")]   # lowercase, different phrasing
        assert detect_traceback(calls) is False


# ── detect_no_progress ───────────────────────────────────────────────────────

class TestNoProgress:
    def test_four_consecutive_read_turns_fires(self):
        calls = [
            _tc("Read", file_path="a.py", turn=0),
            _tc("Read", file_path="b.py", turn=1),
            _tc("Read", file_path="c.py", turn=2),
            _tc("Read", file_path="d.py", turn=3),
        ]
        assert detect_no_progress(calls) is True

    def test_three_consecutive_does_not_fire(self):
        calls = [
            _tc("Read", file_path="a.py", turn=0),
            _tc("Read", file_path="b.py", turn=1),
            _tc("Read", file_path="c.py", turn=2),
        ]
        assert detect_no_progress(calls) is False

    def test_write_interrupts_streak(self):
        # Write at turn 1 breaks the streak; only 2 read-turns at tail
        calls = [
            _tc("Read", file_path="a.py", turn=0),
            _tc("Write", file_path="a.py", turn=1),
            _tc("Read", file_path="b.py", turn=2),
            _tc("Read", file_path="c.py", turn=3),
        ]
        assert detect_no_progress(calls) is False

    def test_bash_interrupts_streak(self):
        # Bash also counts as progress (running tests etc.)
        calls = [
            _tc("Read", file_path="a.py", turn=0),
            _tc("Read", file_path="b.py", turn=1),
            _tc("Bash", output="pytest", turn=2),
            _tc("Read", file_path="d.py", turn=3),
            _tc("Read", file_path="e.py", turn=4),
        ]
        assert detect_no_progress(calls) is False

    def test_only_tail_counts(self):
        # 4 reads early on, then a write, then 1 read - should NOT fire on the tail
        calls = [
            _tc("Read", file_path="a.py", turn=0),
            _tc("Read", file_path="b.py", turn=1),
            _tc("Read", file_path="c.py", turn=2),
            _tc("Read", file_path="d.py", turn=3),
            _tc("Write", file_path="a.py", turn=4),
            _tc("Read", file_path="e.py", turn=5),
        ]
        assert detect_no_progress(calls) is False

    def test_mixed_read_and_write_in_one_turn_counts_as_write(self):
        # A single turn containing both a read and a write should not count as read-only
        calls = [
            _tc("Read", file_path="a.py", turn=0),
            _tc("Read", file_path="b.py", turn=1),
            _tc("Read", file_path="c.py", turn=2),
            _tc("Read", file_path="d.py", turn=3),
            _tc("Write", file_path="x.py", turn=3),    # same turn as the read
        ]
        # Tail turn (3) has a write → streak is 3, not 4
        assert detect_no_progress(calls) is False


# ── detect_patch_attempt_failed ──────────────────────────────────────────────

class TestPatchAttemptFailed:
    def test_write_then_failing_bash_fires(self):
        calls = [
            _tc("Read", file_path="x.py", turn=0),
            _tc("Write", file_path="x.py", turn=1),
            _tc("Bash", output="test_x.py::test_one FAILED", turn=2),
        ]
        assert detect_patch_attempt_failed(calls) is True

    def test_write_then_passing_bash_does_not_fire(self):
        calls = [
            _tc("Write", file_path="x.py", turn=0),
            _tc("Bash", output="5 passed in 0.1s", turn=1),
        ]
        assert detect_patch_attempt_failed(calls) is False

    def test_bash_failure_before_any_write_does_not_fire(self):
        # The baseline (FAIL_TO_PASS) tests fail before any fix is attempted - 
        # don't treat that as a patch-attempt failure.
        calls = [
            _tc("Read", file_path="x.py", turn=0),
            _tc("Bash", output="test_x.py FAILED", turn=1),
        ]
        assert detect_patch_attempt_failed(calls) is False

    def test_write_with_traceback_fires(self):
        # AssertionError in bash output also triggers
        calls = [
            _tc("Write", file_path="x.py", turn=0),
            _tc("Bash", output="AssertionError: foo != bar", turn=1),
        ]
        assert detect_patch_attempt_failed(calls) is True


# ── compute_signals end-to-end ───────────────────────────────────────────────

class TestComputeSignals:
    def test_clean_session_no_signals(self):
        calls = [
            _tc("Read", file_path="a.py", turn=0),
            _tc("Write", file_path="a.py", turn=1),
            _tc("Bash", output="5 passed in 0.1s", turn=2),
        ]
        sigs = compute_signals(calls)
        assert sigs.any is False
        assert sigs.names_fired == []

    def test_stalling_session_fires_multiple(self):
        # Reads thrashing on same file, no writes, traceback in output
        calls = [
            _tc("Read", file_path="x.py", turn=0, output="(no traceback)"),
            _tc("Read", file_path="x.py", turn=1),
            _tc("Read", file_path="x.py", turn=2),
            _tc("Read", file_path="y.py", turn=3, output="Traceback (most recent call last)"),
        ]
        sigs = compute_signals(calls)
        assert sigs.file_thrash is True
        assert sigs.traceback is True
        assert sigs.no_progress is True       # 4 consecutive read-only turns
        assert sigs.any is True
        assert set(sigs.names_fired) >= {"file_thrash", "traceback", "no_progress"}

    def test_empty_session_no_signals(self):
        sigs = compute_signals([])
        assert sigs.any is False

    def test_quality_signals_is_immutable(self):
        s = QualitySignals(file_thrash=True)
        with pytest.raises((AttributeError, Exception)):
            s.file_thrash = False

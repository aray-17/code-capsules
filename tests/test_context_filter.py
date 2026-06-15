"""Tests for controller/context_filter.py — Phase 8 dependency analyzer."""
import pytest
from code_capsules.runtime.stream_parser import ToolCallRecord
from code_capsules.core.context_filter import (
    compute_dependencies,
    compute_theoretical_savings,
    aggregate_savings,
    FilterSavings,
)


def _tc(name, file_path=None, turn=0, output_size=100):
    return ToolCallRecord(
        name=name, file_path=file_path,
        input_size=50, output_size=output_size,
        turn_index=turn, input_tokens_this_turn=500, output_tokens_this_turn=200,
        output_text="",
    )


# ── compute_dependencies ─────────────────────────────────────────────────────

class TestDependencies:
    def test_empty_session(self):
        assert compute_dependencies([]) == {}

    def test_recent_window_default_3(self):
        # 5 reads on different files; each call's deps should include the prior 3 (window=3)
        calls = [_tc("Read", file_path=f"f{i}.py", turn=i) for i in range(5)]
        deps = compute_dependencies(calls)
        assert deps[0] == set()
        assert deps[1] == {0}
        assert deps[2] == {0, 1}
        assert deps[3] == {0, 1, 2}
        assert deps[4] == {1, 2, 3}   # window=3 drops the oldest

    def test_file_locality_picks_up_distant_dependency(self):
        # Read foo.py at turn 0, then 10 unrelated reads, then Edit foo.py at turn 11
        # Edit should depend on the distant Read foo.py via the file-locality rule.
        calls = [_tc("Read", file_path="foo.py", turn=0)]
        for i in range(1, 11):
            calls.append(_tc("Read", file_path=f"bar{i}.py", turn=i))
        calls.append(_tc("Edit", file_path="foo.py", turn=11))
        deps = compute_dependencies(calls, recent_context_window=3)
        # Edit foo.py (index 11) depends on Read foo.py (index 0) via file locality
        # AND on the last 3 calls (indices 8, 9, 10) via the recent window
        assert 0 in deps[11], "file locality should pull in distant same-file read"
        assert 8 in deps[11] and 9 in deps[11] and 10 in deps[11]

    def test_bash_depends_on_recent_writes(self):
        # Edit at turn 0, then 2 unrelated reads, then Bash at turn 3
        # Bash should depend on the prior Edit via the bash_lookback rule.
        calls = [
            _tc("Edit", file_path="src.py", turn=0),
            _tc("Read", file_path="a.py", turn=1),
            _tc("Read", file_path="b.py", turn=2),
            _tc("Bash", turn=3),
        ]
        deps = compute_dependencies(calls)
        assert 0 in deps[3], "Bash should depend on recent Edit"

    def test_bash_lookback_drops_old_writes(self):
        # Edit at turn 0, then 15 reads, then Bash at turn 16 with lookback=10
        calls = [_tc("Edit", file_path="src.py", turn=0)]
        for i in range(1, 16):
            calls.append(_tc("Read", file_path=f"f{i}.py", turn=i))
        calls.append(_tc("Bash", turn=16))
        deps = compute_dependencies(calls, bash_lookback=10)
        # Edit was at index 0, but lookback only goes back 10 → drops it
        # (Recent window picks up indices 13, 14, 15)
        assert 0 not in deps[16], "Bash should not pull in Edit beyond lookback window"

    def test_no_self_dependency(self):
        calls = [_tc("Read", file_path="a.py")]
        deps = compute_dependencies(calls)
        assert 0 not in deps[0]


# ── compute_theoretical_savings ──────────────────────────────────────────────

class TestTheoreticalSavings:
    def test_empty_session(self):
        s = compute_theoretical_savings([])
        assert s.n_calls == 0
        assert s.savings_ratio == 0.0

    def test_short_session_within_window_has_no_savings(self):
        # 3 calls, window=3 → every call already in deps → no filtering
        calls = [_tc("Read", file_path=f"f{i}.py", turn=i, output_size=100) for i in range(3)]
        s = compute_theoretical_savings(calls, recent_context_window=3)
        assert s.savings_ratio == 0.0

    def test_long_session_drops_old_unrelated_calls(self):
        # 10 reads of different files, window=3
        # Late calls baseline includes all 9 prior; filtered only includes last 3
        calls = [_tc("Read", file_path=f"f{i}.py", turn=i, output_size=100) for i in range(10)]
        s = compute_theoretical_savings(calls, recent_context_window=3)
        assert s.savings_ratio > 0
        # Sanity check: at index 9, baseline=9*100=900, filtered=3*100=300 → drop 600
        # Earlier indices have less savings; aggregate should be ~50%
        assert 0.2 < s.savings_ratio < 0.8

    def test_file_locality_reduces_savings(self):
        # 10 reads of THE SAME file → file-locality keeps everything → small savings
        calls = [_tc("Read", file_path="same.py", turn=i, output_size=100) for i in range(10)]
        s_local = compute_theoretical_savings(calls, recent_context_window=3)
        # All-different-files case
        calls_diff = [_tc("Read", file_path=f"f{i}.py", turn=i, output_size=100) for i in range(10)]
        s_diff = compute_theoretical_savings(calls_diff, recent_context_window=3)
        # Same-file should have LESS savings (file-locality keeps all)
        assert s_local.savings_ratio < s_diff.savings_ratio

    def test_savings_reflect_realistic_session(self):
        # Simulate: 4 reads of different files, 1 edit, 1 bash, more reads
        # Roughly mirrors a SWE-bench-like session
        calls = [
            _tc("Read", file_path="src/foo.py", turn=0, output_size=2000),
            _tc("Read", file_path="src/bar.py", turn=1, output_size=1500),
            _tc("Read", file_path="src/baz.py", turn=2, output_size=1200),
            _tc("Read", file_path="src/qux.py", turn=3, output_size=800),
            _tc("Edit", file_path="src/foo.py", turn=4, output_size=300),
            _tc("Bash", turn=5, output_size=400),
            _tc("Read", file_path="src/foo.py", turn=6, output_size=2000),
            _tc("Edit", file_path="src/foo.py", turn=7, output_size=300),
        ]
        s = compute_theoretical_savings(calls)
        assert s.n_calls == 8
        assert s.baseline_context_tokens > s.filtered_context_tokens
        # File-locality picks up foo.py at index 0 from index 4, 6, 7
        # But bar/baz/qux drop out of later calls' context

    def test_returns_filtersavings_dataclass(self):
        s = compute_theoretical_savings([_tc("Read", "a.py")])
        assert isinstance(s, FilterSavings)
        with pytest.raises((AttributeError, Exception)):
            s.savings_ratio = 0.99   # frozen


# ── aggregate_savings ────────────────────────────────────────────────────────

class TestAggregateSavings:
    def test_empty(self):
        a = aggregate_savings([])
        assert a["n_sessions"] == 0

    def test_aggregates_across_sessions(self):
        s1 = FilterSavings(n_calls=10, baseline_context_tokens=1000,
                          filtered_context_tokens=500, savings_ratio=0.5,
                          per_call_dependency_density=0.3, longest_chain_savings=200)
        s2 = FilterSavings(n_calls=20, baseline_context_tokens=4000,
                          filtered_context_tokens=1000, savings_ratio=0.75,
                          per_call_dependency_density=0.25, longest_chain_savings=600)
        a = aggregate_savings([s1, s2])
        assert a["n_sessions"] == 2
        assert a["total_baseline_context_tokens"] == 5000
        assert a["total_filtered_context_tokens"] == 1500
        assert a["aggregate_savings_ratio"] == pytest.approx(0.7)
        assert a["avg_per_session_savings_ratio"] == pytest.approx(0.625)
        assert a["max_single_call_savings_chars"] == 600

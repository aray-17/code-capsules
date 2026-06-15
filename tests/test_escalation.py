"""Tests for controller/escalation.py — Phase 7C escalation decision."""
import pytest
from code_capsules.controller.escalation import EscalationDecision, should_escalate
from code_capsules.runtime.quality_signal import QualitySignals


# ── Fixture helpers ──────────────────────────────────────────────────────────

def _no_signals():
    return QualitySignals()


def _all_signals():
    return QualitySignals(
        file_thrash=True, test_failure=True, traceback=True,
        no_progress=True, patch_attempt_failed=True,
    )


def _one_signal(name="file_thrash"):
    return QualitySignals(**{name: True})


# ── Don't-escalate branches ──────────────────────────────────────────────────

class TestDontEscalate:
    def test_resolved_session_no_escalation(self):
        d = should_escalate(
            resolved=True,
            actual_turns=10,
            current_budget=10,
            signals=_all_signals(),   # signals fired but session resolved
        )
        assert d.escalate is False
        assert "resolved" in d.reason.lower()
        assert d.next_budget is None

    def test_already_at_target_budget(self):
        d = should_escalate(
            resolved=False,
            actual_turns=20,
            current_budget=20,        # already at target
            signals=_all_signals(),
            target_budget=20,
        )
        assert d.escalate is False
        assert "target" in d.reason.lower() or "ceiling" in d.reason.lower()

    def test_low_cap_pressure_no_escalation(self):
        # Used only 5 of 10 turns — self-terminated early, signals notwithstanding
        d = should_escalate(
            resolved=False,
            actual_turns=5,
            current_budget=10,
            signals=_all_signals(),    # even with signals, low pressure → no escalate
            cap_pressure_threshold=0.7,
        )
        assert d.escalate is False
        assert "pressure" in d.reason.lower() or "self-terminated" in d.reason.lower()

    def test_cap_hit_but_no_signals_clean_giveup(self):
        # Used all 10 turns but signals are quiet — model is "stuck cleanly"
        d = should_escalate(
            resolved=False,
            actual_turns=10,
            current_budget=10,
            signals=_no_signals(),
        )
        assert d.escalate is False
        assert "no quality signals" in d.reason.lower() or "give-up" in d.reason.lower()


# ── Escalate branches ────────────────────────────────────────────────────────

class TestEscalate:
    def test_cap_hit_with_signal_escalates(self):
        d = should_escalate(
            resolved=False,
            actual_turns=10,
            current_budget=10,
            signals=_one_signal("file_thrash"),
        )
        assert d.escalate is True
        assert d.next_budget == 20
        assert d.next_prompt_budget_hint == 20
        assert "file_thrash" in d.reason

    def test_high_pressure_with_signal_escalates(self):
        # 7 of 10 turns used (exactly at threshold)
        d = should_escalate(
            resolved=False,
            actual_turns=7,
            current_budget=10,
            signals=_one_signal("traceback"),
        )
        assert d.escalate is True

    def test_custom_target_budget(self):
        # Operator can override target
        d = should_escalate(
            resolved=False, actual_turns=10, current_budget=10,
            signals=_one_signal("test_failure"), target_budget=15,
        )
        assert d.escalate is True
        assert d.next_budget == 15
        assert d.next_prompt_budget_hint == 15

    def test_reason_lists_all_fired_signals(self):
        d = should_escalate(
            resolved=False, actual_turns=10, current_budget=10,
            signals=QualitySignals(file_thrash=True, test_failure=True),
        )
        assert d.escalate is True
        assert "file_thrash" in d.reason
        assert "test_failure" in d.reason


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_budget_does_not_crash(self):
        # Pathological: budget=0 means actual_turns/budget would divide by zero
        d = should_escalate(
            resolved=False, actual_turns=0, current_budget=0,
            signals=_all_signals(),
        )
        assert isinstance(d, EscalationDecision)
        # Should not escalate at budget=0 (no meaningful escalation target)

    def test_actual_turns_above_budget(self):
        # Claude can self-report turns slightly above the cap (cap+1 is common)
        d = should_escalate(
            resolved=False, actual_turns=11, current_budget=10,
            signals=_one_signal("traceback"),
        )
        assert d.escalate is True

    def test_decision_is_immutable(self):
        d = EscalationDecision(escalate=False, reason="x")
        with pytest.raises((AttributeError, Exception)):
            d.escalate = True


# ── Threshold tunability ─────────────────────────────────────────────────────

class TestThresholdTuning:
    def test_lower_pressure_threshold_lets_more_through(self):
        # 50% pressure with signals: blocked at default (0.7), allowed at 0.3
        signals = _one_signal()
        blocked = should_escalate(
            resolved=False, actual_turns=5, current_budget=10,
            signals=signals, cap_pressure_threshold=0.7,
        )
        allowed = should_escalate(
            resolved=False, actual_turns=5, current_budget=10,
            signals=signals, cap_pressure_threshold=0.3,
        )
        assert blocked.escalate is False
        assert allowed.escalate is True

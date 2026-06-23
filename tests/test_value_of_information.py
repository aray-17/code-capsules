"""Tests for the VOI escalation formula (design §1), calibrated on the stage-1 grade study."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from code_capsules.controller.value_of_information import (  # noqa: E402
    expected_marginal_value, voi_escalate, voi_escalate_for_grade,
)
from code_capsules.controller.verifier import NOPATCH, FLAT, BROKEN, PARTIAL  # noqa: E402


def test_emv_and_breakeven():
    # EMV = p*V - C ; escalate iff EMV>0 iff p > C/V
    assert expected_marginal_value(0.5, 10.0, 3.0) == 2.0
    d = voi_escalate(p_resolve=0.5, value=10.0, cost=3.0)
    assert d.escalate is True and abs(d.breakeven_p - 0.3) < 1e-9
    d2 = voi_escalate(p_resolve=0.2, value=10.0, cost=3.0)
    assert d2.escalate is False  # 0.2 < 0.3 break-even


def test_grade_belief_reproduces_exp1_gate():
    """With a C/V break-even between FLAT's p (0) and NOPATCH's p (0.435),
    the formula reproduces the asymmetric gate: escalate NOPATCH,
    drop FLAT/PARTIAL. The C/V ratio is the principled version of the
    abandon-set knob."""
    V, C = 1.0, 0.25          # break-even p = 0.25
    assert voi_escalate_for_grade(NOPATCH, V, C).escalate is True   # 0.435 > 0.25
    assert voi_escalate_for_grade(FLAT, V, C).escalate is False     # 0.0   < 0.25
    assert voi_escalate_for_grade(PARTIAL, V, C).escalate is False  # 0.0   < 0.25
    assert voi_escalate_for_grade(BROKEN, V, C).escalate is False   # 0.143 < 0.25


def test_cost_tolerance_knob():
    """Lower C/V (cheaper escalation / more valuable resolve) escalates more
    grades; raise it and even NOPATCH stops paying."""
    # generous: break-even 0.10 -> NOPATCH(0.435) and BROKEN(0.143) escalate
    assert voi_escalate_for_grade(BROKEN, value=1.0, cost=0.10).escalate is True
    # stingy: break-even 0.50 -> even NOPATCH(0.435) stops
    assert voi_escalate_for_grade(NOPATCH, value=1.0, cost=0.50).escalate is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} VOI tests passed")

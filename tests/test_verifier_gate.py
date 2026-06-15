"""Unit tests for the execution verifier + VerifierGate (the EXP-1 cost lever).

Fast, no Docker. Validates that:
  - grade_execution maps execution counts to the right coarse grade,
  - the SWE-bench adapter parses docker_eval output (pytest + django/sympy
    runtests styles) to the right grade,
  - VerifierGate reproduces the EXP-1-validated asymmetric policy: stop
    RESOLVED, drop the confident-doomed tail, escalate the ambiguous middle.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from code_capsules.controller.verifier import (  # noqa: E402
    ExecutionReading, grade_execution, from_swebench_eval,
    RESOLVED, PARTIAL, FLAT, BROKEN, NOPATCH, UNKNOWN,
)
from code_capsules.controller.cascade_triggers import VerifierGate  # noqa: E402


def _r(has_patch=True, applied=True, n_targets=2, n_pass=0, n_error=0, resolved=False):
    return ExecutionReading(has_patch, applied, n_targets, n_pass, n_error, resolved)


def test_grade_execution_classes():
    assert grade_execution(_r(has_patch=False)) == NOPATCH
    assert grade_execution(_r(resolved=True)) == RESOLVED
    assert grade_execution(_r(applied=False)) == BROKEN
    assert grade_execution(_r(n_error=3, n_pass=0)) == BROKEN
    assert grade_execution(_r(n_targets=2, n_pass=1)) == PARTIAL
    assert grade_execution(_r(n_targets=2, n_pass=0)) == FLAT
    assert grade_execution(_r(n_targets=0)) == UNKNOWN


def test_from_swebench_eval_adapter():
    assert from_swebench_eval({"resolved": True, "stdout": "2 passed"}, 2) == RESOLVED
    assert from_swebench_eval({"resolved": False, "note": "empty patch", "stdout": ""}, 1) == NOPATCH
    assert from_swebench_eval({"resolved": False, "stdout": "ImportError: x\nFAILED (errors=3)"}, 3) == BROKEN
    assert from_swebench_eval({"resolved": False, "stdout": "Ran 5 tests\nFAILED (failures=5)"}, 5) == FLAT
    assert from_swebench_eval({"resolved": False, "stdout": "Ran 5 tests\nFAILED (failures=3)"}, 5) == PARTIAL
    assert from_swebench_eval({"resolved": False, "stdout": "3 passed, 1 failed in 0.2s"}, 4) == PARTIAL


def test_verifier_gate_acts_on_tails_only():
    g = VerifierGate()  # default abandon = {FLAT}
    # stop-for-success
    assert g.should_escalate({"verifier_grade": RESOLVED}, 1) is False
    assert g.should_escalate({"resolved": True}, 1) is False
    # confident-doomed tail -> drop
    assert g.should_escalate({"verifier_grade": FLAT}, 1) is False
    # ambiguous middle -> escalate (NOPATCH is 43% winnable per EXP-1)
    assert g.should_escalate({"verifier_grade": NOPATCH}, 1) is True
    assert g.should_escalate({"verifier_grade": PARTIAL}, 1) is True
    assert g.should_escalate({"verifier_grade": BROKEN}, 1) is True  # not in default abandon
    # tier cap respected
    assert g.should_escalate({"verifier_grade": NOPATCH}, 2) is False
    # no reading -> safe fallback (escalate when unresolved)
    assert g.should_escalate({}, 1) is True


def test_verifier_gate_aggressive_abandon():
    g = VerifierGate(abandon=("FLAT", "BROKEN"))
    assert g.should_escalate({"verifier_grade": BROKEN}, 1) is False  # now dropped
    assert g.should_escalate({"verifier_grade": NOPATCH}, 1) is True


def test_exp1_policy_keeps_converters():
    """Reproduce EXP-1: drop-FLAT + stop-RESOLVED loses ZERO converters.

    EXP-1 cross-tab (n=50 django): RESOLVED 6/0, PARTIAL 0/3, FLAT 0/5,
    BROKEN 1/6, NOPATCH 10/13. A converter is an instance that eventually
    resolves. The default gate must escalate (or stop-as-resolved) every
    converter -- never drop one.
    """
    g = VerifierGate()  # abandon = {FLAT}
    # (grade, eventually_resolved) from EXP-1 distribution
    cohort = (
        [(RESOLVED, True)] * 6
        + [(PARTIAL, False)] * 3
        + [(FLAT, False)] * 5
        + [(BROKEN, True)] * 1 + [(BROKEN, False)] * 6
        + [(NOPATCH, True)] * 10 + [(NOPATCH, False)] * 13
    )
    converters_dropped = 0
    for grade, resolved in cohort:
        escalate = g.should_escalate({"verifier_grade": grade}, current_tier=1)
        stopped_as_success = grade == RESOLVED
        kept = escalate or stopped_as_success
        if resolved and not kept:
            converters_dropped += 1
    assert converters_dropped == 0, f"gate dropped {converters_dropped} converters"


if __name__ == "__main__":
    # Runnable without pytest (the system interpreter lacks it).
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} verifier-gate tests passed")

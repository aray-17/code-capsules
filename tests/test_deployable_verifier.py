"""Tests for make_repro_verifier / make_repro_grade_fn -- the in-package reference
DEPLOYABLE verifier (non-gold repro grade + regression channel built from
repo-agnostic callables). Mock callables; no docker, no gold. Runs under pytest
or `python3`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from code_capsules.controller.verifier import (  # noqa: E402
    make_repro_verifier, make_repro_grade_fn,
    RESOLVED, FLAT, PARTIAL, BROKEN, NOPATCH, UNKNOWN,
)


def test_empty_patch_is_nopatch():
    g = make_repro_grade_fn(["r1"], lambda p, r: True)
    assert g("") == NOPATCH and g("   ") == NOPATCH


def test_no_repros_is_unknown():
    # No signal -> UNKNOWN (lever must not stop/abandon on it).
    g = make_repro_grade_fn([], lambda p, r: True)
    assert g("patch") == UNKNOWN


def test_all_repros_pass_is_resolved():
    g = make_repro_grade_fn(["r1", "r2"], lambda p, r: True)
    assert g("patch") == RESOLVED


def test_some_repros_pass_is_partial():
    g = make_repro_grade_fn(["r1", "r2"], lambda p, r: r == "r1")
    assert g("patch") == PARTIAL


def test_no_repros_pass_is_flat():
    g = make_repro_grade_fn(["r1", "r2"], lambda p, r: False)
    assert g("patch") == FLAT


def test_regression_break_is_broken():
    # All repros pass but the patch broke existing tests -> BROKEN, not RESOLVED.
    g = make_repro_grade_fn(["r1"], lambda p, r: True, regression_ok=lambda p: False)
    assert g("patch") == BROKEN


def test_regression_green_or_none_is_resolved():
    assert make_repro_grade_fn(["r1"], lambda p, r: True, regression_ok=lambda p: True)("patch") == RESOLVED
    assert make_repro_grade_fn(["r1"], lambda p, r: True, regression_ok=lambda p: None)("patch") == RESOLVED


def test_grade_fn_never_reads_gold():
    # The callables only ever receive (patch, repro) / (patch) -- no gold/instance.
    seen = []
    g = make_repro_grade_fn(["r1"], lambda p, r: (seen.append((p, r)), True)[1])
    g("the-patch")
    assert seen == [("the-patch", "r1")]


# ── make_repro_verifier: the two-channel verifier (grade + regression) ─────────

def test_verifier_keeps_regression_out_of_the_grade():
    # Demote-vs-rescue separation: the channel REPORTS, the governor decides.
    # All repros pass + regression broke -> .grade stays RESOLVED (no BROKEN
    # demotion); the False verdict lives on the .regression channel.
    v = make_repro_verifier(["r1"], lambda p, r: True, regression_ok=lambda p: False)
    assert v.grade("patch") == RESOLVED
    assert v.regression("patch") is False


def test_verifier_grade_matches_repro_only_classes():
    assert make_repro_verifier([], lambda p, r: True).grade("p") == UNKNOWN
    v = make_repro_verifier(["r1", "r2"], lambda p, r: r == "r1")
    assert v.grade("p") == PARTIAL and v.grade("") == NOPATCH
    assert make_repro_verifier(["r1"], lambda p, r: False).grade("p") == FLAT


def test_verifier_grade_memoized_one_sweep_per_distinct_patch():
    calls = []
    v = make_repro_verifier(["r1", "r2"], lambda p, r: (calls.append((p, r)), True)[1])
    assert v.grade("pA") == v.grade("pA") == RESOLVED   # second read is cached
    v.grade("pB")
    assert len(calls) == 4                              # 2 patches x 2 repros, once each


def test_regression_channel_memoized_once_per_distinct_patch():
    calls = []
    v = make_repro_verifier(["r1"], lambda p, r: True,
                            regression_ok=lambda p: (calls.append(p), True)[1])
    assert v.regression("pA") is True
    assert v.regression("pA") is True                   # cached, not re-run
    assert v.regression("pB") is True
    assert calls == ["pA", "pB"]


def test_regression_channel_none_safe():
    # No regression callable -> None; empty patch -> None without running.
    assert make_repro_verifier(["r1"], lambda p, r: True).regression("p") is None
    calls = []
    v = make_repro_verifier(["r1"], lambda p, r: True,
                            regression_ok=lambda p: (calls.append(p), True)[1])
    assert v.regression("") is None and v.regression("   ") is None
    assert calls == []                                  # never invoked on empty


def test_back_compat_wrapper_equivalence():
    # make_repro_grade_fn == verifier channels + the old BROKEN fold-in, on a
    # grid spanning every grade class x regression verdict.
    scenarios = [
        # (repros, passes(p, r), regression verdict, patch, expected old grade)
        (["r1", "r2"], lambda p, r: True, False, "p", BROKEN),    # the fold-in
        (["r1", "r2"], lambda p, r: True, True, "p", RESOLVED),
        (["r1", "r2"], lambda p, r: True, None, "p", RESOLVED),
        (["r1", "r2"], lambda p, r: r == "r1", False, "p", PARTIAL),  # only RESOLVED demotes
        (["r1"], lambda p, r: False, False, "p", FLAT),
        ([], lambda p, r: True, False, "p", UNKNOWN),
        (["r1"], lambda p, r: True, False, "", NOPATCH),
    ]
    for repros, passes, verdict, patch, expected in scenarios:
        wrapped = make_repro_grade_fn(repros, passes, regression_ok=lambda p: verdict)
        assert wrapped(patch) == expected
        v = make_repro_verifier(repros, passes, regression_ok=lambda p: verdict)
        folded = BROKEN if (v.grade(patch) == RESOLVED and v.regression(patch) is False) \
            else v.grade(patch)
        assert wrapped(patch) == folded


def test_wrapper_runs_each_channel_once_per_distinct_patch():
    # Back-compat wrapper inherits the memoization: repeated grading of the same
    # patch costs one repro sweep + one regression run.
    repro_calls, reg_calls = [], []
    g = make_repro_grade_fn(
        ["r1"], lambda p, r: (repro_calls.append(p), True)[1],
        regression_ok=lambda p: (reg_calls.append(p), True)[1])
    assert g("p") == g("p") == RESOLVED
    assert repro_calls == ["p"] and reg_calls == ["p"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"deployable verifier: {passed}/{len(fns)} pass")

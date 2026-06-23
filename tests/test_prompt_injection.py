"""Tests for the mid-session prompt generation moved into the framework
(code_capsules.controller.prompt) in the controller boundary close -- including the
signal-injection partner's prompt builders. Pure functions over duck-typed
session/signals; runs under pytest or `python3` directly."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from code_capsules.controller.prompt import (  # noqa: E402
    build_prompt,
    _extract_failing_test_info, _build_injection_prompt, _signal_explanation,
    _build_stage2_prompt_generic, _build_stage2_prompt_enriched, _summarize_stage1,
)


def _sig(**kw):
    base = dict(patch_attempt_failed=False, traceback=False, test_failure=False,
                file_thrash=False, no_progress=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _tc(is_read=False, is_write=False, is_bash=False, file_path=None, output_text=None):
    return types.SimpleNamespace(is_read=is_read, is_write=is_write, is_bash=is_bash,
                                 file_path=file_path, output_text=output_text)


def test_extract_pytest_failure():
    info = _extract_failing_test_info("... FAILED tests/test_foo.py::test_bar ...")
    assert info["failing_test"] == "tests/test_foo.py::test_bar"


def test_extract_django_failure():
    info = _extract_failing_test_info("FAIL: test_x (mod.sub.ClassName)")
    assert info["failing_test"] == "test_x (mod.sub.ClassName)"


def test_extract_traceback_uses_last_frame():
    out = 'File "a/b.py", line 10\n...\nFile "c/d.py", line 42\n'
    info = _extract_failing_test_info(out)
    assert info["traceback_file"] == "c/d.py" and info["traceback_line"] == "42"


def test_extract_empty_is_safe():
    info = _extract_failing_test_info("")
    assert info == {"failing_test": "", "traceback_file": "", "traceback_line": ""}


def test_build_injection_prompt_injects_signals():
    p = _build_injection_prompt(
        {"failing_test": "t::a", "traceback_file": "x.py", "traceback_line": "9"},
        last_patch="diff --git a/x b/x\n+fix\n", cycle_idx=1, total_cycles=4, extra_turns=10)
    assert "Iteration 2 of up to 4" in p
    assert "t::a" in p and "x.py:9" in p
    assert "+fix" in p and "10 more turns" in p


def test_build_injection_prompt_no_patch():
    p = _build_injection_prompt({}, last_patch="", cycle_idx=0, total_cycles=3, extra_turns=5)
    assert "(no patch produced last cycle)" in p and "Iteration 1 of up to 3" in p


def test_signal_explanation_priority_and_fallback():
    assert "did not resolve" in _signal_explanation(_sig(patch_attempt_failed=True, traceback=True))
    assert "traceback" in _signal_explanation(_sig(traceback=True))
    assert "still failing" in _signal_explanation(_sig(test_failure=True))
    assert "same file" in _signal_explanation(_sig(file_thrash=True))
    assert "reading without writing" in _signal_explanation(_sig(no_progress=True))
    assert "fresh eyes" in _signal_explanation(_sig())


def test_stage2_generic_has_budget():
    p = _build_stage2_prompt_generic(num_turns_used=8, total_budget=25)
    assert "total of 25 turns" in p and "used 8" in p


def test_stage2_enriched_includes_summary_and_directive():
    session = types.SimpleNamespace(tool_calls=[
        _tc(is_read=True, file_path="src/a.py"),
        _tc(is_write=True, file_path="src/a.py"),
        _tc(is_bash=True, output_text="E   assert 1 == 2"),
    ])
    p = _build_stage2_prompt_enriched(session, _sig(test_failure=True),
                                      num_turns_used=10, extra_turns=15, total_budget=25)
    assert "src/a.py" in p and "Files read" in p and "Files edited" in p
    assert "assert 1 == 2" in p          # bash tail surfaced
    assert "still failing" in p          # signal directive spliced in


def test_summarize_stage1_empty_session():
    s = _summarize_stage1(types.SimpleNamespace(tool_calls=[]), _sig())
    assert "no tool activity" in s


# ── G2 disclosure flag: include_fail_to_pass on build_prompt ─────────────────
# The FAIL_TO_PASS test IDs are SWE-bench evaluation metadata a real deployment
# does not have (see the paper's limitations section). Default
# True preserves ALL historical eval behavior; False omits the section.

_G2_INSTANCE = {
    "problem_statement": "Something is broken in the widget.",
    "FAIL_TO_PASS": ["tests/test_widget.py::test_fix",
                     "tests/test_widget.py::test_other"],
}


def test_build_prompt_embeds_fail_to_pass_by_default():
    out = build_prompt(_G2_INSTANCE)
    assert "## Tests that must pass after your fix" in out
    assert "tests/test_widget.py::test_fix" in out
    assert "tests/test_widget.py::test_other" in out
    # explicit True == the default (the historical-eval contract)
    assert build_prompt(_G2_INSTANCE, include_fail_to_pass=True) == out


def test_build_prompt_omits_fail_to_pass_when_disabled():
    out = build_prompt(_G2_INSTANCE, include_fail_to_pass=False)
    assert "Tests that must pass" not in out
    assert "test_widget" not in out                    # no test ID leaks through
    # the rest of the prompt is intact
    assert "## Issue" in out and "Something is broken in the widget." in out
    assert "## Instructions" in out


def test_build_prompt_flag_composes_with_other_knobs():
    out = build_prompt(_G2_INSTANCE, prompt_budget_hint=10,
                       prompt_variant="plan_first",
                       preselect_block="## Files\n- a.py\n",
                       include_fail_to_pass=False)
    assert "10 turns" in out and "written plan" in out and "a.py" in out
    assert "test_widget" not in out


def test_build_prompt_json_string_fail_to_pass_unaffected_by_default():
    # FAIL_TO_PASS as a JSON string (the raw dataset form) still parses when on
    inst = dict(_G2_INSTANCE, FAIL_TO_PASS='["tests/test_widget.py::test_fix"]')
    assert "tests/test_widget.py::test_fix" in build_prompt(inst)
    assert "test_widget" not in build_prompt(inst, include_fail_to_pass=False)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn(); passed += 1
    print(f"all prompt-injection checks PASS ({passed}/{len(fns)})")

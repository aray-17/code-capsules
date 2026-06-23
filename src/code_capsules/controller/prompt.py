"""Prompt assembly (moved from the SWE-bench harness into the framework).

build_prompt turns an issue + a policy's budget/variant/preselect knobs into the
user prompt for a coding-agent session. It lives here, not in the harness, so the
framework owns prompt construction (a deployer building on Code-Capsules gets it
without the SWE-bench harness). Self-contained: depends only on the issue dict +
stdlib. The harness re-exports these names for backward compatibility.
"""
from __future__ import annotations

import re
from typing import Optional

# Prompt variants (plan-then-execute, per-tool-class hint, phase-staged).
PROMPT_VARIANTS = ("plan_first", "tool_alloc", "phase_staged")


def build_prompt(instance: dict, prompt_budget_hint: Optional[int] = None,
                 preselect_block: str = "",
                 prompt_variant: Optional[str] = None,
                 include_fail_to_pass: bool = True) -> str:
    """
    Build the user prompt for a coding-agent (SWE-bench) instance.

    If `prompt_budget_hint` is set, prepend an explicit turn-budget statement
    (the signaled-budget regime, vs the implicit-cap regime where the model never
    sees the number). If `preselect_block` is non-empty, splice the (already
    rendered) issue-relevant-files block after the budget block. If `prompt_variant`
    is set, prepend a per-variant "## Approach" block (plan_first / tool_alloc /
    phase_staged).

    include_fail_to_pass (plan §3 G2): when True (the default), embed the
    instance's gold FAIL_TO_PASS test IDs as the "## Tests that must pass
    after your fix" section. Those IDs are SWE-bench evaluation metadata a
    real deployment does NOT have (test names/paths often encode the fix
    location): see the paper's limitations section for the H2H fairness
    asymmetry vs the issue-only Agentless baseline. The default True matches
    the historical evals;
    set False for deployment-realistic prompts or the no-test-ID bounding
    cell (plan experiment E5b) -- policy surface:
    `controller.prompt_includes_fail_to_pass` on RunnerPolicy.
    """
    tests_block = ""
    if include_fail_to_pass:
        fail_tests = instance.get("FAIL_TO_PASS", [])
        if isinstance(fail_tests, str):
            import json as _json
            fail_tests = _json.loads(fail_tests)
        if fail_tests:
            ids = "\n".join(f"  - {t}" for t in fail_tests)
            tests_block = (
                f"\n## Tests that must pass after your fix\n\n{ids}\n\n"
                "These tests currently fail. Your fix must make them pass "
                "without breaking existing passing tests.\n"
            )

    budget_block = ""
    if prompt_budget_hint is not None:
        budget_block = (
            f"## Turn budget\n\n"
            f"You have at most **{prompt_budget_hint} turns** to complete this task. "
            f"Skip speculative exploration; locate the bug efficiently, patch it, "
            f"and verify. Plan your turns deliberately.\n\n"
        )

    variant_block = _render_variant_block(prompt_variant, prompt_budget_hint)

    return (
        "You are working in a software repository. Fix the following bug.\n\n"
        f"{budget_block}"
        f"{variant_block}"
        f"{preselect_block}"
        "## Issue\n\n"
        f"{instance['problem_statement']}\n"
        f"{tests_block}\n"
        "## Instructions\n\n"
        "1. Read the relevant source files to understand the codebase\n"
        "2. Implement the minimal fix needed to resolve the issue in **source files only**\n"
        "3. Do **not** modify files under `tests/` — the test suite is "
        "applied separately by the evaluation harness. Test-file changes "
        "in your patch will be discarded.\n"
        "4. Make changes directly to the source files\n"
    )


def _render_variant_block(variant: Optional[str], budget: Optional[int]) -> str:
    """Render the per-variant '## Approach' block. Empty if no variant."""
    if not variant:
        return ""
    if variant == "plan_first":
        return (
            "## Approach\n\n"
            "**Turn 1 must be a written plan only — no tool calls.** "
            "In one paragraph: (a) where you think the bug is, (b) which "
            "files you will edit, (c) what the patch shape will be, "
            "(d) which test failure you will check first. "
            "Turn 2 onward: execute the plan.\n\n"
        )
    if variant == "tool_alloc":
        return (
            "## Approach\n\n"
            "Suggested turn allocation:\n"
            "- ~5 file reads (exploration)\n"
            "- ~3 patch attempts (Edit/Write)\n"
            "- ~5 test-and-iterate cycles (Bash + Read + Edit)\n\n"
            "If you've re-read the same file twice without editing it, "
            "commit to an edit instead. Each test failure should be followed "
            "by a targeted re-read of the implicated file and one refined "
            "edit, not generic exploration.\n\n"
        )
    if variant == "phase_staged":
        b = budget or 35
        explore_n = 10
        patch_n = 10
        iter_n = max(5, b - explore_n - patch_n)
        return (
            "## Approach\n\n"
            "Run this task in three explicit phases:\n\n"
            f"**Phase 1 (turns 1-{explore_n}): Explore.** Read the relevant "
            "source files to understand the bug. Do not edit yet. End the "
            "phase when you can describe the fix in one sentence.\n\n"
            f"**Phase 2 (turns {explore_n+1}-{explore_n+patch_n}): "
            "First patch attempt.** Make one minimal edit and run the failing "
            "tests once. Do not over-engineer — get a candidate fix in place.\n\n"
            f"**Phase 3 (turns {explore_n+patch_n+1}-{b}): Iterate.** "
            "Read test output, re-read only the implicated file, refine the "
            "patch, re-run tests. Repeat until tests pass.\n\n"
        )
    raise ValueError(f"unknown prompt_variant: {variant!r}")


# ---------------------------------------------------------------------------
# Mid-session prompt generation (moved from the SWE-bench harness into the
# framework so the validated two-config ensemble -- the baseline single-pass
# configuration plus the SIGNAL-INJECTION partner -- is self-contained in the
# runtime; the harness re-exports these names).
# These are pure builders: they read a duck-typed session/signals object (the
# attributes a coding-agent session exposes) and emit a prompt string. No
# benchmark or harness internals.
# ---------------------------------------------------------------------------

def _summarize_stage1(session, signals) -> str:
    """Concise summary of a completed stage-1 session for the enriched stage-2
    prompt: files read/edited and the most recent bash output tail (what failed)."""
    from collections import Counter
    reads_per_file = Counter(
        tc.file_path for tc in session.tool_calls if tc.is_read and tc.file_path
    )
    writes_per_file = Counter(
        tc.file_path for tc in session.tool_calls if tc.is_write and tc.file_path
    )
    bash_calls = [tc for tc in session.tool_calls if tc.is_bash]

    parts: list[str] = []
    if reads_per_file:
        items = [f"{p} (×{c})" if c > 1 else p
                 for p, c in reads_per_file.most_common(8)]
        parts.append(f"**Files read:** {', '.join(items)}")
    if writes_per_file:
        items = [f"{p} (×{c})" if c > 1 else p
                 for p, c in writes_per_file.most_common(8)]
        parts.append(f"**Files edited:** {', '.join(items)}")
    if bash_calls:
        last = bash_calls[-1]
        tail = (last.output_text or "")[-300:].strip()
        parts.append(
            f"**Commands run:** {len(bash_calls)} total. Most recent output tail:\n```\n{tail}\n```"
        )
    if not parts:
        parts.append("(no tool activity captured in stage 1)")

    return "\n\n".join(parts)


def _signal_explanation(signals) -> str:
    """Map fired signals to a directive hint for the model."""
    if signals.patch_attempt_failed:
        return (
            "Your most recent fix attempt was followed by a failing test run — "
            "the patch you tried did not resolve the issue. Re-examine whether "
            "you're targeting the right file or the right behavior."
        )
    if signals.traceback:
        return (
            "Your most recent attempt produced a Python traceback — your fix "
            "may have broken imports or runtime behavior. Roll back mentally "
            "and try a smaller, more surgical change."
        )
    if signals.test_failure:
        return (
            "Tests are still failing after your attempts. Read the failure "
            "message carefully — what does it tell you about the actual bug?"
        )
    if signals.file_thrash:
        return (
            "You re-read the same file multiple times without committing to "
            "a fix. This suggests you may be focused on the wrong area, or "
            "missing context elsewhere. Broaden your search."
        )
    if signals.no_progress:
        return (
            "You spent several consecutive turns reading without writing. "
            "Commit to a fix attempt now — even a partial one is more useful "
            "than further exploration."
        )
    return "Re-examine your approach with fresh eyes."


def _build_stage2_prompt_generic(num_turns_used: int, total_budget: int) -> str:
    """Generic budget-extension stage-2 prompt."""
    return (
        f"Your turn budget has been extended. You now have a total of "
        f"{total_budget} turns; you've used {num_turns_used} so far. "
        "Continue diagnosing the issue and complete the fix. Tests must pass."
    )


def _build_stage2_prompt_enriched(session1, signals1, num_turns_used: int,
                                  extra_turns: int, total_budget: int) -> str:
    """Enriched stage-2 prompt: stage-1 activity summary + a signal-specific
    directive (tell the model what it just did and why we noticed it was stuck)."""
    summary = _summarize_stage1(session1, signals1)
    explanation = _signal_explanation(signals1)
    return (
        f"Your turn budget has been extended. You used {num_turns_used} turns "
        f"in the initial attempt; you have {extra_turns} more turns now "
        f"(total budget: {total_budget}).\n\n"
        f"## What you did in the initial attempt\n\n{summary}\n\n"
        f"## Why I'm extending your budget\n\n{explanation}\n\n"
        f"## What to do now\n\n"
        "Step back from your current approach. Consider files or aspects you "
        "haven't explored yet. Tests must pass. Do not repeat the same "
        "investigation pattern — if you've already inspected a file, trust "
        "your prior reading unless something new has come to light."
    )


# Failing-test / traceback extraction for the signal-injection partner.
_PYTEST_FAIL_RE = re.compile(r"FAILED ([^\s]+(?:::[^\s]+)?)")
_DJANGO_FAIL_RE = re.compile(r"FAIL: (\w+ \(\S+\))")
_TRACEBACK_FILE_RE = re.compile(r'File "([^"]+\.py)", line (\d+)')


def _extract_failing_test_info(eval_stdout: str) -> dict:
    """Parse evaluation stdout for the failing test name + most-recent traceback
    frame. Returns {failing_test, traceback_file, traceback_line} (may be empty)."""
    info = {"failing_test": "", "traceback_file": "", "traceback_line": ""}
    if not eval_stdout:
        return info
    m = _PYTEST_FAIL_RE.search(eval_stdout)
    if m:
        info["failing_test"] = m.group(1)
    else:
        m = _DJANGO_FAIL_RE.search(eval_stdout)
        if m:
            info["failing_test"] = m.group(1)
    matches = list(_TRACEBACK_FILE_RE.finditer(eval_stdout))
    if matches:
        last = matches[-1]
        info["traceback_file"] = last.group(1)
        info["traceback_line"] = last.group(2)
    return info


def _build_injection_prompt(failing_info: dict, last_patch: str,
                            cycle_idx: int, total_cycles: int,
                            extra_turns: int) -> str:
    """The signal-injection partner's resume prompt: inject the failing test +
    traceback frame + last-patch excerpt back to the agent for a refined attempt."""
    test_line = ""
    if failing_info.get("failing_test"):
        test_line = f"Failing test: `{failing_info['failing_test']}`\n"
    if failing_info.get("traceback_file"):
        tb = failing_info["traceback_file"]
        ln = failing_info.get("traceback_line", "?")
        test_line += f"Most-recent traceback frame: `{tb}:{ln}`\n"
    patch_excerpt = last_patch[:1500] if last_patch else "(no patch produced last cycle)"
    return (
        f"## Iteration {cycle_idx + 1} of up to {total_cycles}\n\n"
        f"Your previous patch did not pass the failing tests.\n\n"
        f"{test_line}\n"
        f"Your last patch (excerpt):\n```diff\n{patch_excerpt}\n```\n\n"
        f"You have {extra_turns} more turns. Refine the patch — focus narrowly on the "
        f"failing test. Re-read the implicated file at the traceback line, "
        f"identify what your last patch missed, and produce a corrected version. "
        f"Avoid restarting exploration; iterate on the patch you have.\n"
    )


__all__ = [
    "build_prompt", "_render_variant_block", "PROMPT_VARIANTS",
    "_summarize_stage1", "_signal_explanation",
    "_build_stage2_prompt_generic", "_build_stage2_prompt_enriched",
    "_extract_failing_test_info", "_build_injection_prompt",
]

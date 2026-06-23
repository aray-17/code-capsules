"""
Generic prompt assembly for variants.

Builds the user prompt from a TaskDescriptor + VariantConfig. Adds:
- Optional turn-budget hint (signaled regime)
- Optional approach block (plan_first / tool_alloc / phase_staged)
- Optional preselect block (relevance-ranker output)
- Optional context block (caller-provided pre-prompt content, e.g. tests list)

Domain-agnostic: this module does NOT know about SWE-bench, FAIL_TO_PASS,
HumanEval test asserts, etc. Benchmark harnesses prepend their domain content
into task.text or pass it through task.context['pre_prompt_blocks'].
"""
from __future__ import annotations

from typing import Optional


def render_approach_block(variant: Optional[str], budget: Optional[int]) -> str:
    """Render an `## Approach` block for the given prompt variant.

    Variants: plan_first, tool_alloc, phase_staged. The block wording is the
    load-bearing experimental treatment.
    """
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


def build_user_prompt(
    task_text: str,
    *,
    prompt_budget_hint: Optional[int] = None,
    prompt_variant: Optional[str] = None,
    preselect_block: str = "",
    pre_blocks: Optional[list[str]] = None,
    instructions: Optional[str] = None,
) -> str:
    """Assemble the user prompt sent to claude -p.

    Sections, in order:
      1. Turn budget block (if prompt_budget_hint is set)
      2. Approach block (if prompt_variant is set)
      3. Preselect block (if non-empty; from a RelevanceRanker wrapper)
      4. Any caller-supplied pre_blocks (e.g. SWE-bench tests list)
      5. The task text itself (under an ## Issue heading)
      6. Instructions (default: generic 4-step "read, fix, don't touch tests")

    Callers can override the instructions for non-SWE-bench tasks.
    """
    budget_block = ""
    if prompt_budget_hint is not None:
        budget_block = (
            f"## Turn budget\n\n"
            f"You have at most **{prompt_budget_hint} turns** to complete this task. "
            f"Skip speculative exploration; locate the bug efficiently, patch it, "
            f"and verify. Plan your turns deliberately.\n\n"
        )

    variant_block = render_approach_block(prompt_variant, prompt_budget_hint)
    preamble_extras = "".join(pre_blocks) if pre_blocks else ""

    instr = instructions or (
        "## Instructions\n\n"
        "1. Read the relevant source files to understand the codebase\n"
        "2. Implement the minimal fix needed to resolve the issue in **source files only**\n"
        "3. Do **not** modify files under `tests/` — the test suite is "
        "applied separately by the evaluation harness. Test-file changes "
        "in your patch will be discarded.\n"
        "4. Make changes directly to the source files\n"
    )

    return (
        "You are working in a software repository. Fix the following bug.\n\n"
        f"{budget_block}"
        f"{variant_block}"
        f"{preselect_block}"
        f"{preamble_extras}"
        f"## Issue\n\n{task_text}\n\n"
        f"{instr}"
    )


__all__ = ["build_user_prompt", "render_approach_block"]

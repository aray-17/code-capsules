"""
Relevance-ranker wrapper variant.

Composable wrapper that ranks files in the worktree by relevance to the issue,
prepends a "Likely relevant files" block to the prompt, then delegates to
another variant.

Cross-tier ranker-inverse-capability finding:
  - Haiku: +4pp STACKS on plan-then-execute (the cheap-tier model benefits from
    the hint)
  - Sonnet: +1pp at most (mid-tier breaks even)
  - Opus: -6pp REGRESSES (high-capability self-locates files; the block becomes
    wasted tokens AND a distraction)

Deployment guidance: wrap on Haiku tier; do NOT wrap on Opus tier; Sonnet
optional. Choose top_n by tier (default 10 for Haiku/Sonnet, smaller or zero
for Opus).

The inner variant is identified by config.extra['inner_variant'] (defaults
to plan_then_execute) and looked up from the registry. top_n comes from
config.preselect_top_n.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from code_capsules.api import RunResult, TaskDescriptor, VariantConfig


class RelevanceRankerWrapper:
    name = "relevance_ranker"

    def run(self, task: TaskDescriptor, config: VariantConfig) -> RunResult:
        from code_capsules.controller.file_ranker import rank_relevant_files, format_for_prompt
        from code_capsules.api.registry import get

        top_n = config.preselect_top_n or int(config.extra.get("top_n", 10))
        inner_name = config.extra.get("inner_variant", "plan_then_execute")
        worktree = Path(task.context["worktree_path"])

        # Rank + format
        preselect_block = ""
        if top_n > 0:
            ranked = rank_relevant_files(
                worktree, task.text, top_n=top_n,
            )
            preselect_block = format_for_prompt(ranked)

        # Splice block into task.context so the inner variant picks it up
        new_context = {**task.context, "preselect_block": preselect_block}
        wrapped_task = TaskDescriptor(text=task.text, context=new_context)

        # Drop preselect_top_n / wrapper-only keys from inner config
        inner_extra = {
            k: v for k, v in config.extra.items()
            if k not in ("inner_variant", "top_n")
        }
        inner_config = replace(
            config,
            preselect_top_n=0,  # wrapper has handled it
            extra=inner_extra,
        )

        inner = get("variant", inner_name)
        return inner.run(wrapped_task, inner_config)


__all__ = ["RelevanceRankerWrapper"]

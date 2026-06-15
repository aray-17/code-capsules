"""
Reproduce the calibration methodology on your own workload (paper Section 4.2).

The shipped deployment_defaults are not hand-picked: each cell is selected from a
cost-quality Pareto frontier by a fixed rule. The same recipe is reproducible for
any workload:

  1. Run the variant catalog on a representative sample of the workload through
     the same evaluation harness; record resolved fraction + cache-aware
     $/attempt per cell. (Your harness produces these numbers; here they are
     illustrative, and they are in fact the shipped hard_workload cells, so the
     frontier this example emits is the paper's menu.)
  2. Build a typed CodeCapsulesPolicy per measured cell.
  3. Keep the cost-quality Pareto frontier (no cell another beats on both axes).
  4. Map the frontier, cheapest first, to the menu knees and emit a fresh
     deployment_defaults block via CodeCapsulesPolicy.to_yaml_entry().
  5. The block round-trips: CodeCapsulesPolicy.from_yaml loads it back.

All offline; no model calls.

Run:
    python -m examples.calibrate_deployment_defaults
"""

from __future__ import annotations

import dataclasses

import yaml

from code_capsules import CodeCapsulesPolicy

WORKLOAD = "hard_workload"

# Step 1: (variant, mode, turn_budget, tier, pass_rate, $/attempt) per measured cell.
# Replace with your harness's measurements. The last row is a dominated cell,
# included to show the Pareto filter drop it.
MEASURED_CELLS = [
    ("relevance_ranker",  "sequential", 10,  "sonnet", 0.38, 0.18),
    ("two_pass_critique", "escalating", 10,  "sonnet", 0.44, 0.41),
    ("unbounded_budget",  "sequential", 100, "sonnet", 0.51, 0.47),
    ("implicit_budget",   "sequential", 20,  "opus",   0.61, 0.48),
    ("unbounded_budget",  "sequential", 100, "opus",   0.64, 0.60),
    ("signaled_budget",   "sequential", 5,   "sonnet", 0.30, 0.20),  # dominated
]

KNEES = ["cost_min", "balanced", "quality", "quality_max", "ceiling"]


def pareto_upper_left(cells):
    """Keep each cell no other cell beats on both axes (>= pass_rate AND <= cost)."""
    keep = []
    for c in cells:
        dominated = any(
            o is not c
            and o.pass_rate >= c.pass_rate and o.cost_per_task <= c.cost_per_task
            and (o.pass_rate > c.pass_rate or o.cost_per_task < c.cost_per_task)
            for o in cells
        )
        if not dominated:
            keep.append(c)
    return keep


def main() -> None:
    # Step 2: a typed policy per measured cell.
    cells = [
        CodeCapsulesPolicy(variant=v, mode=m, turn_budget=tb,
                           workload_class=WORKLOAD, tier=t, knee="balanced",
                           pass_rate=pr, cost_per_task=c)
        for (v, m, tb, t, pr, c) in MEASURED_CELLS
    ]

    # Step 3: keep the Pareto frontier, cheapest first.
    frontier = sorted(pareto_upper_left(cells), key=lambda c: c.cost_per_task)
    print(f"{len(cells)} measured cells -> {len(frontier)} on the cost-quality frontier")

    # Step 4: map the frontier to the menu knees and emit deployment_defaults.
    defaults: dict = {}
    for knee, cell in zip(KNEES, frontier):
        cell = dataclasses.replace(cell, knee=knee)
        (defaults.setdefault(cell.workload_class, {})
                 .setdefault(cell.tier, {})[knee]) = cell.to_yaml_entry()

    doc = {"deployment_defaults": defaults}
    print("\n--- generated deployment_defaults ---")
    print(yaml.dump(doc, sort_keys=False).rstrip())

    # Step 5: the emitted block round-trips through from_yaml.
    cheapest = frontier[0]
    reloaded = CodeCapsulesPolicy.from_yaml_string(
        yaml.dump(doc), workload=WORKLOAD, tier=cheapest.tier, knee="cost_min")
    print(f"\nround-trip cost_min: variant={reloaded.variant} "
          f"tier={reloaded.tier} pass_rate={reloaded.pass_rate}")


if __name__ == "__main__":
    main()

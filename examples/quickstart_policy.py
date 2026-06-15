"""
Quickstart: the configuration DSL (pick a preset, or build a custom config).

Two layers, each with a DEPLOYMENT-PRESET form (pick a shipped config by name) and
a NON-PRESET form (build a custom config from scratch):

  THE DEPLOYABLE LEVER (the 3-component controller -- the headline):
    5. deployment preset:  CodeCapsulesRunner.from_policy_file("policy.yaml")
    6. custom (non-preset): CodeCapsulesRunner(RunnerPolicy(...))

  A SINGLE VARIANT (one solver, no governor/verifier):
    1. deployment preset:  policy_for(workload, tier, knee)
    2. custom (non-preset): CodeCapsulesPolicy(...)
    3/4. override a preset / YAML round-trip

All offline -- no API keys. To actually RUN the lever (sampler + grade_fn), see
examples/deployable_lever.py.

Run:
    python -m examples.quickstart_policy
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from code_capsules import (
    CodeCapsulesPolicy, policy_for, SHIPPED_PRESETS,
    CodeCapsulesRunner, RunnerPolicy,
)

_POLICY_YAML = str(Path(__file__).parent.parent / "policy.yaml")


def main() -> None:
    print("=" * 60)
    print("0. The shipped cross-tier menu (5 knees)")
    print("=" * 60)
    # The whole menu, cheapest to strongest. cost_min/balanced/quality run on
    # Sonnet; quality_max/ceiling run on Opus. No single knee wins every
    # workload (the quality-max pick reverses on the sympy-heavy held-out
    # split), so the menu, not a single config, is the deliverable.
    for (workload, tier, knee), p in SHIPPED_PRESETS.items():
        print(f"  {knee:<12} {tier:<7} {p.variant:<18} "
              f"pass={p.pass_rate:.0%}  ${p.cost_per_task:.2f}/attempt")

    print()
    print("=" * 60)
    print("1. Shipped preset")
    print("=" * 60)

    # The cost-axis knee (two-pass critique) on Sonnet for hard agent workloads.
    # Anchored on SWE-bench Lite first 150 under cache-aware accounting.
    balanced = policy_for(
        workload="hard_workload",
        tier="sonnet",
        knee="balanced",
    )
    print(f"variant:        {balanced.variant}")
    print(f"mode:           {balanced.mode}")
    print(f"turn_budget:    {balanced.turn_budget}")
    print(f"prompt_variant: {balanced.prompt_variant}")
    print(f"pass_rate:      {balanced.pass_rate}")
    print(f"cost_per_task:  ${balanced.cost_per_task:.3f}")

    print()
    print("=" * 60)
    print("2. Direct construction (advanced)")
    print("=" * 60)

    # The balanced knee, equivalent to policy_for(knee='balanced').
    # Operators construct directly when calibrating a new cell outside
    # the shipped presets.
    knee = CodeCapsulesPolicy(
        variant="two_pass_critique",
        mode="escalating",
        turn_budget=10,
        escalating_start_budget=10,
        escalating_target_budget=25,
        always_escalate=True,
        prompt_variant="two_pass_critique",
        workload_class="hard_workload",
        tier="sonnet",
        knee="balanced",
    )
    print(f"variant:                  {knee.variant}")
    print(f"mode:                     {knee.mode}")
    print(f"escalating_start_budget:  {knee.escalating_start_budget}")
    print(f"escalating_target_budget: {knee.escalating_target_budget}")
    print(f"always_escalate:          {knee.always_escalate}")

    print()
    print("=" * 60)
    print("3. Override a preset for a customer tier (replace)")
    print("=" * 60)

    # A premium customer wants the balanced knee with a tighter
    # turn budget. dataclasses.replace creates a new policy with the
    # override; the original preset is unchanged.
    base = policy_for(knee="balanced")
    premium = replace(base, turn_budget=15)
    print(f"base turn_budget:    {base.turn_budget}")
    print(f"premium turn_budget: {premium.turn_budget}")
    assert base.turn_budget == 10  # original unchanged

    print()
    print("=" * 60)
    print("4. YAML round-trip")
    print("=" * 60)

    # Serialise the knee policy to a YAML entry, then load it back.
    entry = knee.to_yaml_entry()
    print("YAML entry shape:")
    for k, v in entry.items():
        print(f"  {k}: {v}")

    yaml_doc = f"""
deployment_defaults:
  hard_workload:
    sonnet:
      balanced:
        variant: {entry['variant']}
        mode: {entry['mode']}
        turn_budget: {entry['turn_budget']}
        escalating_start_budget: {entry['escalating_start_budget']}
        escalating_target_budget: {entry['escalating_target_budget']}
        always_escalate: {str(entry['always_escalate']).lower()}
        prompt_variant: {entry['prompt_variant']}
"""
    loaded = CodeCapsulesPolicy.from_yaml_string(yaml_doc, knee="balanced")
    print(f"\nLoaded back: variant={loaded.variant} mode={loaded.mode}")
    assert loaded.variant == knee.variant
    assert loaded.mode == knee.mode

    print()
    print("Sections 1-4 configure a SINGLE variant. The deployable LEVER below is the")
    print("3-component controller (solver pair + verifier-select + governor-abandon).")

    print()
    print("=" * 60)
    print("5. The deployable lever: DEPLOYMENT PRESET (shipped policy.yaml)")
    print("=" * 60)

    # Pick the shipped deployable lever: the controller block of policy.yaml supplies
    # the diverse solver pair, the tier ladder, and the governor's abandon default.
    lever = CodeCapsulesRunner.from_policy_file(_POLICY_YAML)
    print(f"solvers (configs):     {lever.policy.configs}")
    print(f"tiers:                 {lever.policy.tiers}")
    print(f"governor (abandon):    escalate_on_agreement={lever.policy.escalate_on_agreement}")
    print(f"min_samples:           {lever.policy.min_samples}")
    print("run it with:  lever.run(sampler, grade_fn)   # see examples/deployable_lever.py")

    print()
    print("=" * 60)
    print("6. The deployable lever: CUSTOM (non-preset) config")
    print("=" * 60)

    # Build a custom lever: your own diverse solver pair + opt into model-tier
    # escalation (the validated default is abandon; escalation is the marginal opt-in).
    custom = CodeCapsulesRunner(RunnerPolicy(
        configs=("floor", "plan_then_execute"),   # any registered solvers
        tiers=("sonnet", "opus"),                 # model ladder, weakest -> strongest
        escalate_on_agreement=True,               # opt in to tier escalation
    ))
    print(f"solvers (configs):     {custom.policy.configs}")
    print(f"tiers:                 {custom.policy.tiers}")
    print(f"governor (escalate):   escalate_on_agreement={custom.policy.escalate_on_agreement}")

    print()
    print("DSL: pick a deployment preset, or build a custom config -- for the lever and "
          "for single variants.")


if __name__ == "__main__":
    main()

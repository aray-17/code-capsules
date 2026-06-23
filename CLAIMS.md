# Paper claims and supporting evidence

This file lists the load-bearing claims of *The Economics of Coding
Agents: Calibrating the Cost-Quality Frontier* and points each claim at
the artifact in this repository that backs it. The paper PDF is at
[`paper/paper.pdf`](paper/paper.pdf); the exact code state cited in the
paper is tagged
[`v1.0-arxiv`](https://github.com/aray-17/code-capsules/releases/tag/v1.0-arxiv).

The headline claims below reproduce **offline** (no Docker, no API keys,
no model calls) from the committed evaluation data under [`evals/`](evals/),
including the deployment-governor claims (the diverse-sample agreement
signal, the regression-suite gate, the escalation gate, and the
value-of-resolve rule).
Reproduce the headline claims at once:

```bash
python3 benchmarks/verify_criteria.py     # PASS/FAIL per headline claim; Section 7 claims cited to the paper
```

or browse them interactively (the evidence explorer):

```bash
bash benchmarks/explore.sh                # builds the data, serves localhost, opens the browser
```

## Headline numbers

Reported as one modeled cost surface (per-instance USD), same models,
same data split, same scorer across systems.

| Claim | Result | Scorer | Evidence |
|---|---|---|---|
| Head-to-head vs. Agentless (Pareto) | Code-Capsules 172/300 resolved (57.3%) @ \$0.445/inst vs. Agentless 152/300 (50.7%) @ \$0.456, **more resolved at lower cost** (McNemar p=0.0055, 13.8% lower cost per resolve) | [`benchmarks/swebench/h2h_canonical_score.py`](benchmarks/swebench/h2h_canonical_score.py) | [`evals/leakfree/tb_forcestage2_first150.jsonl`](evals/leakfree/tb_forcestage2_first150.jsonl), [`tb_forcestage2_second150.jsonl`](evals/leakfree/tb_forcestage2_second150.jsonl), [`evals/scopeC/canonical_tp_h2h.jsonl`](evals/scopeC/canonical_tp_h2h.jsonl), [`evals/scopeC/h2h_agentless_n300_scored_20260616.jsonl`](evals/scopeC/h2h_agentless_n300_scored_20260616.jsonl) |
| Calibrated cross-tier Pareto menu (resolved/150) | cost-min 80 @ \$0.183 (\$0.34/res) · balanced 98 @ \$0.408 (\$0.62/res) · quality 111 @ \$0.466 (\$0.63/res) · quality-max 128 @ \$0.480 (\$0.56/res) · ceiling 138 @ \$0.604 (\$0.66/res) (quality-max out-resolves quality at lower \$/resolve) | [`benchmarks/swebench/cross_tier_pareto_menu.py`](benchmarks/swebench/cross_tier_pareto_menu.py) | one committed JSONL per cell: cost-min [`evals/p10_sonnet_p8c_n150_20260515T235913.jsonl`](evals/p10_sonnet_p8c_n150_20260515T235913.jsonl), balanced [`evals/leakfree/tb_forcestage2_first150.jsonl`](evals/leakfree/tb_forcestage2_first150.jsonl), quality (leak-free re-run) [`evals/leakfree/exp4_lever_floor100_siginject.jsonl`](evals/leakfree/exp4_lever_floor100_siginject.jsonl), quality-max [`evals/p9_opus_implicit20_n150_combined_20260527.jsonl`](evals/p9_opus_implicit20_n150_combined_20260527.jsonl), ceiling [`evals/p9_opus_floor_n150_combined_20260527.jsonl`](evals/p9_opus_floor_n150_combined_20260527.jsonl) |

## Mechanism claims

**C1. Diverse-sample agreement is a tier-local escalation trigger.**
A diverse pair that agrees on failure marks an instance hard at this
tier: the best in-family pair (unbounded floor plus signal injection)
reaches 90.9% precision, firing on 22 instances with 2 false
terminations. A single configuration alone reaches 64.5% and the
median pair 66.7%. The deployment-relevant independent (out-of-family)
precision is roughly 22%, so the signal is best read as a tier-local
escalation trigger rather than an absolute-doom detector: 20 of 126
instances are doomed at this tier, while the oracle union resolves
106 of 126. Paper §agreement; scorer
[`benchmarks/swebench/agreement_signal.py`](benchmarks/swebench/agreement_signal.py).

**C2. The run-both lever lands within two instances of the oracle, yet
adds nothing deployable.** Running a diverse pair and selecting between
candidates resolves 122/150 @ \$118.75 - within two instances of the
per-instance oracle union (124/150) - but selection adds zero net
resolves over its best single member. Deployed as a single-tier abandon
rule it is dominated by the unbounded floor (111/150 @ \$0.63/resolve)
and resolves 84/150 honest at \$1.41/resolve (+124% cost for fewer
resolves), so it is retired as a first-class negative. Ship precision is
77% (first-150) / 70% (held-out). Scorer
[`benchmarks/swebench/oracle_governor.py`](benchmarks/swebench/oracle_governor.py).

**C3. A regression-suite gate recovers a majority of wrongly-abandoned
resolves.** Conditioning the abandon decision on a green regression
suite lifts the honest controller from 84 to 99 resolves by recovering
15 of 28 wrong-abandons, 12 short of the unbounded floor (111), so it
narrows the gap rather than reaching parity. A regression-green
candidate resolves with probability 0.792.
Scorer
[`benchmarks/swebench/regression_gate.py`](benchmarks/swebench/regression_gate.py).

**C4. A model-tier escalation gate stops discriminating once the
strong tier runs.** The strong tier recovers 21 of 21 gate-positive
instances but also 15 of 17 gate-negative ones, so the gate's
selectivity collapses: it no longer separates strong-tier recoveries.
Escalation reaches 85/150 over a \$0.82/resolve base operating cost.
Scorer
[`benchmarks/swebench/escalation_gate.py`](benchmarks/swebench/escalation_gate.py).

**C5. Two-pass critique is Pareto-favorable vs. Agentless.** Same
models, data, and scorer: 172/300 (57.3%) @ \$0.445 vs. Agentless
oracle@k 152/300 (50.7%) @ \$0.456, more resolved at lower
per-instance cost, a +20-instance (+6.7pp) win at McNemar p=0.0055 and
13.8% lower cost per resolve. Scorer
[`benchmarks/swebench/h2h_canonical_score.py`](benchmarks/swebench/h2h_canonical_score.py).

**C6. The shipped policy menu is a calibrated cross-tier Pareto
frontier.** Five presets span the frontier (resolved/150): cost-min
80 (\$0.34/resolve), balanced 98 (\$0.62), quality 111 (\$0.63),
quality-max 128 (\$0.56), ceiling 138 (\$0.66); quality-max
out-resolves quality at a lower cost per resolve. Scorer
[`benchmarks/swebench/cross_tier_pareto_menu.py`](benchmarks/swebench/cross_tier_pareto_menu.py);
the shipped presets are in [`policy.yaml`](policy.yaml).

**C8. Cross-vendor cost ratios on HumanEval / MBPP.** Pass rates
HumanEval 98.2% / 92.7%, MBPP 53.0% / 50.2%; modeled cost ratios 4.8×
(HumanEval) and 15.9× (MBPP). Scorer
[`benchmarks/cross_vendor/score.py`](benchmarks/cross_vendor/score.py);
evidence [`evals/cross_vendor_humaneval_n164_postfix_20260525.csv`](evals/cross_vendor_humaneval_n164_postfix_20260525.csv),
[`evals/cross_vendor_mbpp_n500_postfix_20260525.csv`](evals/cross_vendor_mbpp_n500_postfix_20260525.csv).

**C12. Plan-then-execute and two-pass critique both add lift on every
vendor.** Over the signaled-budget baseline (always-run,
first-150), the lift ladder rises monotonically on both vendors:
Sonnet 84 (56.0%) → plan-then-execute 93 (62.0%) → two-pass critique
98 (65.3%); gpt-5-codex 46 (30.7%) → plan-then-execute 49 (32.7%) →
two-pass critique 66 (44.0%) for the committed run (the paper reports
the three-run mean, 66.7, 44.5%). Two-pass critique adds lift on
both (Sonnet +3pp, codex +11pp). gpt-5-codex is the cheaper per-resolve
corner (\$0.43 vs. Sonnet \$0.62), but it emits empty patches on ~44%
of instances (failing cheaply), which caps its ceiling roughly 21pp
below Sonnet. Scorer
[`benchmarks/swebench/claim12_plan_critique_lift.py`](benchmarks/swebench/claim12_plan_critique_lift.py).

## Negative results

**C9. The workload is almost entirely reachable.** The cross-tier
oracle union resolves 148/150 (98.7%); only 2 matplotlib instances
(`matplotlib-22711` and `matplotlib-25498`) are unreachable across all
tiers tested. Opus alone covers 143/150, and the cross-vendor portfolio
adds 5 more. Scorer
[`benchmarks/swebench/ceiling_union.py`](benchmarks/swebench/ceiling_union.py).

**C10. sympy is scored under its canonical `bin/test` runner.** The
sympy testbed is scored with sympy's own `bin/test` harness, validated
by reference patches; under it the Agentless oracle@k baseline resolves
31/77. Scorer
[`benchmarks/swebench/sympy_canonical_score.py`](benchmarks/swebench/sympy_canonical_score.py).

**C11. Tier escalation recovers a majority of doomed instances at low
cost multiples.** Against a \$0.82/resolve base, escalating an
agreement-doomed set to a stronger tier recovers most of it, and the
figure depends on *which* doomed set you escalate, not on a single
sequential cascade. Escalating the Haiku-doomed set to Opus recovers
40 of 49 (82%) @ \$1.47/resolve; the Sonnet-doomed set to Opus with one
config recovers 21 of 27 (78%) @ \$1.37, and with two configs 22 of 27
(81%) @ \$2.71. The band is \$1.37–\$2.71. These are distinct,
non-comparable escalation options over different doomed sets. Scorer
[`benchmarks/swebench/value_of_resolve.py`](benchmarks/swebench/value_of_resolve.py).

## Reproducibility notes

- [`benchmarks/verify_criteria.py`](benchmarks/verify_criteria.py) is
  the offline gate: it runs every scorer over the committed data,
  checks the paper's load-bearing numbers appear, and writes
  [`benchmarks/claims_results.json`](benchmarks/claims_results.json).
  Every claim reproduces offline (exit code 0).
- The evidence explorer
  ([`benchmarks/explorer/`](benchmarks/explorer/), launched by
  [`benchmarks/explore.sh`](benchmarks/explore.sh)) is a
  dependency-free static page: it shows each claim with its paper
  value, status, and reproduce-command, and lets you drill into any
  evaluation run (per-instance rows, resolved / cost, aggregates).
- Quality is measured by the SWE-bench resolution verdict (the
  patch's effect on the held-out test suite). The held-out verdict
  is used only to *score* - never inside the agent's control flow:
  deployable signals (reproduction scripts, regression suites, patch
  presence) drive every control decision.
- Cross-vendor absolute pass rates are not directly comparable across
  providers; the paper discloses this and reports cost as one modeled
  surface.

## Operational data not in this repository

A larger body of operational evaluation work - per-instance stream
archives (~1.2 GB), overnight resilience harnesses, gap audits, and
multi-week eval logs - is maintained outside this public repository.
If your work depends on understanding *how* the paper's numbers were
produced (rather than verifying *that* they reproduce), email
**research@anindaray.com**.

## Citing

```bibtex
@article{ray2026codecapsules,
  title  = {The Economics of Coding Agents: Calibrating the Cost-Quality Frontier},
  author = {Ray, Aninda},
  year   = {2026},
  note   = {arXiv preprint, forthcoming.},
  url    = {https://github.com/aray-17/code-capsules}
}
```

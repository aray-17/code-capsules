# Paper claims and supporting evidence

This file lists the load-bearing claims of *The Economics of Coding
Agents: Calibrating the Cost-Quality Frontier* and points each claim at
the artifact in this repository that backs it. The paper PDF is at
[`paper/paper.pdf`](paper/paper.pdf); the exact code state cited in the
paper is tagged
[`v1.0-arxiv`](https://github.com/aray-17/code-capsules/releases/tag/v1.0-arxiv).

Every claim below reproduces **offline** — no Docker, no API keys, no
model calls — from the committed evaluation data under [`evals/`](evals/).
Reproduce all of them at once:

```bash
python3 benchmarks/verify_criteria.py     # prints PASS/FAIL per claim; exits 0 iff all 12 reproduce
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
| Head-to-head vs. Agentless (Pareto) | Code-Capsules 136/300 resolved @ \$0.436/inst vs. Agentless 125/300 @ \$0.452 — **more resolved at lower cost** | [`benchmarks/swebench/h2h_combine_score.py`](benchmarks/swebench/h2h_combine_score.py) | [`evals/leakfree/tb_forcestage2_first150.jsonl`](evals/leakfree/tb_forcestage2_first150.jsonl), [`tb_forcestage2_second150.jsonl`](evals/leakfree/tb_forcestage2_second150.jsonl), [`evals/h2h_agentless_sonnet_n150_run1_20260528.jsonl`](evals/h2h_agentless_sonnet_n150_run1_20260528.jsonl) |
| Calibrated cross-tier Pareto menu | cost-min 57 @ \$0.48 · balanced 66 @ \$0.93 · quality 77 @ \$0.91 · quality-max 92 @ \$0.78 · ceiling 96 @ \$0.94 (quality-max out-resolves quality at lower \$/resolve) | [`benchmarks/swebench/cross_tier_pareto_menu.py`](benchmarks/swebench/cross_tier_pareto_menu.py) | one committed JSONL per cell: cost-min [`evals/p10_sonnet_p8c_n150_20260515T235913.jsonl`](evals/p10_sonnet_p8c_n150_20260515T235913.jsonl), balanced [`evals/leakfree/tb_forcestage2_first150.jsonl`](evals/leakfree/tb_forcestage2_first150.jsonl), quality (leak-free re-run) [`evals/leakfree/exp4_lever_floor100_siginject.jsonl`](evals/leakfree/exp4_lever_floor100_siginject.jsonl), quality-max [`evals/p9_opus_implicit20_n150_combined_20260527.jsonl`](evals/p9_opus_implicit20_n150_combined_20260527.jsonl), ceiling [`evals/p9_opus_floor_n150_combined_20260527.jsonl`](evals/p9_opus_floor_n150_combined_20260527.jsonl) |

## Mechanism claims

**C1. Diverse-sample agreement is a deployable doomed-instance
signal.** A diverse pair that agrees on failure marks an instance
doomed at this tier with 98% in-family precision (fires on 65, one
false abandon, 100% recall); an independent re-derivation holds at
75.4%. The signal is anti-circular: Agentless recovers 0 of the
doomed set. Paper §agreement; scorer
[`benchmarks/swebench/agreement_signal.py`](benchmarks/swebench/agreement_signal.py).

**C2. The run-both lever lands within one instance of the oracle —
but is a first-class negative.** Running a diverse pair and selecting
the verified winner reaches 85/150 @ \$118.75, within one instance of
the 86/150 oracle union, at roughly twice the cost of its best single
member. Ship precision 58% (first-150) / 67% (held-out). Reported as a
negative: it adds zero resolves over a single configuration. Scorer
[`benchmarks/swebench/oracle_governor.py`](benchmarks/swebench/oracle_governor.py).

**C3. A regression-suite gate restores wrongly-abandoned resolves at
zero added cost.** Conditioning the abandon decision on a green
regression suite recovers wrong-abandons: first-150 (64, 70, 77),
held-out (76, 77, 87); P(resolve | green) = 0.745; candidate AUC
0.81 → 0.64 with a headline 0.83. Scorer
[`benchmarks/swebench/regression_gate.py`](benchmarks/swebench/regression_gate.py).

**C4. A model-tier escalation gate separates worth-escalating
instances.** Gate-positive instances resolve on escalation 20/21 vs.
3/17 gate-negative; 84/150 @ \$1.60/resolve over a \$1.40 base; held-out
37%. Scorer
[`benchmarks/swebench/escalation_gate.py`](benchmarks/swebench/escalation_gate.py).

**C5. Two-pass critique is Pareto-favorable vs. Agentless.** Same
models, data, and scorer: 136/300 @ \$0.436 vs. 125/300 @ \$0.452 —
more resolved at lower per-instance cost. Scorer
[`benchmarks/swebench/h2h_combine_score.py`](benchmarks/swebench/h2h_combine_score.py).

**C6. The shipped policy menu is a calibrated cross-tier Pareto
frontier.** Five presets span the frontier (numbers above); quality-max
out-resolves quality at a lower cost per resolve. Scorer
[`benchmarks/swebench/cross_tier_pareto_menu.py`](benchmarks/swebench/cross_tier_pareto_menu.py);
the shipped presets are in [`policy.yaml`](policy.yaml).

**C8. Cross-vendor cost ratios on HumanEval / MBPP.** Pass rates
HumanEval 98.2% / 92.7%, MBPP 53.0% / 50.2%; modeled cost ratios 4.8×
(HumanEval) and 15.9× (MBPP). Scorer
[`benchmarks/cross_vendor/score.py`](benchmarks/cross_vendor/score.py);
evidence [`evals/cross_vendor_humaneval_n164_postfix_20260525.csv`](evals/cross_vendor_humaneval_n164_postfix_20260525.csv),
[`evals/cross_vendor_mbpp_n500_postfix_20260525.csv`](evals/cross_vendor_mbpp_n500_postfix_20260525.csv).

**C12. Plan-then-execute helps every model; critique helps only some.**
Over the signaled-budget baseline, plan-then-execute adds +6.7pp on both
Sonnet (37.3 → 44.0%) and gpt-5-codex (20.7 → 27.3%); two-pass critique
adds further on Sonnet but is flat on gpt-5-codex (−0.7pp). Scorer
[`benchmarks/swebench/claim12_plan_critique_lift.py`](benchmarks/swebench/claim12_plan_critique_lift.py).

## Negative results

**C7. Evaluation-gated leakage deflated a reported cost by ~1.32×.** An
earlier two-pass configuration consulted the benchmark's held-out
verdict to skip its second pass, deflating its measured per-instance
cost from \$0.408 (honest, always-run) to \$0.308 (leaky) — a factor of
1.32. The corrected, leak-free configuration runs both passes
unconditionally. Scorer
[`benchmarks/swebench/leakage_cost_deflation_gate.py`](benchmarks/swebench/leakage_cost_deflation_gate.py).

**C9. There is a workload capability ceiling.** The cross-tier oracle
union resolves 103/150 (68.7%); 47 instances are doomed at every tier
tested, and Opus alone covers 101 of the 103. Scorer
[`benchmarks/swebench/ceiling_union.py`](benchmarks/swebench/ceiling_union.py).

**C10. A scoring bug, not a capability ceiling, explained sympy 0/77.**
Scoring sympy with `python -m pytest` (which sympy's testbed lacks)
scored 0/77; the canonical `bin/test` harness recovers Agentless to
31/77. Scorer
[`benchmarks/swebench/sympy_evalbug_gate.py`](benchmarks/swebench/sympy_evalbug_gate.py).

**C11. Tier escalation has steeply diminishing economics.** Against a
\$1.40/resolve base (the 2-config diverse-select), escalating an
agreement-doomed set to a stronger tier costs far more per recovered
resolve — and the figure depends on *which* doomed set you escalate, not
on a single sequential cascade. Escalating the Haiku-doomed set to Opus
is the cheapest at ~\$2.20/recovered resolve; escalating the Sonnet-doomed
set to Opus costs ~\$3.07–\$7.51 depending on how many configs you run.
These are distinct, non-comparable escalation options over different
doomed sets. Scorer
[`benchmarks/swebench/value_of_resolve.py`](benchmarks/swebench/value_of_resolve.py).

## Reproducibility notes

- [`benchmarks/verify_criteria.py`](benchmarks/verify_criteria.py) is
  the offline gate: it runs every scorer over the committed data,
  checks the paper's load-bearing numbers appear, and writes
  [`benchmarks/claims_results.json`](benchmarks/claims_results.json).
  All 12 claims reproduce (exit code 0).
- The evidence explorer
  ([`benchmarks/explorer/`](benchmarks/explorer/), launched by
  [`benchmarks/explore.sh`](benchmarks/explore.sh)) is a
  dependency-free static page: it shows each claim with its paper
  value, status, and reproduce-command, and lets you drill into any
  evaluation run (per-instance rows, resolved / cost, aggregates).
- Quality is measured by the SWE-bench resolution verdict (the
  patch's effect on the held-out test suite). The held-out verdict
  is used only to *score* — never inside the agent's control flow.
  The leakage that violated this is documented as C7.
- Cross-vendor absolute pass rates are not directly comparable across
  providers; the paper discloses this and reports cost as one modeled
  surface.

## Operational data not in this repository

A larger body of operational evaluation work — per-instance stream
archives (~1.2 GB), overnight resilience harnesses, gap audits, and
multi-week eval logs — is maintained outside this public repository.
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

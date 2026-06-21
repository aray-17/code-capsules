# Benchmarks — claim reproduction harness

This directory is the **source of validation** for the paper's key claims. Every scorer
here reproduces a number the paper states, **offline** from the committed evaluation data
under `../evals/` — no Docker, no API keys, no model calls.

## Verify every claim at once

```bash
python3 benchmarks/verify_criteria.py
```

This runs each claim's scorer, checks that the paper's load-bearing numbers appear in the
output, prints a PASS/FAIL line per claim, and writes `benchmarks/claims_results.json`.
Exit code is 0 iff every claim reproduces.

## Browse the claims (web app)

```bash
# from the repo root:
python3 -m http.server 8000
# then open http://localhost:8000/benchmarks/explorer/
```

`explorer/index.html` is a dependency-free static page (also GitHub-Pages hostable). It
loads `claims_results.json`, shows each claim with its paper value, status, and the exact
command to reproduce it, and lets you click an evidence `.jsonl` to inspect the per-instance
rows (resolved / cost / mode) and the aggregate (n, resolved, mean cost). `file://` blocks
`fetch`, so serve over HTTP.

## Individual scorers

All 12 key claims have a reproduction scorer (run `verify_criteria.py` for the gate; the
table below is the per-claim index).

| Claim | Scorer | Reproduces |
|---|---|---|
| Diverse-agreement governor | `swebench/agreement_signal.py` | best in-family pair (floor+siginject) 90.9% precision, fires on 22 with 2 false terminations; single-config 64.5%; median pair 66.7%; out-of-family independent precision ~22%; oracle union 106/126 (tier-local escalation trigger) |
| Diverse-select governor | `swebench/oracle_governor.py` | two-config diverse-select 104/126 @ $85.02 ($0.82/res); single-tier run-both retired (84/150 @ ~$1.41/res, dominated by the unbounded floor 111/150 @ $0.63/res); ship precision 75% (first-150) / 77% (held-out) |
| Regression-suite gate | `swebench/regression_gate.py` | honest single-tier 64 → regok recovers 15 of 28 wrong-abandons → 84 → 99 (12 short of the unbounded floor 111); P(resolve\|green)=0.792 |
| Model-tier escalation | `swebench/escalation_gate.py` | strong tier recovers 21/21 gate-positive and 15/17 gate-negative → selectivity collapses; reaches 85/150; base operating cost $0.82/res |
| Head-to-head vs Agentless | `swebench/h2h_combine_score.py --no-write` | Code-Capsules two-pass 172/300 (57.3%) @ $0.436 vs Agentless oracle@k 152/300 (50.7%) @ $0.452; +20 (+6.7pp), McNemar p=0.0055, 14.7% lower $/res |
| Cross-tier Pareto menu | `swebench/cross_tier_pareto_menu.py` | cost_min 80/$0.34, balanced 98/$0.63, quality 111/$0.64, quality_max 128/$0.56, ceiling 138/$0.65 |
| Leakage cost deflation | `swebench/leakage_cost_deflation_gate.py` | leaky $0.308 vs honest $0.408 (counts 70 vs 66); ratio 1.32× (see nit below) |
| Cross-vendor HE/MBPP | `cross_vendor/score.py` | HE 98.2/92.7%, MBPP 53.0/50.2%, ratios 4.8× / 15.9× |
| Workload ceiling | `swebench/ceiling_union.py` | oracle union 148/150 (98.7%); only 2 matplotlib instances unreachable; Opus-only union 143/150; cross-vendor adds 5 |
| sympy eval-bug | `swebench/sympy_evalbug_gate.py` | Agentless sympy 0/77 (pytest) → 31/77 (bin/test) |
| Plan-helps-all / critique-adds-lift | `swebench/claim12_plan_critique_lift.py` | Sonnet & codex floor→plan +7pp; two-pass critique adds lift on both vendors (Sonnet +3pp, codex +12pp) |
| Value-of-resolve rule | `swebench/value_of_resolve.py` | base $0.82/res; Haiku-doomed→Opus 40/49 (82%) @ $1.47; Sonnet-doomed→Opus(1) 21/27 (78%) @ $1.37; Sonnet-doomed→Opus(2) 22/27 (81%) @ $2.71 (band $1.37-$2.71) |

Each scorer is standalone and prints the paper's stated value next to the value it computes,
so a discrepancy is visible rather than hidden. `leakage_cost_deflation_gate.py` confirms the
deflation ratio is **1.32×**, matching the paper; the underlying costs $0.308 vs $0.408
reproduce exactly. All load-bearing numbers are clean.

All scorers live under `benchmarks/` (`benchmarks/swebench/` and `benchmarks/cross_vendor/`);
each runs as a standalone script from the repo root.

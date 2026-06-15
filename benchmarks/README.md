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
| Diverse-agreement governor | `swebench/agreement_signal.py` | 98% in-family precision, recall 100%, Agentless recovers 0 (anti-circularity) |
| Oracle-relative governor | `swebench/oracle_governor.py` | lever 85/150 @ $118.75 = oracle 86 within one instance; ship precision 58% / 67% |
| Regression-suite gate | `swebench/regression_gate.py` | first-150 64/70/77, held-out 76/77/87, P(resolve\|green)=0.745, AUC 0.81→0.64 / 0.83 |
| Model-tier escalation | `swebench/escalation_gate.py` | 20/21 vs 3/17, 84/150 @ $1.60, base $1.40, close 11/17 vs lost 4/32 |
| Head-to-head vs Agentless | `swebench/h2h_combine_score.py --no-write` | 136/300 @ $0.436 vs 125/300 @ $0.452 (Pareto) |
| Cross-tier Pareto menu | `swebench/cross_tier_pareto_menu.py` | cost_min 57/$0.48, quality 77/$0.91, quality_max 92/$0.78, ceiling 96/$0.94 |
| Leakage cost deflation | `swebench/leakage_cost_deflation_gate.py` | leaky $0.308 vs honest $0.408 (counts 70 vs 66); ratio 1.32× (see nit below) |
| Cross-vendor HE/MBPP | `cross_vendor/score.py` | HE 98.2/92.7%, MBPP 53.0/50.2%, ratios 4.8× / 15.9× |
| Workload ceiling | `swebench/ceiling_union.py` | cross-tier oracle union 103/150 (68.7%); 47 doomed; Opus alone 101/103 |
| sympy eval-bug | `swebench/sympy_evalbug_gate.py` | Agentless sympy 0/77 (pytest) → 31/77 (bin/test) |
| Plan-helps-all / critique-some | `swebench/claim12_plan_critique_lift.py` | Sonnet & codex floor→plan +7pp; two-pass flat on codex |
| Value-of-resolve rule | `swebench/value_of_resolve.py` | base $1.40, escalation $2.20 / $3.07 / $7.51 per resolve |

Each scorer is standalone and prints the paper's stated value next to the value it computes,
so a discrepancy is visible rather than hidden. Two scorers surface verified **paper nits**
(numbers untouched here, logged for reconciliation): `regression_gate.py` flags a *secondary*
scope caveat (`0.83 → 0.67` abandon-subset pairing spans two aggregation conventions), and
`leakage_cost_deflation_gate.py` confirms the deflation ratio is **1.32×**, matching the paper
(corrected from a 1.31× rounding typo); the underlying costs $0.308 vs $0.408 reproduce exactly. All
load-bearing numbers are clean.

All scorers live under `benchmarks/` (`benchmarks/swebench/` and `benchmarks/cross_vendor/`);
each runs as a standalone script from the repo root.

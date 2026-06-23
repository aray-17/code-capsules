# Benchmarks: claim reproduction harness

This directory is the **source of validation** for the paper's claims. Every scorer
here reproduces a number the paper states, **offline** from the committed evaluation
data under `../evals/` (no Docker, no API keys, no model calls).

The result numbers themselves live in one place, [`../CLAIMS.md`](../CLAIMS.md)
(claim by claim, with the backing evidence files). This README is the harness guide:
how to run the gate, how to browse the evidence, and which scorer backs which claim.

## Verify every claim at once

```bash
python3 benchmarks/verify_criteria.py
```

This runs each claim's scorer, checks that the paper's load-bearing numbers appear in
the output, prints a PASS/FAIL line per claim, and writes `benchmarks/claims_results.json`.
Exit code is 0 iff every claim reproduces.

## Browse the claims (web app)

```bash
bash benchmarks/explore.sh        # builds the data, serves localhost, opens the browser
```

`explore.sh` runs `verify_criteria.py` and `build_evidence_index.py`, then serves
`explorer/index.html`, a dependency-free static page (also GitHub-Pages hostable). It
loads `claims_results.json`, shows each claim with its paper value, status, and the
exact reproduce command, and lets you click an evidence `.jsonl` to inspect the
per-instance rows (resolved / cost / mode) and the aggregate (n, resolved, mean cost).
`file://` blocks `fetch`, so serve over HTTP (which `explore.sh` does for you).

## Claim index

Each claim has one standalone scorer; `verify_criteria.py` runs them all. For the
reproduced numbers and the backing evidence files, see [`../CLAIMS.md`](../CLAIMS.md).

| Claim | Scorer | What it checks |
|---|---|---|
| Diverse-sample agreement signal | `swebench/agreement_signal.py` | the diverse-pair agreement signal as a tier-local escalation trigger (precision, fire rate, oracle union) |
| Run-both lever vs. oracle | `swebench/oracle_governor.py` | the diverse-select lever against the per-instance oracle union, and its ship precision |
| Regression-suite gate | `swebench/regression_gate.py` | conditioning the abandon decision on a green regression suite recovers wrong-abandons |
| Model-tier escalation gate | `swebench/escalation_gate.py` | the strong tier recovers gate-positive and gate-negative alike, so selectivity collapses |
| Head-to-head vs. Agentless | `swebench/h2h_canonical_score.py` | two-pass critique vs. the Agentless baseline at full-300: more resolved at lower cost (Pareto), with McNemar significance |
| Cross-tier Pareto menu | `swebench/cross_tier_pareto_menu.py` | the five shipped presets span a calibrated cost-quality frontier |
| Cross-vendor HumanEval / MBPP | `cross_vendor/score.py` | pass rates and modeled cost ratios across vendors |
| Workload capability ceiling | `swebench/ceiling_union.py` | the cross-tier oracle union, and which instances are unreachable |
| sympy canonical scoring | `swebench/sympy_canonical_score.py` | sympy scored under its canonical `bin/test` runner (Agentless 31/77) |
| Value-of-resolve rule | `swebench/value_of_resolve.py` | escalating a doomed set to a stronger tier, by which doomed set, at what cost per recovered resolve |
| Plan + critique lift across vendors | `swebench/claim12_plan_critique_lift.py` | plan-then-execute and two-pass critique each add lift on both vendors |
| Held-out cascade generalization | `swebench/held_out_cascade.py` | the value-of-resolve cascade generalizes to the held-out (off-django) split |

Each scorer prints the paper's stated value next to the value it computes, so a
discrepancy is visible rather than hidden. All scorers live under `benchmarks/`
(`benchmarks/swebench/` and `benchmarks/cross_vendor/`); each runs as a standalone
script from the repo root.

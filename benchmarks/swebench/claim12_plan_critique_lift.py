#!/usr/bin/env python3
"""Claim 12 scorer: plan-then-execute helps every model; critique helps both.

Paper claim (paper.tex):
  - Abstract / contributions: the data-axis scaffold (plan-then-execute) lifts
    Sonnet vs the comparable baseline, and the iteration-axis scaffold (two-pass
    critique) adds further lift beyond plan-then-execute on both vendors.

  - Cross-vendor variant table (Table 2), SWE-bench Lite n=150:
        signaled budget (b=10)   Sonnet 84/150 (56.0%)   codex 46/150 (30.7%)
        plan-then-execute (b=20) Sonnet 93/150 (62.0%)   codex 49/150 (32.7%)
        two-pass critique        Sonnet 98/150 (65.3%)   codex 66/150 (44.0%)

  - Cross-vendor findings: the codex variant ladder spans a 13pp band at n=150
    (30.7 -> 32.7 -> 44.0%); two-pass critique lands on top of plan-then-execute
    on both vendors.

Derived targets reproduced here (all OFFLINE from committed eval JSONL):
  plan-then-execute lift over signaled budget:
      Sonnet: 62.0 - 56.0 = +6.0 pp
      codex : 32.7 - 30.7 = +2.0 pp
  two-pass critique delta over plan-then-execute (adds lift on both vendors):
      Sonnet: 65.3 - 62.0 = +3.3 pp
      codex : 44.0 - 32.7 = +11.3 pp

Schema: each eval file is JSONL, one row per SWE-bench Lite instance with a
boolean ``resolved`` field and a ``cost_usd`` field. Pass rate = resolved/total.

Run:
  PYTHONNOUSERSITE=1 PYTHONPATH=src python3 \
    benchmarks/swebench/claim12_plan_critique_lift.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals"

# (label, vendor, variant, filename) -- the n=150 cells in Table 2.
# Codex two-pass uses the postfix run reported in Table 2 (66/150).
CELLS = [
    ("signaled budget (b=10)",   "Sonnet", "signaled",
     "p10_sonnet_signaled10_n150_20260515T231227.jsonl"),
    ("plan-then-execute (b=20)",  "Sonnet", "planFirst",
     "p10_sonnet_planFirst_b20_n150_20260516T152721.jsonl"),
    ("two-pass critique",         "Sonnet", "twopass",
     "leakfree/tb_forcestage2_first150.jsonl"),
    ("signaled budget (b=10)",   "codex",  "signaled",
     "swe_codex_signaled10_n150_combined_postfix_20260526.jsonl"),
    ("plan-then-execute (b=20)",  "codex",  "planFirst",
     "swe_codex_planFirst20_n150_combined_postfix_20260525.jsonl"),
    ("two-pass critique",         "codex",  "twopass",
     "swe_codex_twoPass_n150_postfix_20260525.jsonl"),
]

# Paper's stated values for each cell: (resolved, total, pass_pct, cost).
PAPER_CELLS = {
    ("Sonnet", "signaled"):  (84, 150, 56.0, 0.174),
    ("Sonnet", "planFirst"): (93, 150, 62.0, 0.287),
    ("Sonnet", "twopass"):   (98, 150, 65.3, 0.408),
    ("codex",  "signaled"):  (46, 150, 30.7, 0.142),
    ("codex",  "planFirst"): (49, 150, 32.7, 0.297),
    ("codex",  "twopass"):   (66, 150, 44.0, 0.378),
}


def tally(path):
    n = resolved = 0
    cost_sum = 0.0
    cost_n = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            if row.get("resolved"):
                resolved += 1
            c = row.get("cost_usd")
            if isinstance(c, (int, float)):
                cost_sum += c
                cost_n += 1
    mean_cost = cost_sum / cost_n if cost_n else float("nan")
    return n, resolved, mean_cost


def main():
    results = {}  # (vendor, variant) -> (n, resolved, pct, cost)
    all_ok = True

    print("=" * 78)
    print("CLAIM 12 -- plan-then-execute helps every model; critique helps both")
    print("=" * 78)
    print()
    print("Per-cell reproduction (computed vs paper Table 2, n=150):")
    print("-" * 78)
    hdr = (f"{'vendor':7s} {'variant':10s} "
           f"{'computed':>16s}  {'paper':>16s}  {'pass match':>10s}")
    print(hdr)
    print("-" * 78)

    for label, vendor, variant, fname in CELLS:
        n, res, cost = tally(EVALS / fname)
        pct = 100.0 * res / n
        results[(vendor, variant)] = (n, res, pct, cost)

        p_res, p_tot, p_pct, p_cost = PAPER_CELLS[(vendor, variant)]
        pass_match = (res == p_res and n == p_tot)
        all_ok = all_ok and pass_match
        comp = f"{res}/{n} ({pct:.1f}%)"
        papr = f"{p_res}/{p_tot} ({p_pct:.1f}%)"
        flag = "OK" if pass_match else "MISMATCH"
        print(f"{vendor:7s} {variant:10s} {comp:>16s}  {papr:>16s}  {flag:>10s}")
        # cost is a softer check (cost-model dependent); report side by side.
        print(f"        {'cost/attempt':10s} "
              f"{'$'+format(cost,'.3f'):>16s}  {'$'+format(p_cost,'.3f'):>16s}")

    print("-" * 78)
    print()

    # --- Derived claim: plan-then-execute lift on BOTH models ---
    print("Derived claim A -- plan-then-execute lift over signaled baseline:")
    print("-" * 78)
    for vendor in ("Sonnet", "codex"):
        base = results[(vendor, "signaled")][2]
        plan = results[(vendor, "planFirst")][2]
        lift = plan - base
        # paper-derived expected lift
        pbase = PAPER_CELLS[(vendor, "signaled")][2]
        pplan = PAPER_CELLS[(vendor, "planFirst")][2]
        plift = pplan - pbase
        positive = lift > 0.0  # plan-then-execute helps
        print(f"  {vendor:7s}: signaled {base:5.1f}%  ->  planFirst {plan:5.1f}% "
              f"=  +{lift:.1f} pp   (paper +{plift:.1f} pp; helps: "
              f"{'YES' if positive else 'NO'})")
    print()

    # --- Derived claim: two-pass critique adds lift on BOTH vendors ---
    print("Derived claim B -- two-pass critique delta over plan-then-execute:")
    print("-" * 78)
    for vendor in ("Sonnet", "codex"):
        plan = results[(vendor, "planFirst")][2]
        twop = results[(vendor, "twopass")][2]
        delta = twop - plan
        pplan = PAPER_CELLS[(vendor, "planFirst")][2]
        ptwop = PAPER_CELLS[(vendor, "twopass")][2]
        pdelta = ptwop - pplan
        adds = delta > 0.0  # two-pass critique adds lift
        print(f"  {vendor:7s}: planFirst {plan:5.1f}%  ->  twopass {twop:5.1f}% "
              f"=  {delta:+.1f} pp   (paper {pdelta:+.1f} pp; adds lift: "
              f"{'YES' if adds else 'NO'})")
    print()

    # --- codex variant ladder spans a ~13pp band ---
    cs = results[("codex", "signaled")][2]
    cp = results[("codex", "planFirst")][2]
    ct = results[("codex", "twopass")][2]
    band = max(cs, cp, ct) - min(cs, cp, ct)
    print("Derived claim C -- codex n=150 ladder spans a ~13pp band:")
    print("-" * 78)
    print(f"  codex: signaled {cs:.1f}% -> planFirst {cp:.1f}% -> "
          f"twopass {ct:.1f}%   band = {band:.1f} pp "
          f"(paper: 30.7 -> 32.7 -> 44.0%, ~13pp band)")
    print()

    print("=" * 78)
    print(f"ALL PER-CELL PASS COUNTS MATCH PAPER: {all_ok}")
    print("=" * 78)
    return all_ok


if __name__ == "__main__":
    main()
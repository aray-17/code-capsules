#!/usr/bin/env python3
"""Claim 12 scorer: plan-then-execute helps every model; critique helps only some.

Paper claim (paper.tex):
  - Abstract / contributions (lines 225-230, 2116-2126): the data-axis scaffold
    (plan-then-execute) lifts BOTH Sonnet and codex by approximately SEVEN
    percentage points vs the comparable baseline; the iteration-axis scaffold
    (two-pass critique) adds no lift beyond plan-then-execute that clears the
    noise band on either vendor.

  - Cross-vendor variant table (Table 2, lines 2076-2087), SWE-bench Lite n=150:
        signaled budget (b=10)   Sonnet 56/150 (37.3%)   codex 31/150 (20.7%)
        plan-then-execute (b=20) Sonnet 66/150 (44.0%)   codex 41/150 (27.3%)
        two-pass critique        Sonnet 70/150 (46.7%)   codex 40/150 (26.7%)

  - Cross-vendor findings (lines 2119-2126): the codex variant ladder collapses
    into a 6pp band at n=150 (20.7 -> 27.3 -> 26.7%); plan-then-execute and
    two-pass tied near 27% above signaled-budget at 21%.

Derived targets reproduced here (all OFFLINE from committed eval JSONL):
  plan-then-execute lift over signaled budget:
      Sonnet: 44.0 - 37.3 = +6.7 pp   (~7 pp)
      codex : 27.3 - 20.7 = +6.6 pp   (~7 pp)
  two-pass critique delta over plan-then-execute (the "flat on codex" point):
      codex : 26.7 - 27.3 = -0.6 pp   (flat, within noise)
      Sonnet: 46.7 - 44.0 = +2.7 pp   (small positive)

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
# Codex two-pass uses the *postfix* run, the single run reported in Table 2
# (40/150); the three codex two-pass runs are 48/40/42 (paper sec. 10.4).
CELLS = [
    ("signaled budget (b=10)",   "Sonnet", "signaled",
     "p10_sonnet_signaled10_n150_20260515T231227.jsonl"),
    ("plan-then-execute (b=20)",  "Sonnet", "planFirst",
     "p10_sonnet_planFirst_b20_n150_20260516T152721.jsonl"),
    ("two-pass critique",         "Sonnet", "twopass",
     "p10_sonnet_twopass_b10p15_n150_20260516T152721.jsonl"),
    ("signaled budget (b=10)",   "codex",  "signaled",
     "swe_codex_signaled10_n150_combined_postfix_20260526.jsonl"),
    ("plan-then-execute (b=20)",  "codex",  "planFirst",
     "swe_codex_planFirst20_n150_combined_postfix_20260525.jsonl"),
    ("two-pass critique",         "codex",  "twopass",
     "swe_codex_twoPass_n150_postfix_20260525.jsonl"),
]

# Paper's stated values for each cell: (resolved, total, pass_pct, cost).
PAPER_CELLS = {
    ("Sonnet", "signaled"):  (56, 150, 37.3, 0.174),
    ("Sonnet", "planFirst"): (66, 150, 44.0, 0.287),
    ("Sonnet", "twopass"):   (70, 150, 46.7, 0.308),
    ("codex",  "signaled"):  (31, 150, 20.7, 0.142),
    ("codex",  "planFirst"): (41, 150, 27.3, 0.297),
    ("codex",  "twopass"):   (40, 150, 26.7, 0.378),
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
    print("CLAIM 12 -- plan-then-execute helps every model; critique helps some")
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

    # --- Derived claim: plan-then-execute lift (~7pp) on BOTH models ---
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
        near7 = abs(lift - 7.0) <= 1.5  # "approximately seven pp"
        print(f"  {vendor:7s}: signaled {base:5.1f}%  ->  planFirst {plan:5.1f}% "
              f"=  +{lift:.1f} pp   (paper +{plift:.1f} pp; ~7pp: "
              f"{'YES' if near7 else 'NO'})")
    print()

    # --- Derived claim: two-pass critique delta vs plan-then-execute ---
    print("Derived claim B -- two-pass critique delta over plan-then-execute:")
    print("-" * 78)
    for vendor in ("Sonnet", "codex"):
        plan = results[(vendor, "planFirst")][2]
        twop = results[(vendor, "twopass")][2]
        delta = twop - plan
        pplan = PAPER_CELLS[(vendor, "planFirst")][2]
        ptwop = PAPER_CELLS[(vendor, "twopass")][2]
        pdelta = ptwop - pplan
        # "flat" = within the ~+-4pp n=150 binomial noise band
        flat = abs(delta) <= 4.0
        print(f"  {vendor:7s}: planFirst {plan:5.1f}%  ->  twopass {twop:5.1f}% "
              f"=  {delta:+.1f} pp   (paper {pdelta:+.1f} pp; flat<=4pp: "
              f"{'YES' if flat else 'NO'})")
    print()

    # --- codex variant ladder collapse into a ~6pp band ---
    cs = results[("codex", "signaled")][2]
    cp = results[("codex", "planFirst")][2]
    ct = results[("codex", "twopass")][2]
    band = max(cs, cp, ct) - min(cs, cp, ct)
    print("Derived claim C -- codex n=150 ladder collapses into a ~6pp band:")
    print("-" * 78)
    print(f"  codex: signaled {cs:.1f}% -> planFirst {cp:.1f}% -> "
          f"twopass {ct:.1f}%   band = {band:.1f} pp "
          f"(paper: 20.7 -> 27.3 -> 26.7%, ~6pp band)")
    print()

    print("=" * 78)
    print(f"ALL PER-CELL PASS COUNTS MATCH PAPER: {all_ok}")
    print("=" * 78)
    return all_ok


if __name__ == "__main__":
    main()
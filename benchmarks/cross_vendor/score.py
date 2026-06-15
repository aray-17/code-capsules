#!/usr/bin/env python3
"""
Scorer for CLAIM 8 (SOFT): cross-vendor HumanEval/MBPP cost ratios + pass rates.

Reproduces Table tab:cross_vendor_hemmbpp (paper.tex L2023-2026) and the cost
ratios in paper.tex L2031:

    HumanEval gpt-5-codex      161/164 (98.2%) @ 0.0161
    HumanEval Gemini 2.5-flash 152/164 (92.7%) @ 0.0033
    MBPP      gpt-5-codex      265/500 (53.0%) @ 0.0219
    MBPP      Gemini 2.5-flash 251/500 (50.2%) @ 0.0014
    Cost ratios: 4.8x (HumanEval), 15.9x (MBPP), in Gemini's favor.

Ratios are computed on UNROUNDED per-task mean costs. The table rounds $/instance
to 4 decimal places; ratios of the *displayed* (rounded) costs would be ~4.9x /
~15.6x, so the scorer prints both to show only the unrounded ratio matches.

Run:
    PYTHONNOUSERSITE=1 PYTHONPATH=src python3 score_claim8.py
"""

import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = {
    "HumanEval": os.path.join(
        REPO, "evals/cross_vendor_humaneval_n164_postfix_20260525.csv"
    ),
    "MBPP": os.path.join(
        REPO, "evals/cross_vendor_mbpp_n500_postfix_20260525.csv"
    ),
}

# provider value in CSV -> display label used in the paper
VENDOR_LABEL = {"openai": "gpt-5-codex", "gemini": "Gemini 2.5-flash"}


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def is_pass(v):
    return str(v).strip().lower() == "true"


def summarize(rows):
    """Return {provider: (n, n_pass, sum_cost)} aggregated over rows."""
    agg = {}
    for r in rows:
        p = r["provider"]
        n, npass, scost = agg.get(p, (0, 0, 0.0))
        n += 1
        npass += 1 if is_pass(r["resolved"]) else 0
        scost += float(r["cost_usd"])
        agg[p] = (n, npass, scost)
    return agg


def main():
    for bench, path in FILES.items():
        rows = load(path)
        agg = summarize(rows)
        print(f"=== {bench} ===")

        # ordered: codex (openai) first, then gemini, matching paper rows
        means = {}
        for prov in ("openai", "gemini"):
            n, npass, scost = agg[prov]
            mean_cost = scost / n
            means[prov] = mean_cost
            rate = 100.0 * npass / n
            print(
                f"  {VENDOR_LABEL[prov]:<18} "
                f"{npass}/{n} ({rate:.1f}%)  "
                f"mean_cost_unrounded={mean_cost!r}  "
                f"displayed(4dp)={mean_cost:.4f}"
            )

        # ratio in Gemini's favor = codex_cost / gemini_cost
        ratio_unrounded = means["openai"] / means["gemini"]
        ratio_displayed = round(means["openai"], 4) / round(means["gemini"], 4)
        print(
            f"  cost ratio (codex/gemini): "
            f"unrounded={ratio_unrounded:.4f}  ->  {ratio_unrounded:.1f}x   "
            f"| displayed-cost ratio={ratio_displayed:.4f} -> {ratio_displayed:.1f}x"
        )
        print()


if __name__ == "__main__":
    main()
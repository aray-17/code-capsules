#!/usr/bin/env python3
"""Standalone offline scorer for CLAIM 6 -- the calibrated cross-tier Pareto menu.

Reproduces, from committed eval JSONLs (no Docker / no API / no model calls), every
cell of the shipped-presets menu the paper states in Table~\ref{tab:shipped_presets}
(paper/paper.tex L693-697) and re-quotes in the abstract (L201-205) and
Section~\ref{sec:strong_tier_menu} (L1483-1489).

Target numbers (quoted from paper.tex, the source of truth):

        cost_min     signaled budget (Sonnet, b=10)    84/150  $0.17/att  $0.31/res
        balanced     two-pass critique (Sonnet,10->25) 98/150  $0.41/att  $0.62/res
  L695  quality      unbounded budget (Sonnet, b=100)  111/150 $0.47/att  $0.63/res
  L696  quality_max  implicit budget (Opus, b=20)      128/150 $0.48/att  $0.56/res
  L697  ceiling      unbounded budget (Opus, b=100)    138/150 $0.60/att  $0.66/res

Per-cell metrics:
  resolved = count of rows with a true resolved flag (gold-scored, offline)
  $/att    = sum(cost_usd) / n          (cost per attempt over all 150 instances)
  $/res    = sum(cost_usd) / resolved   (cost per resolved instance)

Data sources (one committed JSONL per cell):
  cost_min     evals/p10_sonnet_signaled10_n150_20260515T231227.jsonl
                 (the bare ten-turn signaled budget; dominates the relevance
                  ranker on Sonnet; flat per-instance schema, key 'resolved' / 'cost_usd')
  balanced     evals/leakfree/tb_forcestage2_first150.jsonl
                 (leak-free two-pass critique, second pass run UNCONDITIONALLY --
                  the always-run config; flat per-instance schema, cost is the
                  always-run cost.)
  quality      evals/p10_sonnet_floor_n150_20260516T040248.jsonl
                 (unbounded Sonnet floor@100; flat per-instance schema, key
                  'resolved' / 'cost_usd'. This is the sweep floor stream that
                  the cross-model table at paper L1380 prints and that
                  Table~\ref{tab:shipped_presets} and policy.yaml quote for the
                  'quality' cell. See [SOURCE NOTE] below.)
  quality_max  evals/p9_opus_implicit20_n150_combined_20260527.jsonl
                 (Opus implicit budget b=20; flat per-instance schema)
  ceiling      evals/p9_opus_floor_n150_combined_20260527.jsonl
                 (Opus unbounded floor b=100; flat per-instance schema)

[SOURCE NOTE -- which floor file is the menu cell]
  The 'quality' cell (111/150) is scored from the sweep floor stream
  p10_sonnet_floor_n150_20260516T040248.jsonl. This is the floor stream the
  cross-model table at paper L1380 prints, and Table~\ref{tab:shipped_presets}
  and policy.yaml quote it for the 'quality' cell. Per-attempt cost is $0.4663.

Run with: PYTHONNOUSERSITE=1 PYTHONPATH=src python3 \
            benchmarks/swebench/cross_tier_pareto_menu.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def score_flat(path):
    """Flat per-instance schema: top-level 'resolved' / 'cost_usd'."""
    rows = load_jsonl(path)
    n = len(rows)
    resolved = sum(1 for r in rows if r.get("resolved"))
    cost = sum((r.get("cost_usd") or 0.0) for r in rows)
    return n, resolved, cost


def score_candidate(path, cand_key):
    """Candidates-dict schema: per-row candidates[cand_key].gold_resolved / cost_usd."""
    rows = load_jsonl(path)
    n = len(rows)
    resolved = 0
    cost = 0.0
    for r in rows:
        c = r["candidates"].get(cand_key, {})
        if c.get("gold_resolved"):
            resolved += 1
        cost += (c.get("cost_usd") or 0.0)
    return n, resolved, cost


# (menu entry, description, scorer-callable, paper resolved, paper $/att, paper $/res)
CELLS = [
    (
        "cost_min",
        "signaled budget (Sonnet, b=10)",
        lambda: score_flat(ROOT / "evals/p10_sonnet_signaled10_n150_20260515T231227.jsonl"),
        84, 0.17, 0.31,
    ),
    (
        "balanced",
        "two-pass critique (Sonnet, 10->25)",
        lambda: score_flat(ROOT / "evals/leakfree/tb_forcestage2_first150.jsonl"),
        98, 0.41, 0.62,
    ),
    (
        "quality",
        "unbounded budget (Sonnet, b=100)",
        lambda: score_flat(
            ROOT / "evals/p10_sonnet_floor_n150_20260516T040248.jsonl"
        ),
        111, 0.47, 0.63,
    ),
    (
        "quality_max",
        "implicit budget (Opus, b=20)",
        lambda: score_flat(
            ROOT / "evals/p9_opus_implicit20_n150_combined_20260527.jsonl"
        ),
        128, 0.48, 0.56,
    ),
    (
        "ceiling",
        "unbounded budget (Opus, b=100)",
        lambda: score_flat(
            ROOT / "evals/p9_opus_floor_n150_combined_20260527.jsonl"
        ),
        138, 0.60, 0.66,
    ),
]


def approx(a, b, tol):
    return abs(a - b) <= tol


def main():
    print("=" * 84)
    print("CLAIM 6 -- calibrated cross-tier Pareto menu (paper Table tab:shipped_presets)")
    print("=" * 84)
    print(
        f"{'menu entry':<12} {'resolved (paper)':<18} "
        f"{'$/att (paper)':<22} {'$/res (paper)':<22} match"
    )
    print("-" * 84)

    all_ok = True
    for entry, desc, scorer, p_res, p_att, p_resc in CELLS:
        n, resolved, cost = scorer()
        att = cost / n if n else float("nan")
        resc = cost / resolved if resolved else float("nan")

        # resolved must be exact; costs match the paper's 2-decimal printed value
        ok_res = (resolved == p_res) and (n == 150)
        ok_att = approx(round(att, 2), p_att, 0.005)
        ok_res_cost = approx(round(resc, 2), p_resc, 0.005)
        ok = ok_res and ok_att and ok_res_cost
        all_ok = all_ok and ok

        res_str = f"{resolved}/{n} ({p_res}/150)"
        att_str = f"{att:.4f} ({p_att:.2f})"
        resc_str = f"{resc:.4f} ({p_resc:.2f})"
        flag = "OK" if ok else "MISMATCH"
        print(f"{entry:<12} {res_str:<18} {att_str:<22} {resc_str:<22} {flag}")
        print(f"             {desc}")

    print("-" * 84)
    print("ALL CELLS REPRODUCE" if all_ok else "ONE OR MORE CELLS DO NOT REPRODUCE")
    print("=" * 84)

    # ---- cross-check the headline abstract claim (L204-205) ----
    print()
    print("Abstract cross-check (L204-205): quality_max out-resolves quality at lower $/res")
    _, qm_res, qm_cost = score_flat(
        ROOT / "evals/p9_opus_implicit20_n150_combined_20260527.jsonl"
    )
    _, q_res, q_cost = score_flat(
        ROOT / "evals/p10_sonnet_floor_n150_20260516T040248.jsonl"
    )
    qm_resc = qm_cost / qm_res
    q_resc = q_cost / q_res
    print(
        f"  quality_max (Opus impl-20): {qm_res}/150 @ ${qm_resc:.2f}/res  vs  "
        f"quality (Sonnet floor): {q_res}/150 @ ${q_resc:.2f}/res"
    )
    more_resolved = qm_res > q_res
    cheaper_per_res = qm_resc < q_resc
    print(
        f"  more resolved? {more_resolved} (paper: yes, 128>111);  "
        f"cheaper per resolve? {cheaper_per_res} (paper: yes, $0.56<$0.63)"
    )
    print("  -> abstract claim holds" if (more_resolved and cheaper_per_res)
          else "  -> abstract claim DOES NOT hold")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
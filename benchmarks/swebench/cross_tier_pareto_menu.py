#!/usr/bin/env python3
"""Standalone offline scorer for CLAIM 6 -- the calibrated cross-tier Pareto menu.

Reproduces, from committed eval JSONLs (no Docker / no API / no model calls), every
cell of the shipped-presets menu the paper states in Table~\ref{tab:shipped_presets}
(paper/paper.tex L693-697) and re-quotes in the abstract (L201-205) and
Section~\ref{sec:strong_tier_menu} (L1483-1489).

Target numbers (quoted from paper.tex, the source of truth):

  L693  cost_min     relevance ranker (Sonnet, b=10)   57/150  $0.18/att  $0.48/res
  L694  balanced     two-pass critique (Sonnet,10->25) 66/150  $0.41/att  $0.93/res
  L695  quality      unbounded budget (Sonnet, b=100)  77/150  $0.47/att  $0.91/res
  L696  quality_max  implicit budget (Opus, b=20)      92/150  $0.48/att  $0.78/res
  L697  ceiling      unbounded budget (Opus, b=100)    96/150  $0.60/att  $0.94/res

Per-cell metrics:
  resolved = count of rows with a true resolved flag (gold-scored, offline)
  $/att    = sum(cost_usd) / n          (cost per attempt over all 150 instances)
  $/res    = sum(cost_usd) / resolved   (cost per resolved instance)

Data sources (one committed JSONL per cell):
  cost_min     evals/p10_sonnet_p8c_n150_20260515T235913.jsonl
                 ("p8c" == relevance-ranker stacked on a 10-turn signaled budget;
                  flat per-instance schema, key 'resolved' / 'cost_usd')
  balanced     evals/leakfree/tb_forcestage2_first150.jsonl
                 (leak-free two-pass critique, second pass run UNCONDITIONALLY --
                  the corrected always-run config; flat per-instance schema. This
                  file is ALREADY the corrected variant, so NO 1.31x deflation
                  factor is applied. The 1.31x factor in paper L1010-1012 applied
                  to the OLD eval-gated cost, not to this committed file.)
  quality      evals/leakfree/exp4_lever_floor100_siginject.jsonl
                 (LEAK-FREE re-run of the unbounded Sonnet floor@100; 'candidates'
                  dict schema -- the 'floor' candidate's gold_resolved / cost_usd.
                  Paper L1485 explicitly says "77/150 in the leak-free re-run".
                  NOTE: the older non-leak-free file
                  evals/p10_sonnet_floor_n150_20260516T040248.jsonl scores 75/150;
                  that pre-leak-free number is what the cross-model table at L1380
                  prints, NOT the shipped menu cell. See [DISCREPANCY] below.)
  quality_max  evals/p9_opus_implicit20_n150_combined_20260527.jsonl
                 (Opus implicit budget b=20; flat per-instance schema)
  ceiling      evals/p9_opus_floor_n150_combined_20260527.jsonl
                 (Opus unbounded floor b=100; flat per-instance schema)

[DISCREPANCY -- documented, not fudged]
  The 'quality' cell (77/150) reproduces ONLY from the leak-free re-run file
  exp4_lever_floor100_siginject.jsonl (floor candidate). The originally-named
  candidate file p10_sonnet_floor_n150_*.jsonl gives 75/150 -- that is the
  pre-leak-free floor (matches the cross-model table at L1380, 75/150 @ $0.466),
  not the shipped menu cell. Both files share the same per-attempt cost ($0.4663),
  so only the resolved count differs (leak-free re-run picks up +2). We report from
  the leak-free file because L1485 names it as the source of the menu's 77/150.

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
        "relevance ranker (Sonnet, b=10)",
        lambda: score_flat(ROOT / "evals/p10_sonnet_p8c_n150_20260515T235913.jsonl"),
        57, 0.18, 0.48,
    ),
    (
        "balanced",
        "two-pass critique (Sonnet, 10->25)",
        lambda: score_flat(ROOT / "evals/leakfree/tb_forcestage2_first150.jsonl"),
        66, 0.41, 0.93,
    ),
    (
        "quality",
        "unbounded budget (Sonnet, b=100) [leak-free re-run]",
        lambda: score_candidate(
            ROOT / "evals/leakfree/exp4_lever_floor100_siginject.jsonl", "floor"
        ),
        77, 0.47, 0.91,
    ),
    (
        "quality_max",
        "implicit budget (Opus, b=20)",
        lambda: score_flat(
            ROOT / "evals/p9_opus_implicit20_n150_combined_20260527.jsonl"
        ),
        92, 0.48, 0.78,
    ),
    (
        "ceiling",
        "unbounded budget (Opus, b=100)",
        lambda: score_flat(
            ROOT / "evals/p9_opus_floor_n150_combined_20260527.jsonl"
        ),
        96, 0.60, 0.94,
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
    _, q_res, q_cost = score_candidate(
        ROOT / "evals/leakfree/exp4_lever_floor100_siginject.jsonl", "floor"
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
        f"  more resolved? {more_resolved} (paper: yes, 92>77);  "
        f"cheaper per resolve? {cheaper_per_res} (paper: yes, $0.78<$0.91)"
    )
    print("  -> abstract claim holds" if (more_resolved and cheaper_per_res)
          else "  -> abstract claim DOES NOT hold")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
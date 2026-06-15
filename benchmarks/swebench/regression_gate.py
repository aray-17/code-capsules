#!/usr/bin/env python3
"""Standalone offline scorer for CLAIM 3 -- the regression-suite gate.

Reproduces, from the committed leak-free lever-evaluation JSONLs (no Docker / no API /
no model calls), every number the paper states for
Section "A regression-suite gate restores the wrong abandons"
(paper/paper.tex, sec:agreement_regression_gate) plus the governor-replay anchors.

Target numbers (quoted from paper.tex):
  - L195-197 / L1858 / L1873 / L1877:
      first-150  current / pure_regok / hybrid_regok = 64 / 70 / 77
      held-out   current / pure_regok / hybrid_regok = 76 / 77 / 87
      floor-alone (unbounded baseline) = 77 (first) and 90 (held-out)
  - L1875: "a regression-green candidate resolves with probability 0.745"
  - L1876: "the single feature separates resolving from non-resolving
            candidates at AUC 0.83"
  - L1884: "per-candidate AUC falls from 0.81 to 0.64 across splits
            (0.83 to 0.67 on the abandon subset)"
  - L1886-1887: "the probability that a gate-negative candidate in fact
            resolves rises from 0.11 to 0.37"

Replay semantics (identical to tests/test_governor_replay.py):
  current    = shipped_gold_resolved AND lever_decision != ABANDON
  pure_regok = shipped_gold_resolved AND selected candidate regression_ok is True
  hybrid     = OR of the two (regression_ok None NEVER rescues).

VERIFIED DISCREPANCY (printed in [6]): the paired abandon-subset AUC '0.83 to 0.67'
is reported across TWO different aggregation universes. No single universe yields
both 0.83 (first) and 0.67 (held). 0.83 = abandon/all-cand/drop-None (0.8371);
0.67 = abandon/selected/None=0 (0.6706). The headline 0.83 also matches only by
truncation of 0.8371 (round-half-up gives 0.84). All other gate numbers are clean.

AUC is the Mann-Whitney rank statistic, computed by hand with 0.5 tie credit
(cross-checked against a closed-form 2x2 contingency), so no sklearn/numpy needed.

Run with: PYTHONNOUSERSITE=1 PYTHONPATH=src python3 verify_regression_gate.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIRST150 = ROOT / "evals/leakfree/exp4_lever_floor100_siginject.jsonl"
HELD_OUT = ROOT / "evals/leakfree/exp4_lever_second150.jsonl"


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def selected(row):
    """The selected candidate sub-dict."""
    return row["candidates"][row["selected_config"]]


# -- governor replay (mirrors tests/test_governor_replay.py) -------------------
def replay(rows):
    current = sum(1 for r in rows
                  if r["shipped_gold_resolved"] and r["lever_decision"] != "ABANDON")
    pure = sum(1 for r in rows
               if r["shipped_gold_resolved"] and selected(r).get("regression_ok") is True)
    hybrid = sum(1 for r in rows
                 if r["shipped_gold_resolved"]
                 and (r["lever_decision"] != "ABANDON"
                      or selected(r).get("regression_ok") is True))
    return current, pure, hybrid


def floor_alone(rows):
    """Unbounded baseline = the floor candidate's gold verdict, always shipped."""
    return sum(1 for r in rows if r["candidates"].get("floor", {}).get("gold_resolved"))


# -- conditional resolve probabilities ----------------------------------------
def p_resolve(cands, want):
    """P(gold_resolved | regression_ok is `want`) over a candidate iterable."""
    sub = [c for c in cands if c.get("regression_ok") is want]
    res = sum(1 for c in sub if c.get("gold_resolved"))
    return res, len(sub), (res / len(sub) if sub else float("nan"))


# -- AUC of the single regression_ok feature ----------------------------------
def auc(pairs):
    """Rank AUC with 0.5 tie credit. pairs = [(score, label0/1), ...]."""
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return float("nan")
    c = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                c += 1.0
            elif sp == sn:
                c += 0.5
    return c / (len(pos) * len(neg))


def feature_pairs(cands, drop_none=True, none_score=0.0):
    """(regression_ok-as-score, gold-label) pairs. True->1, False->0,
    None dropped (drop_none) or scored none_score."""
    out = []
    for c in cands:
        v = c.get("regression_ok")
        if v is None:
            if drop_none:
                continue
            score = none_score
        else:
            score = 1.0 if v else 0.0
        out.append((score, 1 if c.get("gold_resolved") else 0))
    return out


def main():
    first = load(FIRST150)
    held = load(HELD_OUT)
    splits = [("first-150", first), ("held-out", held)]

    print("=" * 72)
    print("CLAIM 3 -- regression-suite gate -- offline reproduction")
    print(f"    rows loaded: first-150={len(first)}  held-out={len(held)} (no rows dropped)")
    print("=" * 72)

    # 1) governor replay anchors + floor-alone baseline
    print("\n[1] Governor replay  (current / pure_regok / hybrid_regok):")
    repl = {}
    for name, rows in splits:
        repl[name] = replay(rows)
        fa = floor_alone(rows)
        print(f"    {name:9s}  = {repl[name]}   floor-alone(baseline) = {fa}")
    ok_first = repl["first-150"] == (64, 70, 77) and floor_alone(first) == 77
    ok_held = repl["held-out"] == (76, 77, 87) and floor_alone(held) == 90
    print(f"    paper anchors: first-150 (64,70,77) base 77 ; held-out (76,77,87) base 90")
    print(f"    MATCH: first-150={ok_first}  held-out={ok_held}")

    # 2) P(resolve | regok green) -- SELECTED candidate
    print("\n[2] P(resolve | regression-green)  [SELECTED candidate]:")
    sel_green = {}
    for name, rows in splits:
        sel = [selected(r) for r in rows]
        res, n, p = p_resolve(sel, True)
        sel_green[name] = p
        print(f"    {name:9s}  = {res}/{n} = {p:.4f}")
    print(f"    paper (first-150): 0.745   MATCH: {round(sel_green['first-150'],3)==0.745}")

    # 3) P(resolve | regok red/negative) -- ALL candidates
    print("\n[3] P(resolve | gate-negative)  [ALL candidates]:")
    allred = {}
    for name, rows in splits:
        allc = [c for r in rows for c in r["candidates"].values()]
        res, n, p = p_resolve(allc, False)
        allred[name] = p
        print(f"    {name:9s}  = {res}/{n} = {p:.4f}")
    print(f"    paper: 0.11 (first) -> 0.37 (held)   "
          f"MATCH: {round(allred['first-150'],2)==0.11 and round(allred['held-out'],2)==0.37}")

    # 4) candidate single-feature AUC -- ALL candidates, drop None
    print("\n[4] Candidate single-feature AUC  [ALL candidates, drop None]:")
    cand_auc = {}
    for name, rows in splits:
        allc = [c for r in rows for c in r["candidates"].values()]
        a = auc(feature_pairs(allc, drop_none=True))
        cand_auc[name] = a
        print(f"    {name:9s}  AUC = {a:.4f}")
    print(f"    paper: 0.81 (first) -> 0.64 (held)   "
          f"MATCH: {round(cand_auc['first-150'],2)==0.81 and round(cand_auc['held-out'],2)==0.64}")

    # 5) headline single-feature AUC 0.83 -- ABANDON-subset, ALL candidates, drop None
    print("\n[5] Headline single-feature AUC 0.83  [ABANDON rows, ALL candidates, drop None]:")
    head_auc = {}
    for name, rows in splits:
        ab = [r for r in rows if r["lever_decision"] == "ABANDON"]
        allc = [c for r in ab for c in r["candidates"].values()]
        a = auc(feature_pairs(allc, drop_none=True))
        head_auc[name] = a
        print(f"    {name:9s}  n_abandon_rows = {len(ab):2d}   AUC = {a:.4f}")
    # 0.8371 -> paper states 0.83 (third decimal truncated, not round-half-up).
    head_match = abs(head_auc['first-150'] - 0.83) < 0.01  # within rounding band
    print(f"    paper (first-150, L1876 & L1884): 0.83   raw = {head_auc['first-150']:.4f}")
    print(f"    MATCH (within 0.01 band; paper truncates 0.8371->0.83 vs round-half-up 0.84): {head_match}")

    # 6) abandon-subset companion AUC '0.83 -> 0.67' -- VERIFIED CROSS-UNIVERSE
    print("\n[6] Abandon-subset AUC '0.83 -> 0.67' (paper L1884-1885) -- universe scan:")
    print(f"    {'universe':35s} {'first':>8s} {'held':>8s}")
    for cl, cfn in [("all-cand", lambda rs: [c for r in rs for c in r["candidates"].values()]),
                    ("selected", lambda rs: [selected(r) for r in rs])]:
        for nl, dn, ns in [("drop-None", True, 0.0), ("None=0", False, 0.0), ("None=1", False, 1.0)]:
            ab_f = [r for r in first if r["lever_decision"] == "ABANDON"]
            ab_h = [r for r in held if r["lever_decision"] == "ABANDON"]
            af = auc(feature_pairs(cfn(ab_f), drop_none=dn, none_score=ns))
            ah = auc(feature_pairs(cfn(ab_h), drop_none=dn, none_score=ns))
            flag = " <== BOTH ~0.83/0.67" if (abs(af - 0.83) < 0.005 and abs(ah - 0.67) < 0.005) else ""
            print(f"    abandon/{cl}/{nl:9s}            {af:8.4f} {ah:8.4f}{flag}")
    print("    paper: first 0.83, held 0.67.")
    print("    DISCREPANCY: NO single universe yields both 0.83 AND 0.67.")
    print("      0.83 = abandon/all-cand/drop-None (0.8371; held there = 0.5784 -> 0.58).")
    print("      0.67 = abandon/selected/None=0    (0.6706; first there = 0.8400 -> 0.84).")
    print("      The paired '0.83 -> 0.67' is reported across two aggregation conventions.")

    # overall verdict on the load-bearing CLAIM-3 numbers
    primary_ok = (ok_first and ok_held
                  and round(sel_green['first-150'], 3) == 0.745
                  and round(allred['first-150'], 2) == 0.11
                  and round(allred['held-out'], 2) == 0.37
                  and round(cand_auc['first-150'], 2) == 0.81
                  and round(cand_auc['held-out'], 2) == 0.64
                  and abs(head_auc['first-150'] - 0.83) < 0.01)
    print("\n" + "=" * 72)
    print(f"PRIMARY CLAIM-3 NUMBERS ALL MATCH: {primary_ok}")
    print("SECONDARY caveat figure (abandon-subset 0.67 held-out): cross-universe; see [6].")
    print("=" * 72)
    return 0 if primary_ok else 1


if __name__ == "__main__":
    sys.exit(main())
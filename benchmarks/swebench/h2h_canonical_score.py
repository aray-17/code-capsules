#!/usr/bin/env python3
"""Canonical head-to-head scorer: two-pass critique vs. the Agentless baseline.

Reproduces the paper's head-to-head result (Claim C5) offline, with no Docker,
no API, and no model calls, from committed per-instance evaluation data:

  two-pass critique   172/300 (57.3%)  @ $0.445/inst  $0.776/resolve
  Agentless oracle@k  152/300 (50.7%)  @ $0.456/inst  $0.900/resolve
  more resolved at lower cost (Pareto): +20 instances (+6.7pp),
  13.8% lower cost per resolve, McNemar exact p=0.0055.

Each system gets exactly ONE verdict per instance over the shared 300-instance
universe (no union inflation). Per-instance precedence:

  1. the canonical score of the attempt's own patch
  2. a fresh single draw, used ONLY for instances that had no scoreable patch
     (infrastructure-lost / no-patch); never combined with tier 1
  3. the original run verdict, when neither of the above applies

A parser-blindspot overlay restores instances the robust FAILED-set scorer
confirms resolved. Cost is summed per instance on the one modeled cost surface
(the run that produced the counted verdict), then divided by 300 (per instance)
and by the resolved count (per resolve).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=src python3 benchmarks/swebench/h2h_canonical_score.py
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SC = ROOT / "evals" / "scopeC"
EV = ROOT / "evals"

# Paper targets (Claim C5).
T_TP_RES, T_AG_RES, N = 172, 152, 300
T_TP_CPI, T_AG_CPI = 0.445, 0.456
T_TP_CPR, T_AG_CPR = 0.776, 0.900
T_PCT, T_PVAL = 13.8, 0.0055


def _load(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _canon_map(path: Path) -> dict[str, bool]:
    """instance_id -> bool(canonical_resolved), dropping None (unscored)."""
    m = {}
    for r in _load(path):
        v = r.get("canonical_resolved")
        if v is not None:
            m[r["instance_id"]] = bool(v)
    return m


def _orig(files: list[Path]) -> dict[str, dict]:
    """instance_id -> {resolved, cost_usd} from the original run files."""
    m = {}
    for f in files:
        for r in _load(f):
            m[r["instance_id"]] = {"resolved": bool(r.get("resolved")),
                                   "cost_usd": float(r.get("cost_usd") or 0.0)}
    return m


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def _blindspot_overlay(cell: str) -> dict[str, bool]:
    """Confirmed parser-blindspot restores for this cell: rows the robust
    FAILED-set scorer proves resolved. instance_id -> True."""
    m = {}
    for r in _load(SC / "parser_blindspot_restores.jsonl"):
        if r.get("cell") == cell and r.get("resolved") is True:
            m[r["instance_id"]] = True
    return m


def finalize(name, orig_files, h2h_path, rerun_paths, rerun_cost_paths, cell):
    orig = _orig(orig_files)
    universe = set(orig)                       # the 300 instances actually run
    h2h = _canon_map(h2h_path)
    rerun = {}
    for p in rerun_paths:
        for iid, v in _canon_map(p).items():
            rerun.setdefault(iid, v)          # first rerun source wins

    rcost = {}
    for p in rerun_cost_paths:
        for r in _load(p):
            if r.get("cost_usd") is not None:
                rcost[r["instance_id"]] = float(r["cost_usd"])

    overlay = _blindspot_overlay(cell)
    resolved, total_cost = set(), 0.0
    for iid in universe:
        if iid in overlay:                    # tier 0: confirmed parser-blindspot restore
            v = overlay[iid]
            total_cost += rcost.get(iid, orig.get(iid, {}).get("cost_usd", 0.0))
        elif iid in h2h:                      # tier 1: canonical score of the stored patch
            v = h2h[iid]
            total_cost += orig.get(iid, {}).get("cost_usd", 0.0)
        elif iid in rerun:                    # tier 2: fresh single draw
            v = rerun[iid]
            total_cost += rcost.get(iid, orig.get(iid, {}).get("cost_usd", 0.0))
        else:                                 # tier 3: original verdict
            v = orig.get(iid, {}).get("resolved", False)
            total_cost += orig.get(iid, {}).get("cost_usd", 0.0)
        if v:
            resolved.add(iid)

    n = len(universe)
    nres = len(resolved)
    cpi = total_cost / n if n else 0.0
    cpr = total_cost / nres if nres else float("nan")
    print(f"=== {name} ===")
    print(f"  universe={n}  resolved={nres}/{n} ({100*nres/n:.1f}%)")
    print(f"  cost: ${cpi:.3f}/inst   ${cpr:.3f}/resolve   (total ${total_cost:.2f})")
    return resolved, universe, cpi, cpr


def _ok(a, b, tol):
    return abs(a - b) <= tol


def main() -> int:
    print("=" * 72)
    print("Claim C5 -- canonical head-to-head: two-pass critique vs. Agentless")
    print("=" * 72)

    TP_ORIG = [EV / "leakfree" / "tb_forcestage2_first150.jsonl",
               EV / "leakfree" / "tb_forcestage2_second150.jsonl"]
    TP, U_tp, tp_cpi, tp_cpr = finalize(
        "two-pass critique", TP_ORIG,
        SC / "canonical_tp_h2h.jsonl",
        [SC / "canonical_tp_rerun.jsonl"],
        [SC / "tp_rerun_recovery.jsonl"], cell="tp")

    AG_ORIG = [SC / "h2h_agentless_n300_scored_20260616.jsonl"]
    AG, U_ag, ag_cpi, ag_cpr = finalize(
        "Agentless oracle@k", AG_ORIG,
        SC / "canonical_agentless.jsonl",
        [SC / "canonical_agentless_16rerun.jsonl"],
        [SC / "agentless_sympy_rerun.jsonl"], cell="agentless")

    b = len(TP - AG); c = len(AG - TP)
    p = mcnemar(b, c)
    pct = (1 - tp_cpr / ag_cpr) * 100 if ag_cpr else 0.0
    pareto = (len(TP) > len(AG)) and (tp_cpi < ag_cpi)

    print("\n=== head-to-head (McNemar exact, paired) ===")
    print(f"  two-pass critique {len(TP)}/300 vs Agentless oracle@k {len(AG)}/300   "
          f"margin {len(TP)-len(AG):+d}  (only-two-pass={b}, only-Agentless={c})")
    print(f"  McNemar exact p={p:.4f}")
    print(f"  cost per resolve: two-pass ${tp_cpr:.3f} vs Agentless ${ag_cpr:.3f}  "
          f"=> {pct:.1f}% lower cost per resolve")
    print(f"  cost per instance: two-pass ${tp_cpi:.3f} vs Agentless ${ag_cpi:.3f}")
    print(f"  Pareto (more resolved AND cheaper per instance): {'PARETO' if pareto else 'NOT PARETO'}")

    checks = {
        f"two-pass critique {len(TP)}/300": len(TP) == T_TP_RES,
        f"Agentless oracle@k {len(AG)}/300": len(AG) == T_AG_RES,
        "two-pass $/inst": _ok(round(tp_cpi, 3), T_TP_CPI, 0.0006),
        "Agentless $/inst": _ok(round(ag_cpi, 3), T_AG_CPI, 0.0006),
        "two-pass $/resolve": _ok(round(tp_cpr, 3), T_TP_CPR, 0.0006),
        "Agentless $/resolve": _ok(round(ag_cpr, 3), T_AG_CPR, 0.0006),
        "13.8% lower cost per resolve": _ok(round(pct, 1), T_PCT, 0.05),
        "McNemar exact p=0.0055": _ok(round(p, 4), T_PVAL, 0.0001),
        "Pareto": pareto,
    }
    print("\n=== reproduction check ===")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'MISMATCH'}] {k}")
    all_ok = all(checks.values())
    print("-" * 72)
    print("RESULT:", "ALL TARGET NUMBERS REPRODUCE" if all_ok
          else "ONE OR MORE TARGETS DO NOT REPRODUCE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

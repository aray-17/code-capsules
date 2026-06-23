#!/usr/bin/env python3
"""Standalone offline scorer for CLAIM 3 -- the regression-suite gate.

Reproduces, from committed per-instance JSONLs (no Docker / no API / no model
calls), every number the paper states for the regression-suite gate
(Section "A regression-suite gate restores the wrong abandons", paper sec
9.6) and the verifier-driven single-tier architecture it repairs (sec 9.2).

Data sources
------------
Lever DECISIONS (gold-INDEPENDENT, unchanged): the leak-free lever evaluation.
  first-150 : evals/leakfree/exp4_lever_floor100_siginject.jsonl
  held-out  : evals/leakfree/exp4_lever_second150.jsonl
These supply lever_decision (SHIP / NO_REPRO_FALLBACK / ABANDON), the selected
config, and the per-candidate regression_ok flag -- none of which depend on
gold labels.

Gold RESOLVE labels (CANONICAL): joined read-only by instance_id. The lever
files' own per-candidate gold fields are stale (a superseded Docker scoring);
the canonical resolved labels live in the per-configuration result JSONLs.
  first-150 floor    : evals/p10_sonnet_floor_n150_20260516T040248.jsonl       (resolved)
  first-150 siginject: evals/p10_sonnet_siginject_b10x3_n150_20260517T035438.jsonl (resolved)
  held-out  floor    : evals/canonical_p10_sonnet_strat_holdout_floor_20260520T075105.jsonl       (canonical_resolved)
  held-out  siginject: evals/canonical_p10_sonnet_strat_holdout_siginject_b10x3_20260517T173631.jsonl (canonical_resolved)

Canonical targets (paper sec 9.2 + 9.6)
---------------------------------------
Lever decisions, first-150 (gold-independent): 100 SHIP, 12 NO_REPRO_FALLBACK, 38 ABANDON.

  first-150 : honest controller = 84  ;  regression gate = 99  ;  floor-alone unbounded baseline = 111
  held-out  : honest controller = 76  ;  regression gate = 89  ;  floor-alone unbounded baseline = 93

  P(resolve | regression-green) = 0.792  (over ALL candidates, = 126/159).
  SHIP precision = 75.0% (84/112 SHIP+fallback) / 77.0% (77/100 SHIP-only); 28 false-positive ships.
  Single-tier abandon destroys 28 wrong-abandons; the gate RECOVERS 15 of 28 (84 -> 99),
  staying 12 short of the unbounded 111.

Replay semantics
----------------
  honest (current) = canonical_resolved(selected) AND lever_decision != ABANDON
  gate (hybrid)    = canonical_resolved(selected) AND
                     (lever_decision != ABANDON OR selected.regression_ok is True)
  floor-alone      = canonical_resolved(floor candidate), always shipped (unbounded baseline).
A None regression_ok never rescues. Gold labels are used for scoring only,
never on the deployable decision path.

Run with: PYTHONNOUSERSITE=1 PYTHONPATH=.:src python3.12 benchmarks/swebench/regression_gate.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals"

# gold-independent lever decisions
LEVER = {
    "first-150": EVALS / "leakfree/exp4_lever_floor100_siginject.jsonl",
    "held-out": EVALS / "leakfree/exp4_lever_second150.jsonl",
}

# canonical gold resolve labels, per split per configuration: (path, field)
CANON = {
    "first-150": {
        "floor": (EVALS / "p10_sonnet_floor_n150_20260516T040248.jsonl", "resolved"),
        "siginject": (EVALS / "p10_sonnet_siginject_b10x3_n150_20260517T035438.jsonl", "resolved"),
    },
    "held-out": {
        "floor": (EVALS / "canonical_p10_sonnet_strat_holdout_floor_20260520T075105.jsonl", "canonical_resolved"),
        "siginject": (EVALS / "canonical_p10_sonnet_strat_holdout_siginject_b10x3_20260517T173631.jsonl", "canonical_resolved"),
    },
}


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def canon_labels(split):
    """{config: {instance_id: bool resolved}} for a split, from canonical files."""
    out = {}
    for cfg, (path, field) in CANON[split].items():
        out[cfg] = {r["instance_id"]: bool(r.get(field)) for r in load(path)}
    return out


def gold(labels, cfg, iid):
    """Canonical resolved for (config, instance); absent -> not resolved."""
    return labels.get(cfg, {}).get(iid, False)


def selected(row):
    return row["candidates"][row["selected_config"]]


# -- governor replay against CANONICAL gold -----------------------------------
def replay(rows, labels):
    sc = lambda r: r["selected_config"]
    honest = sum(1 for r in rows
                 if gold(labels, sc(r), r["instance_id"]) and r["lever_decision"] != "ABANDON")
    gate = sum(1 for r in rows
               if gold(labels, sc(r), r["instance_id"])
               and (r["lever_decision"] != "ABANDON"
                    or selected(r).get("regression_ok") is True))
    return honest, gate


def floor_alone(rows, labels):
    """Unbounded baseline = the floor candidate's canonical gold, always shipped."""
    return sum(1 for r in rows if gold(labels, "floor", r["instance_id"]))


def p_resolve_green_all(rows, labels):
    """P(canonical resolve | regression_ok is True) over ALL candidates."""
    res = tot = 0
    for r in rows:
        for cfg, c in r["candidates"].items():
            if c.get("regression_ok") is True:
                tot += 1
                if gold(labels, cfg, r["instance_id"]):
                    res += 1
    return res, tot


def ship_precision(rows, labels):
    """(ship-only res, ship-only n), (ship+fallback res, ship+fallback n)."""
    sc = lambda r: r["selected_config"]
    g = lambda r: gold(labels, sc(r), r["instance_id"])
    ship = [r for r in rows if r["lever_decision"] == "SHIP"]
    fb = [r for r in rows if r["lever_decision"] == "NO_REPRO_FALLBACK"]
    sf = ship + fb
    return (sum(1 for r in ship if g(r)), len(ship)), (sum(1 for r in sf if g(r)), len(sf))


def wrong_abandons(rows, labels):
    """Abandoned instances that carried at least one canonically-resolving candidate."""
    return sum(1 for r in rows
               if r["lever_decision"] == "ABANDON"
               and (gold(labels, "floor", r["instance_id"])
                    or gold(labels, "siginject", r["instance_id"])))


def main():
    print("=" * 72)
    print("CLAIM 3 -- regression-suite gate -- offline reproduction (canonical gold)")
    print("=" * 72)

    targets = {  # (honest, gate, floor_alone) per split, from paper sec 9.6
        "first-150": (84, 99, 111),
        "held-out": (76, 89, 93),
    }

    all_ok = True
    splits = {}
    for split in ("first-150", "held-out"):
        rows = load(LEVER[split])
        labels = canon_labels(split)
        splits[split] = (rows, labels)
        print(f"    rows loaded: {split}={len(rows)}")

    # [1] decisions, first-150 (gold-independent) ----------------------------
    from collections import Counter
    dec = Counter(r["lever_decision"] for r in splits["first-150"][0])
    dec_ok = dec.get("SHIP") == 100 and dec.get("NO_REPRO_FALLBACK") == 12 and dec.get("ABANDON") == 38
    all_ok &= dec_ok
    print("\n[1] Lever decisions first-150 (gold-independent):")
    print(f"    SHIP={dec.get('SHIP')}  NO_REPRO_FALLBACK={dec.get('NO_REPRO_FALLBACK')}  ABANDON={dec.get('ABANDON')}")
    print(f"    paper: 100 / 12 / 38     MATCH: {dec_ok}")

    # [2] honest controller / regression gate / floor-alone baseline ---------
    print("\n[2] Honest controller / regression gate / floor-alone baseline:")
    for split in ("first-150", "held-out"):
        rows, labels = splits[split]
        honest, gate = replay(rows, labels)
        fa = floor_alone(rows, labels)
        t = targets[split]
        ok = (honest, gate, fa) == t
        all_ok &= ok
        print(f"    {split:9s}  honest={honest}  gate={gate}  floor-alone={fa}   "
              f"paper={t}   MATCH: {ok}")
        if split == "first-150":
            print(f"               gate recovers {gate - honest} of "
                  f"{wrong_abandons(rows, labels)} wrong-abandons; "
                  f"{fa - gate} short of unbounded {fa}.")

    # [3] P(resolve | regression-green) over ALL candidates, first-150 -------
    rows, labels = splits["first-150"]
    res, tot = p_resolve_green_all(rows, labels)
    p = res / tot
    p_ok = round(p, 3) == 0.792 and (res, tot) == (126, 159)
    all_ok &= p_ok
    print("\n[3] P(resolve | regression-green)  [ALL candidates, first-150]:")
    print(f"    {res}/{tot} = {p:.4f}     paper: 0.792 (126/159)   MATCH: {p_ok}")

    # [4] SHIP precision + false-positive ships, first-150 -------------------
    rows, labels = splits["first-150"]
    (so_r, so_n), (sf_r, sf_n) = ship_precision(rows, labels)
    fp = sf_n - sf_r
    prec_ok = (so_r, so_n) == (77, 100) and (sf_r, sf_n) == (84, 112) and fp == 28
    all_ok &= prec_ok
    print("\n[4] SHIP precision + false-positive ships [first-150]:")
    print(f"    SHIP-only       = {so_r}/{so_n} = {so_r/so_n:.3f}  ({so_r/so_n*100:.1f}%)")
    print(f"    SHIP+fallback   = {sf_r}/{sf_n} = {sf_r/sf_n:.3f}  ({sf_r/sf_n*100:.1f}%)")
    print(f"    false-positive ships = {fp}")
    print(f"    paper: 77.0% (77/100) / 75.0% (84/112), 28 FP ships   MATCH: {prec_ok}")

    print("\n" + "=" * 72)
    print(f"ALL CLAIM-3 NUMBERS REPRODUCE PAPER: {all_ok}")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

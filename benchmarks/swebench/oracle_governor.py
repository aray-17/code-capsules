#!/usr/bin/env python3
"""
Scorer for CLAIM 2 -- the oracle-relative governor (run-both diverse-select lever).

Reproduces, OFFLINE from committed eval data (no Docker / API / model calls), the
paper's numbers in Section 8 (gold-graded oracle replay) and Section 8/13 (deployable
integration test):

  (A) The run-both diverse-select lever resolves 85/150 gold-union instances at a
      modeled total cost of $118.75 (first-150).                 paper.tex L1641-1642, L1714, L2428

  (B) That is WITHIN ONE INSTANCE of the omniscient oracle. In the gold-graded
      replay over the first-150 ten-turn config sweeps:
        - the oracle UNION across the shipped configurations reaches 86/150;
          64/150 are doomed (no config resolves).               paper.tex L1608, L1610-1611
        - floor + signal-injection (the lever pair) resolve 85/150;
          65 instances are both-fail (the lever abandons all 65). paper.tex L1641, L1708, L1714
        - escalating every both-fail instance to a THIRD (plan-then-execute)
          configuration recovers NONE of them, at +$19.58 -> $138.33 for 85/150.
                                                                  paper.tex L1645, L1709-1711
        - the single both-fail instance the oracle recovers is resolved by ANOTHER
          shipped configuration (not plan-then-execute), at $0.70, reaching 86/150
          for $119.45.                                           paper.tex L1711-1713
        => lever 85 vs oracle 86 = within ONE instance and $0.70 of the omniscient
           oracle.                                               paper.tex L1714-1715

  (C) Deployment-honest SHIP precision (shipped & gold-resolved / shipped-total):
      58% on the first 150, 67% held-out.                        paper.tex L1694, L1742-1744, L2647-2648

Ship definition (paper L1742-1744): a "ship" is a SHIP lever decision. The lever
ships 100 of 150 first-half and 105 of 150 held-out, with 42 and 35 false-positive
ships (shipped but not gold-resolved); ship precision is 58% and 67%.

-----------------------------------------------------------------------------------
IMPORTANT METHODOLOGY NOTE (the canonical derivation of part B):

The paper's 86/150 oracle ceiling, the 64 doomed, the 65 both-fail, and the claim
that the third (plan-then-execute) configuration "recovers none" are all the SINGLE
gold-graded replay over the first-150 ten-turn configuration sweeps
(evals/p10_sonnet_*n150*.jsonl), exactly as the project's canonical tool
benchmarks/swebench/agreement_signal.py computes them. We replicate that tool's config-loading
here (one file per config name; same-strategy variance re-rolls "*_var2/var3*"
excluded, since a re-roll resolving an instance is sampling noise, not a recoverable
strategy). This is the data the paper is built on.

In this canonical replay the single both-fail instance the omniscient oracle
recovers is matplotlib-25311, resolved by the implicit-budget-40 configuration at
$0.697 ~ $0.70 -- NOT plan-then-execute, which recovers zero of the 65 both-fail.
This is consistent with paper L1711-1713 ("another shipped configuration ... at
$0.70, reaching 86/150 for $119.45") and L1709-1711 / L1645 (plan-then-execute
recovers none).

Data sources (all committed):
  evals/leakfree/exp4_lever_floor100_siginject.jsonl  -- run-both lever, first-150,
      deployable integration test (per-row 'candidates' dict floor+siginject with
      gold_resolved/regression_ok/cost_usd; top-level lever_decision/selected_config/
      shipped_gold_resolved/oracle_union_gold/lever_cost_usd). Used for (A) and (C).
  evals/leakfree/exp4_lever_second150.jsonl           -- run-both lever, held-out. (C).
  evals/p10_sonnet_*n150*.jsonl                        -- the first-150 ten-turn config
      sweeps (floor, siginject_b10x3, planFirst_b20, implicit40, ...). Used for (B),
      the gold-graded oracle replay, identically to benchmarks/swebench/agreement_signal.py.

Run:
  PYTHONNOUSERSITE=1 PYTHONPATH=src python3 oracle_governor.py
"""
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEVER_FIRST = REPO / "evals/leakfree/exp4_lever_floor100_siginject.jsonl"
LEVER_HELDOUT = REPO / "evals/leakfree/exp4_lever_second150.jsonl"
SWEEP_GLOB = str(REPO / "evals/p10_sonnet_*n150*.jsonl")

# The lever's two diverse configurations, by the config-name token in the sweep files.
LEVER_PAIR = ("floor", "siginject_b10x3")
THIRD_CONFIG = "planFirst_b20"   # plan-then-execute escalation tier (recovers none)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_sweep_configs(pattern):
    """Replicate agreement_signal.load_configs: one file per config-name token,
    excluding same-strategy variance re-rolls (*_var2/var3*). This is the canonical
    first-150 gold-graded config pool the paper's Section 8 oracle replay uses."""
    configs = {}
    for f in sorted(glob.glob(pattern)):
        if re.search(r"_var\d+_n\d+", f):
            continue
        m = re.search(r"p10_[a-z]+_(.+?)_n\d+", f)
        name = m.group(1) if m else f
        d = {}
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[r["instance_id"]] = dict(
                resolved=(r.get("resolved") is True),
                cost=float(r.get("cost_usd") or 0.0),
            )
        if d:
            configs[name] = d
    return configs


def ship_precision(rows):
    """Deployment-honest SHIP precision: of the SHIP decisions, the fraction whose
    shipped candidate gold-resolves."""
    ships = [r for r in rows if r["lever_decision"] == "SHIP"]
    good = sum(1 for r in ships if r.get("shipped_gold_resolved"))
    n = len(ships)
    return n, good, n - good, (good / n if n else 0.0)


def main():
    ok = True

    def check(label, got, want, paper):
        nonlocal ok
        match = got == want
        ok = ok and match
        print(f"  [{'OK ' if match else 'XX '}] {label}: {got}  (paper: {want}, {paper})")

    first = load_jsonl(LEVER_FIRST)
    heldout = load_jsonl(LEVER_HELDOUT)

    # ----------------------------------------------------------------------------
    # (A) deployable lever cost = $118.75 for 85 gold-union resolves (first-150).
    # ----------------------------------------------------------------------------
    lever_cost = sum(r["lever_cost_usd"] for r in first)
    union_resolved = sum(1 for r in first if r.get("oracle_union_gold"))

    # ----------------------------------------------------------------------------
    # (B) gold-graded oracle replay over the first-150 ten-turn config sweeps,
    #     computed the canonical way (benchmarks/swebench/agreement_signal.py).
    # ----------------------------------------------------------------------------
    cfg = load_sweep_configs(SWEEP_GLOB)
    missing = [c for c in LEVER_PAIR + (THIRD_CONFIG,) if c not in cfg]
    if missing:
        print(f"FATAL: expected config(s) {missing} not found in sweep glob", file=sys.stderr)
        return 2
    ids = sorted(set.intersection(*[set(d) for d in cfg.values()]))

    def resolves(c, i):
        return cfg[c][i]["resolved"]

    # oracle union across ALL shipped configs (the 86 ceiling); doomed = 64.
    oracle_union = sum(1 for i in ids if any(resolves(c, i) for c in cfg))
    doomed = sum(1 for i in ids if not any(resolves(c, i) for c in cfg))

    # lever pair (floor + siginject): resolves 85; both-fail = 65 (all abandoned).
    lever_pair_resolved = sum(
        1 for i in ids if any(resolves(c, i) for c in LEVER_PAIR)
    )
    both_fail = [i for i in ids if not any(resolves(c, i) for c in LEVER_PAIR)]

    # escalating every both-fail to the third (plan-then-execute) config: recovers none.
    third_recovers = [i for i in both_fail if resolves(THIRD_CONFIG, i)]
    third_esc_cost = sum(cfg[THIRD_CONFIG][i]["cost"] for i in both_fail)

    # the single both-fail instance the oracle still recovers (the +1 -> 86), and which
    # config resolves it (the paper's "another shipped configuration", not the third).
    oracle_recoverable_bf = [
        i for i in both_fail if any(resolves(c, i) for c in cfg)
    ]
    lever_to_oracle_gap = oracle_union - lever_pair_resolved

    # 2-config select and 3-config escalate modeled cost (cross-check vs paper).
    def policy_cost(members):
        return sum(cfg[c][i]["cost"] for i in ids for c in members)
    select_cost = policy_cost(LEVER_PAIR)
    three_cost = policy_cost(LEVER_PAIR + (THIRD_CONFIG,))

    # ----------------------------------------------------------------------------
    # (C) deployment-honest SHIP precision: 58% first-150, 67% held-out.
    # ----------------------------------------------------------------------------
    n1, good1, fp1, prec1 = ship_precision(first)
    n2, good2, fp2, prec2 = ship_precision(heldout)

    print("CLAIM 2 -- oracle-relative governor (run-both diverse-select lever)")
    print()
    print("(A) Run-both diverse-select lever, first-150 deployable replay:")
    check("gold-union resolved", union_resolved, 85, "L1641/L1714/L2428 ($118.75/85)")
    check("lever total cost", f"${lever_cost:.2f}", "$118.75", "L1641-1642/L1714/L2428")
    print(f"       (unrounded lever cost = ${lever_cost:.4f})")
    print()
    print("(B) Distance from the omniscient oracle (first-150 gold-graded sweep replay):")
    print(f"       configs in pool: {sorted(cfg)} (n={len(ids)})")
    check("oracle union ceiling", oracle_union, 86, "L1610-1611 (86/150)")
    check("doomed (no config resolves)", doomed, 64, "L1608 (64/150)")
    check("lever pair (floor+siginject) resolved", lever_pair_resolved, 85, "L1641 (85/150)")
    check("two-config select modeled cost", f"${select_cost:.2f}", "$118.75", "L1642")
    check("both-fail instances (all abandoned by agreement)", len(both_fail), 65, "L1708/L1714")
    check("3rd config (plan-then-execute) recovers", len(third_recovers), 0, "L1645/L1709-1711 (none)")
    check("3rd-config escalation cost (+ on both-fail)", f"${third_esc_cost:.2f}", "$19.58", "L1709 (-> $138.33)")
    check("3-config escalate modeled cost", f"${three_cost:.2f}", "$161.86", "L1645")
    check("oracle-recoverable both-fail (the +1 -> 86)", len(oracle_recoverable_bf), 1, "L1711-1712")
    check("lever vs oracle gap (instances)", lever_to_oracle_gap, 1, "L1714 ('within one instance')")
    if oracle_recoverable_bf:
        iid = oracle_recoverable_bf[0]
        who = [c for c in cfg if resolves(c, iid)]
        cost = min(cfg[c][iid]["cost"] for c in who)
        print(f"       (the single oracle-recovered both-fail instance: {iid})")
        print(f"       (recovered by 'another shipped configuration' {who}; cheapest "
              f"marginal cost = ${cost:.4f} ~ $0.70 -> $119.45 total; paper L1712-1713)")
        print(f"       (NOTE: it is NOT recovered by the third plan-then-execute config, "
              f"matching paper's 'recovers none')")
    print()
    print("(C) Deployment-honest SHIP precision (shipped & gold-resolved / shipped):")
    check("first-150 ships", n1, 100, "L1742")
    check("first-150 false-positive ships", fp1, 42, "L1743")
    check("first-150 ship precision", f"{prec1*100:.0f}%", "58%", "L1694/L1744/L2647")
    print(f"       (first-150: {good1}/{n1} = {prec1:.3f})")
    check("held-out ships", n2, 105, "L1743")
    check("held-out false-positive ships", fp2, 35, "L1744")
    check("held-out ship precision", f"{prec2*100:.0f}%", "67%", "L1694/L1744/L2648")
    print(f"       (held-out: {good2}/{n2} = {prec2:.3f})")
    print()
    print("RESULT:", "ALL TARGET NUMBERS REPRODUCE" if ok else "DISCREPANCY -- see [XX] rows")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
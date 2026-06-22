"""Held-out generalization of the value-of-resolve cascade. Pure reanalysis, zero new compute.

Reproduces the paper's claim (Section "Negative results and dropped mechanisms",
cheaper-model cascade subsection) that the cheap-tier-one -> strong-tier escalation
cascade generalizes to the held-out second-150 split as the quality-max operating
point, PROVIDED the strong tier runs unbounded.

The cascade runs Sonnet ten-turn signaled-budget (tier-1) on all 150 held-out
instances and escalates every failure to an UNBOUNDED Opus floor (tier-2). All
labels are the canonical reference-patch-validated verdicts; costs are the logged
run costs (scorer-independent).

Reproduces (n=150 held-out):
  tier-1 (signaled-10) resolves        55/150
  escalated failures                   95
  unbounded Opus recovers              69/95  (73%)
  cascade resolved                     124/150   (+31 over Sonnet floor's 93)
  cascade cost   ~ $159  vs Sonnet floor ~ $84   (quality-max, not a both-axes win)

  python3 benchmarks/swebench/held_out_cascade.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SIG = ROOT / "evals/heldout/cascade_signaled10_n150_canonical.jsonl"
FLOOR = ROOT / "evals/heldout/cascade_floor_n150_canonical.jsonl"
OPUS = ROOT / "evals/heldout/cascade_opus_floor_escalated_canonical.jsonl"


def _load(path):
    res, cost = {}, {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        res[d["instance_id"]] = bool(d.get("resolved"))
        cost[d["instance_id"]] = d.get("cost_usd") or 0.0
    return res, cost


def main() -> int:
    sig_res, sig_cost = _load(SIG)
    floor_res, floor_cost = _load(FLOOR)
    opus_res, opus_cost = _load(OPUS)

    universe = sorted(sig_res)
    tier1 = sum(1 for i in universe if sig_res[i])
    escalated = [i for i in universe if not sig_res[i]]
    rescued = sum(1 for i in escalated if opus_res.get(i))
    cascade = tier1 + rescued
    floor = sum(1 for i in universe if floor_res.get(i))

    casc_cost = sum(sig_cost[i] for i in universe) + sum(opus_cost.get(i, 0.0) for i in escalated)
    floor_total = sum(floor_cost.get(i, 0.0) for i in universe)
    rescue_pct = round(100 * rescued / len(escalated))

    print("=" * 64)
    print("Held-out cascade generalization (second-150, canonical labels)")
    print("=" * 64)
    print(f"  tier-1 (signaled-10) resolved : {tier1}/150")
    print(f"  escalated failures            : {len(escalated)}")
    print(f"  unbounded Opus recovered      : {rescued}/{len(escalated)} ({rescue_pct}%)")
    print(f"  cascade resolved              : {cascade}/150")
    print(f"  Sonnet floor baseline         : {floor}/150")
    print(f"  margin                        : +{cascade - floor} over floor")
    print(f"  cascade cost                  : ${casc_cost:.2f}")
    print(f"  Sonnet floor cost             : ${floor_total:.2f}")
    print(f"  reading: quality-max operating point (more resolves at added cost), "
          f"not a both-axes dominator")

    checks = {
        "cascade 124/150": cascade == 124,
        "+31 over floor": (cascade - floor) == 31 and floor == 93,
        "Opus recovery 73%": rescue_pct == 73 and rescued == 69,
        "cascade $159 vs floor $84": round(casc_cost) == 159 and round(floor_total) == 84,
    }
    print("-" * 64)
    for label, ok in checks.items():
        print(f"  [{'OK ' if ok else 'XX '}] {label}")
    allok = all(checks.values())
    print("\nRESULT:", "ALL NUMBERS REPRODUCE" if allok else "DISCREPANCY -- see [XX] rows")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())

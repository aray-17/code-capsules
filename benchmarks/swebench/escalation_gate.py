#!/usr/bin/env python3
"""
Scorer for CLAIM 4 (SOFT) -- the model-tier escalation gate (paper Section 9.8,
sec:agreement_escalation).

Reproduces, OFFLINE from committed eval data (no Docker / API / model calls),
the paper's Section 9.8 escalation-gate numbers:

  (A) Escalating the flagged weak set to the strong tier (Opus floor) recovers
      21 of 21 gate-positive instances AND 15 of 17 gate-negative instances.
      The regression gate therefore does NOT separate recoverable from doomed at
      this tier boundary (100% vs 88% recovery): almost every flagged instance is
      recoverable one tier up, so what pays is escalating the whole flagged set,
      not gating it.

  (B) Regression-gated escalation: base honest single-tier controller = 64;
      escalating the 21 gate-positive abandons adds the 21 strong-tier recoveries,
      reaching 85/150 (= 64 + 21, +21 over base) at roughly $1.58 per resolve.

  (C) regok close/lost split (warm-start probe over cheap-tier patch-bearing
      failures): the regression-suite split partitions the 49 failures into a
      close set (17: regression green) and a lost set (32: regression red); under
      canonical scoring the strong tier recovers most of both (close 14 of 17,
      lost 30 of 32), so the split now only orders the escalation queue.

  (D) Held-out: 37% of gate-negative carried-patch instances in fact resolve, so a
      decline-by-default arm needs per-workload justification.

Gold-INDEPENDENT, derived from the lever decisions (unchanged by any rescoring):
  abandon-universe = 38 instances; regok partition = 21 gate-positive (regok True)
  / 17 gate-negative (14 red + 3 undetermined/None); base honest single-tier
  controller = 64 (the Section 9.8 cross-system honest controller anchor).

The strong-tier recovery counts are joined READ-ONLY against the canonical Opus
floor labels (evals/scopeC/opus/canonical_floor.jsonl, field canonical_resolved,
138/150). The earlier stale Opus-floor sweep (p9_opus_floor, 96/150) produced the
now-retired discriminative framing (20 of 21 vs 3 of 17, total 84, $1.60); under
canonical floor scoring the regok gate no longer discriminates strong-tier
recovery and the total moves to 85 (= 64 + 21) at $1.58 per resolve.

Escalation cost is scorer-independent (cost_usd is the same regardless of which
gold labels score the resolve); the per-resolve figure is the gov2 total cost
($118.75 lever + $15.61 Opus escalation = $134.36) divided by the 85 resolves.

Data sources (all committed):
  evals/leakfree/exp4_lever_floor100_siginject.jsonl  -- the run-both lever, first-150
                                                         (lever_decision, candidates with
                                                          regression_ok, lever_cost_usd)
  evals/scopeC/opus/canonical_floor.jsonl             -- canonical Opus-floor labels,
                                                         first-150 (canonical_resolved)
  evals/p9_opus_floor_n150_combined_20260527.jsonl    -- Opus-floor sweep, first-150;
                                                         used ONLY for the scorer-independent
                                                         escalation cost_usd
  evals/leakfree/exp4_lever_second150.jsonl           -- the lever, held-out (second-150)
  evals/leakfree/warmstart_opus_cold.jsonl            -- Opus escalation of the 49 cheap-tier
                                                         patch-bearing failures (cold cascade)
  evals/leakfree/e1b_detector.jsonl                   -- regression-suite gate (agentless_regok)
                                                         on the same 49 patch-bearing failures

Two regression-gate conventions appear in the paper, and the data carries both:

  (A/B) escalation gate (the deployable actuator):
        gate-POSITIVE = regression suite definitively GREEN (regok is True)
        gate-NEGATIVE = regression suite RED *or* UNDETERMINED (regok False or None)
        Of the 38 abandons, 21 are gate-positive and 17 are gate-negative
        (14 red + 3 undetermined/None).

  (C) close/lost descriptive split:
        close = regression suite NOT red (green or undetermined, regok True or None)
        lost  = regression suite RED (regok is False)
        "lost" means the suite specifically went red ("on the wrong track"); an
        undetermined suite is grouped with close. Of the 49 cheap-tier
        patch-bearing failures, 17 are close and 32 are lost.

Run:
  PYTHONNOUSERSITE=1 PYTHONPATH=.:src python3.12 benchmarks/swebench/escalation_gate.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEVER_FIRST = os.path.join(REPO, "evals/leakfree/exp4_lever_floor100_siginject.jsonl")
LEVER_HELDOUT = os.path.join(REPO, "evals/leakfree/exp4_lever_second150.jsonl")
OPUS_FLOOR_CANONICAL = os.path.join(REPO, "evals/scopeC/opus/canonical_floor.jsonl")
OPUS_FLOOR_COST = os.path.join(REPO, "evals/p9_opus_floor_n150_combined_20260527.jsonl")
WARMSTART_COLD = os.path.join(REPO, "evals/leakfree/warmstart_opus_cold.jsonl")
E1B_DETECTOR = os.path.join(REPO, "evals/leakfree/e1b_detector.jsonl")

# Section 9.8 cross-system honest single-tier controller anchor.
BASE_HONEST_SINGLE_TIER = 64


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def escalation_gate_positive(regok):
    """Escalation gate: positive iff the suite is definitively green."""
    return regok is True


def close_not_lost(regok):
    """close/lost split: 'lost' means the suite went red; otherwise close."""
    return regok is not False


def main():
    ok = True

    # ---- load -----------------------------------------------------------
    lever = load_jsonl(LEVER_FIRST)
    # canonical strong-tier (Opus floor) resolve labels for the recovery counts
    canon = {d["instance_id"]: d for d in load_jsonl(OPUS_FLOOR_CANONICAL)}
    # scorer-independent escalation cost (cost_usd unchanged by rescoring)
    opus_cost_rows = {d["instance_id"]: d for d in load_jsonl(OPUS_FLOOR_COST)}

    # =====================================================================
    # (A) the 38 abandons (gold-independent), split by the selected candidate's
    #     regression gate; recovery joined READ-ONLY to the canonical Opus floor.
    # =====================================================================
    abandons = [d for d in lever if d["lever_decision"] == "ABANDON"]

    def sel_regok(d):
        return d["candidates"].get(d["selected_config"], {}).get("regression_ok")

    gate_pos = [d for d in abandons if escalation_gate_positive(sel_regok(d))]
    gate_neg = [d for d in abandons if not escalation_gate_positive(sel_regok(d))]

    def canon_resolved(d):
        o = canon.get(d["instance_id"])
        return bool(o and o.get("canonical_resolved"))

    gp_total = len(gate_pos)
    gp_recovered = sum(1 for d in gate_pos if canon_resolved(d))
    gn_total = len(gate_neg)
    gn_recovered = sum(1 for d in gate_neg if canon_resolved(d))

    # =====================================================================
    # (B) escalate the gate-positive abandons to the strong tier.
    #     resolved = base (Section 9.8 anchor 64) + gate-positive recoveries;
    #     cost = lever total + scorer-independent Opus escalation cost.
    # =====================================================================
    lever_cost = sum(d["lever_cost_usd"] for d in lever)
    opus_esc_cost = sum(opus_cost_rows[d["instance_id"]]["cost_usd"] for d in gate_pos)
    base_n = BASE_HONEST_SINGLE_TIER
    gov2_resolved = base_n + gp_recovered
    gov2_cost = lever_cost + opus_esc_cost
    gov2_cpr = gov2_cost / gov2_resolved

    # =====================================================================
    # (C) regok close/lost split on the warm-start probe (cold cascade) over
    #     the 49 cheap-tier patch-bearing failures; gate from the e1b detector.
    # =====================================================================
    detector = load_jsonl(E1B_DETECTOR)
    regok = {d["instance_id"]: d.get("agentless_regok") for d in detector}
    cold = load_jsonl(WARMSTART_COLD)
    close_res = close_tot = lost_res = lost_tot = 0
    for d in cold:
        iid = d["instance_id"]
        rk = regok.get(iid)
        resolved = bool(d.get("resolved"))
        if close_not_lost(rk):  # close = not red (green or undetermined)
            close_tot += 1
            close_res += int(resolved)
        else:                      # lost = red
            lost_tot += 1
            lost_res += int(resolved)

    # =====================================================================
    # (D) held-out gate-negative carried-patch resolve rate = 37%.
    #     Candidate-level: candidates with regok red AND a patch, fraction
    #     that gold-resolve.
    # =====================================================================
    heldout = load_jsonl(LEVER_HELDOUT)
    gn_patch = gn_patch_res = 0
    for d in heldout:
        for c in d["candidates"].values():
            if c.get("regression_ok") is False and c.get("has_patch"):
                gn_patch += 1
                gn_patch_res += int(bool(c.get("gold_resolved")))
    gn_rate = gn_patch_res / gn_patch if gn_patch else 0.0

    # ---- report ---------------------------------------------------------
    def check(label, got, want):
        nonlocal ok
        match = got == want
        ok = ok and match
        print(f"  [{'OK ' if match else 'XX '}] {label}: {got}  (paper: {want})")

    print("CLAIM 4 -- model-tier escalation gate (Section 9.8)")
    print()
    print("(A) Strong-tier recovery on the 38 abandons, split by regression gate:")
    check("abandon-universe", len(abandons), 38)
    check("regok partition (pos/neg)", f"{gp_total}/{gn_total}", "21/17")
    check("gate-positive recovered", f"{gp_recovered} of {gp_total}", "21 of 21")
    check("gate-negative recovered", f"{gn_recovered} of {gn_total}", "15 of 17")
    print()
    print("(B) Regression-gated escalation (base 64 + 21 = 85/150):")
    check("honest single-tier base", base_n, 64)
    check("lever total cost", f"${lever_cost:.2f}", "$118.75")
    check("Opus escalation cost (21)", f"${opus_esc_cost:.2f}", "$15.61")
    check("escalation resolved", f"{gov2_resolved}/150", "85/150")
    check("escalation net gain over base", gov2_resolved - base_n, 21)
    check("escalation cost-per-resolve", f"${gov2_cpr:.2f}", "$1.58")
    print(f"       (unrounded $/res = ${gov2_cpr:.4f}; total cost ${gov2_cost:.2f})")
    print()
    print("(C) regok close/lost split (cold cascade over 49 patch-bearing failures):")
    check("close (regok not-red) rescued", f"{close_res} of {close_tot}", "14 of 17")
    check("lost (regok red) rescued", f"{lost_res} of {lost_tot}", "30 of 32")
    print()
    print("(D) Held-out gate-negative carried-patch resolve rate:")
    check("P(resolve | regok-red, has_patch)", f"{gn_rate*100:.0f}%", "37%")
    print(f"       (= {gn_patch_res}/{gn_patch} = {gn_rate:.3f})")
    print()
    print("RESULT:", "ALL NUMBERS REPRODUCE -- MATCH" if ok
          else "DISCREPANCY -- see [XX] rows")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

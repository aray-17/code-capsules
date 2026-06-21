#!/usr/bin/env python3
"""
Scorer for CLAIM 4 (SOFT) -- the model-tier escalation gate.

Reproduces, OFFLINE from committed eval data (no Docker / API / model calls),
the paper's numbers in Sections 1, 4.3 (user-story) and 8/13 (escalation /
negative warm-start):

  (A) Strong tier converts 21 of 21 gate-positive instances escalated to it,
      against 15 of 17 gate-negative.                  paper.tex L197-198, L816, L1949-1951
  (B) Regression-gated Opus escalation: 105/150 at $1.28 per resolve,
      +21 over the honest single-tier controller's 84.   paper.tex L817, L1947-1948
  (C) Base two-configuration diverse-select cost ~ $0.97/resolve = $118.75/122.  paper.tex L2427-2428
  (D) regok close/lost split (warm-start probe over cheap-tier patch-bearing
      failures): close set rescued 14 of 17, lost set 30 of 32.   paper.tex L1966-1967
  (E) Held-out: 37% of gate-negative carried patches in fact resolve.  paper.tex L820, L1957

Data sources (all committed):
  evals/leakfree/exp4_lever_floor100_siginject.jsonl  -- the run-both lever, first-150
                                                         (candidates incl. floor/siginject grades,
                                                          regression_ok, gold_resolved, lever_decision,
                                                          oracle_union_gold, lever_cost_usd)
  evals/p9_opus_floor_n150_combined_20260527.jsonl    -- Opus-floor sweep, first-150 (resolved, cost_usd)
  evals/leakfree/exp4_lever_second150.jsonl           -- the lever, held-out (second-150)
  evals/leakfree/warmstart_opus_cold.jsonl            -- Opus escalation of the 49 cheap-tier
                                                         patch-bearing failures (cold cascade -- the
                                                         canonical re-derive-everything escalation,
                                                         see paper.tex L2553)
  evals/leakfree/e1b_detector.jsonl                   -- regression-suite gate (agentless_regok) on
                                                         the same 49 patch-bearing failures

Two regression-gate conventions appear in the paper, and the data carries both:

  (A/B) escalation gate (the deployable actuator):
        gate-POSITIVE = regression suite definitively GREEN (regok is True) -> escalate
        gate-NEGATIVE = regression suite RED *or* UNDETERMINED (regok False or None) -> abandon
        Of the 38 abandons, 21 are gate-positive and 17 are gate-negative
        (14 red + 3 undetermined/None).

  (D) close/lost descriptive split:
        close = regression suite NOT red (green or undetermined, regok True or None)
        lost  = regression suite RED (regok is False)
        Here "lost" means the suite specifically went red ("on the wrong track"); an
        undetermined suite is grouped with close ("nearly there"). Of the 49 cheap-tier
        patch-bearing failures, 17 are close (15 green + 2 None) and 32 are lost (red).

The two splits differ only in how the handful of UNDETERMINED (None) cases are bucketed,
and each convention is the natural/conservative one for its decision. Both are reported
exactly as the paper states (paper.tex L1962-1964 defines close=green / lost=red).

Run:
  PYTHONNOUSERSITE=1 python3 verify_escalation_gate.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEVER_FIRST = os.path.join(REPO, "evals/leakfree/exp4_lever_floor100_siginject.jsonl")
LEVER_HELDOUT = os.path.join(REPO, "evals/leakfree/exp4_lever_second150.jsonl")
OPUS_FLOOR = os.path.join(REPO, "evals/p9_opus_floor_n150_combined_20260527.jsonl")
WARMSTART_COLD = os.path.join(REPO, "evals/leakfree/warmstart_opus_cold.jsonl")
E1B_DETECTOR = os.path.join(REPO, "evals/leakfree/e1b_detector.jsonl")


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
    opus_rows = load_jsonl(OPUS_FLOOR)
    opus = {d["instance_id"]: d for d in opus_rows}

    # =====================================================================
    # (B) base honest single-tier delivered = 84; lever total cost = $118.75
    #     (deployment-honest = lever_decision in {SHIP, NO_REPRO_FALLBACK}
    #      AND the shipped candidate gold-resolved)
    # =====================================================================
    base = [
        d for d in lever
        if d["lever_decision"] in ("SHIP", "NO_REPRO_FALLBACK")
        and d.get("shipped_gold_resolved")
    ]
    base_n = len(base)
    lever_cost = sum(d["lever_cost_usd"] for d in lever)

    # =====================================================================
    # (A) the 38 abandons, split by the selected candidate's regression gate,
    #     joined to the Opus-floor sweep for the recovery counts.
    # =====================================================================
    abandons = [d for d in lever if d["lever_decision"] == "ABANDON"]

    def sel_regok(d):
        return d["candidates"].get(d["selected_config"], {}).get("regression_ok")

    gate_pos = [d for d in abandons if escalation_gate_positive(sel_regok(d))]
    gate_neg = [d for d in abandons if not escalation_gate_positive(sel_regok(d))]

    def opus_resolved(d):
        o = opus.get(d["instance_id"])
        return bool(o and o.get("resolved"))

    gp_total = len(gate_pos)
    gp_recovered = sum(1 for d in gate_pos if opus_resolved(d))
    gn_total = len(gate_neg)
    gn_recovered = sum(1 for d in gate_neg if opus_resolved(d))

    # =====================================================================
    # (B cont.) escalate the gate-positive abandons to Opus.
    #     resolved = base + gate-positive Opus recoveries; cost = lever + Opus.
    # =====================================================================
    opus_esc_cost = sum(opus[d["instance_id"]]["cost_usd"] for d in gate_pos)
    gov2_resolved = base_n + gp_recovered
    gov2_cost = lever_cost + opus_esc_cost
    gov2_cpr = gov2_cost / gov2_resolved

    # =====================================================================
    # (C) base two-config diverse-select cost = $118.75 / 122 gold-graded.
    # =====================================================================
    union_resolved = sum(1 for d in lever if d.get("oracle_union_gold"))
    base_cpr = lever_cost / union_resolved

    # =====================================================================
    # (D) regok close/lost split on the warm-start probe (cold cascade) over
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
    # (E) held-out gate-negative carried-patch resolve rate = 37%.
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
    def check(label, got, want, paper):
        nonlocal ok
        match = got == want
        ok = ok and match
        print(f"  [{'OK ' if match else 'XX '}] {label}: {got}  (paper: {want}, {paper})")

    print("CLAIM 4 -- model-tier escalation gate")
    print()
    print("(A) Strong-tier recovery on the 38 abandons, split by regression gate:")
    check("gate-positive recovered", f"{gp_recovered} of {gp_total}", "21 of 21", "L197/L816/L1949")
    check("gate-negative recovered", f"{gn_recovered} of {gn_total}", "15 of 17", "L1950-1951")
    print()
    print("(B) Regression-gated Opus escalation:")
    check("honest single-tier base", base_n, 84, "L1948")
    check("lever total cost", f"${lever_cost:.2f}", "$118.75", "L2428")
    check("Opus escalation cost (21)", f"${opus_esc_cost:.2f}", "$15.61", "deep-eval/tracking")
    check("escalation resolved", f"{gov2_resolved}/150", "105/150", "L817/L1947")
    check("escalation net gain over base", gov2_resolved - base_n, 21, "L1947 (+21)")
    check("escalation cost-per-resolve", f"${gov2_cpr:.2f}", "$1.28", "L817/L1947")
    print(f"       (unrounded $/res = ${gov2_cpr:.4f}; total cost ${gov2_cost:.2f})")
    print()
    print("(C) Base two-config diverse-select cost (gold-graded replay):")
    check("gold-graded resolved", union_resolved, 122, "L2428 ($118.75/122)")
    check("base cost-per-resolve", f"${base_cpr:.2f}", "$0.97", "L2427")
    print(f"       (unrounded base $/res = ${base_cpr:.4f})")
    print()
    print("(D) regok close/lost split (cold cascade over 49 patch-bearing failures):")
    check("close (regok not-red) rescued", f"{close_res} of {close_tot}", "14 of 17", "L1966-1967")
    check("lost (regok red) rescued", f"{lost_res} of {lost_tot}", "30 of 32", "L1967")
    print()
    print("(E) Held-out gate-negative carried-patch resolve rate:")
    check("P(resolve | regok-red, has_patch)", f"{gn_rate*100:.0f}%", "37%", "L820/L1957")
    print(f"       (= {gn_patch_res}/{gn_patch} = {gn_rate:.3f})")
    print()
    print("RESULT:", "ALL NUMBERS REPRODUCE" if ok else "DISCREPANCY -- see [XX] rows")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Run every claim-reproduction scorer and report PASS/FAIL per paper claim.

This is the offline "source of validation" gate for the paper's key claims. It runs
each claim's reproduction scorer over the committed evaluation data (no Docker, no API,
no model calls) and checks that the scorer's output contains the load-bearing numbers
the paper states. It writes benchmarks/claims_results.json, which the claims-explorer
web app (benchmarks/explorer/) renders.

Run:
    python3 benchmarks/verify_criteria.py

Exit code 0 iff every claim reproduces.

Every scorer lives under benchmarks/ (benchmarks/swebench/ and benchmarks/cross_vendor/);
each runs as a standalone script from the repo root. The three offline replays formerly
under tools/ (agreement_signal, h2h_combine_score, value_of_resolve) now live in
benchmarks/swebench/ alongside the rest.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT / "src")}

# Each claim: a scorer command (run from repo root) plus the exact substrings that
# must appear in its output for the claim to count as reproduced. Markers are the
# paper's load-bearing numbers, so a PASS means the committed data regenerates them.
CLAIMS = [
    dict(
        id="C1",
        # Gated offline: agreement_signal.py now reproduces Figure 4 (§9.1) EXACTLY on
        # the n=126 in-family Sonnet intersection (universe read from the canonical
        # overlays under evals/scopeC/menu/), so the offline figures match the paper.
        title="Diverse-sample agreement: doomed precision + anti-circularity",
        section="sec:agreement_signal, sec:agreement_validity",
        paper="Figure 4 (§9.1, n=126 in-family): single-config 64.5%, median pair 66.7%, "
              "best pair floor+siginject 90.9% (fires 22, false 2); two-config select 104/126 @ $85.02 "
              "($0.82/res), floor-alone 95/126 @ $43.20, 3rd config adds 0; independent ~22% (6/27); "
              "Agentless recovers 0 on the Sonnet-doomed set (anti-circularity)",
        cmd=["python3", "benchmarks/swebench/agreement_signal.py"],
        markers=["MATCH: all 19 Figure 4", "precision 90.9%", "Agentless recovers 0"],
        data=["evals/scopeC/menu/canonical_p10_sonnet_*_n150_*.jsonl",
              "evals/p10_sonnet_*_n150_*.jsonl",
              "evals/scopeC/opus/canonical_floor.jsonl",
              "evals/scopeC/canonical_agentless.jsonl",
              "evals/p9_opus_floor_n150_combined_20260527.jsonl",
              "evals/h2h_agentless_sonnet_n150_run1_20260528.jsonl"],
    ),
    dict(
        id="C3",
        title="Regression-suite gate restores wrong-abandons at zero cost",
        section="sec:agreement_regression_gate",
        paper="first-150 honest 84 -> gate 99 (baseline 111), held-out 76 -> 89 (baseline 93); "
              "P(resolve|regression-green)=0.792; SHIP precision 77%/75%, 28 false-positive ships",
        cmd=["python3", "benchmarks/swebench/regression_gate.py"],
        markers=["honest=84  gate=99  floor-alone=111", "0.792", "ALL CLAIM-3 NUMBERS REPRODUCE PAPER: True"],
        data=["evals/leakfree/exp4_lever_floor100_siginject.jsonl",
              "evals/leakfree/exp4_lever_second150.jsonl",
              "evals/p10_sonnet_floor_n150_20260516T040248.jsonl",
              "evals/p10_sonnet_siginject_b10x3_n150_20260517T035438.jsonl",
              "evals/canonical_p10_sonnet_strat_holdout_floor_20260520T075105.jsonl",
              "evals/canonical_p10_sonnet_strat_holdout_siginject_b10x3_20260517T173631.jsonl"],
    ),
    dict(
        id="C4",
        title="Model-tier escalation gate",
        section="sec:agreement_escalation",
        paper="21/21 gate-positive and 15/17 gate-negative recovered; base 64 -> 85/150 at $1.58 per resolve",
        cmd=["python3", "benchmarks/swebench/escalation_gate.py"],
        markers=["escalation resolved: 85/150", "$1.58", "ALL NUMBERS REPRODUCE"],
        data=["evals/leakfree/exp4_lever_floor100_siginject.jsonl",
              "evals/leakfree/exp4_lever_second150.jsonl",
              "evals/scopeC/opus/canonical_floor.jsonl",
              "evals/p9_opus_floor_n150_combined_20260527.jsonl"],
    ),
    dict(
        id="C5",
        title="Two-pass critique vs Agentless head-to-head (Pareto)",
        section="sec:h2h",
        paper="Code-Capsules 172/300 (57.3%) @ $0.445 vs Agentless 152/300 (50.7%) @ $0.456 (+6.7pp, Pareto)",
        cmd=["python3", "benchmarks/swebench/h2h_combine_score.py", "--no-write"],
        markers=["172/300", "PARETO"],
        data=["evals/leakfree/tb_forcestage2_first150.jsonl",
              "evals/leakfree/tb_forcestage2_second150.jsonl",
              "evals/h2h_agentless_sonnet_n150_run1_20260528.jsonl",
              "evals/h2h_agentless_sonnet_150_300_sympyfixed_20260601T003717.jsonl"],
    ),
    dict(
        id="C8",
        title="Cross-vendor HumanEval/MBPP cost ratios",
        section="tab:cross_vendor_hemmbpp",
        paper="HumanEval 98.2/92.7%, MBPP 53.0/50.2%; cost ratios 4.8x (HE), 15.9x (MBPP)",
        cmd=["python3", "benchmarks/cross_vendor/score.py"],
        markers=["4.8x", "15.9x", "98.2%", "53.0%"],
        data=["evals/cross_vendor_humaneval_n164_postfix_20260525.csv",
              "evals/cross_vendor_mbpp_n500_postfix_20260525.csv"],
    ),
    dict(
        id="C11",
        title="Value-of-resolve decision rule (escalation economics)",
        section="sec:negative_cascade",
        paper="base $0.82/resolve (n=126 in-family); tier escalation $1.37 / $2.71 / $1.47 per recovered resolve",
        cmd=["python3", "benchmarks/swebench/value_of_resolve.py"],
        markers=["base $/resolve: $0.82", "$1.37", "ALL NUMBERS REPRODUCE"],
        data=["evals/scopeC/value_of_resolve_corrected.json",
              "evals/scopeC/menu/canonical_p10_sonnet_floor_n150_20260516T040248.jsonl",
              "evals/p9_opus_floor_n150_combined_20260527.jsonl"],
    ),
    dict(
        id="C2",
        title="Oracle-relative governor (run-both lever within two instances of the oracle)",
        section="sec:agreement_oracle",
        paper="run-both lever 122/150 @ $118.75, within two instances of the oracle union 124/150; "
              "ship precision 77% (first-150) / 70% (held-out)",
        cmd=["python3", "benchmarks/swebench/oracle_governor.py"],
        markers=["oracle union ceiling: 124", "lever pair (floor+siginject) resolved: 122",
                 "ALL TARGET NUMBERS REPRODUCE"],
        data=["evals/leakfree/exp4_lever_floor100_siginject.jsonl",
              "evals/leakfree/exp4_lever_second150.jsonl",
              "evals/p10_sonnet_floor_n150_20260516T040248.jsonl"],
    ),
    dict(
        id="C6",
        title="Calibrated cross-tier Pareto menu",
        section="tab:shipped_presets, tab:cross_tier_cells",
        paper="cost_min 57/$0.48, balanced 66/$0.93, quality 77/$0.91, quality_max 92/$0.78, ceiling 96/$0.94; "
              "quality_max out-resolves quality at lower $/resolve",
        cmd=["python3", "benchmarks/swebench/cross_tier_pareto_menu.py"],
        markers=["ALL CELLS REPRODUCE", "more resolved? True", "cheaper per resolve? True"],
        data=["evals/p10_sonnet_p8c_n150_20260515T235913.jsonl",
              "evals/leakfree/tb_forcestage2_first150.jsonl",
              "evals/p9_opus_implicit20_n150_combined_20260527.jsonl",
              "evals/p9_opus_floor_n150_combined_20260527.jsonl"],
    ),
    dict(
        id="C9",
        title="Workload capability ceiling (cross-tier oracle union)",
        section="sec:ceiling",
        paper="union 148/150 (98.7%); 2 doomed; Opus alone covers 143/148",
        cmd=["python3", "benchmarks/swebench/ceiling_union.py"],
        markers=["union=148/150", "doomed=2", "RESULT: ALL MATCH"],
        data=["evals/p10_sonnet_*_n150_*.jsonl",
              "evals/p9_opus_floor_n150_combined_20260527.jsonl",
              "evals/p9_opus_implicit20_n150_combined_20260527.jsonl"],
    ),
    dict(
        id="C10",
        title="sympy canonical bin/test scoring (Agentless 31/77)",
        section="sec:h2h, sec:ceiling",
        paper="Agentless resolves 31/77 sympy under the canonical bin/test runner",
        cmd=["python3", "benchmarks/swebench/sympy_canonical_score.py"],
        markers=["31/77", "RESULT: PASS"],
        data=["evals/cc_sympy_rerun_20260601T003717.jsonl",
              "evals/h2h_agentless_sonnet_150_300_sympyfixed_20260601T003717.jsonl"],
    ),
    dict(
        id="C12",
        title="Plan-then-execute helps every model; two-pass critique adds lift on both vendors",
        section="sec:cross_vendor_findings",
        paper="Sonnet signaled->plan +6pp, codex +2pp; two-pass critique adds lift on both (Sonnet +3pp, codex +11pp)",
        cmd=["python3", "benchmarks/swebench/claim12_plan_critique_lift.py"],
        markers=["ALL PER-CELL PASS COUNTS MATCH PAPER: True"],
        data=["evals/p10_sonnet_planFirst_b20_n150_20260516T152721.jsonl",
              "evals/p10_sonnet_signaled10_n150_20260515T231227.jsonl"],
    ),
    dict(
        id="C13",
        title="Held-out cascade generalization (value-of-resolve lever, off-django split)",
        section="sec:negative_cascade",
        paper="held-out second-150: signaled-10 -> UNBOUNDED Opus cascade resolves 124/150, "
              "+31 over the Sonnet floor (93); Opus recovers 73% of the escalated failures; "
              "quality-max at $159 vs floor $84 (more resolves at added cost, not a both-axes win)",
        cmd=["python3", "benchmarks/swebench/held_out_cascade.py"],
        markers=["124/150", "+31 over floor", "Opus recovery 73%", "ALL NUMBERS REPRODUCE"],
        data=["evals/heldout/cascade_signaled10_n150_canonical.jsonl",
              "evals/heldout/cascade_floor_n150_canonical.jsonl",
              "evals/heldout/cascade_opus_floor_escalated_canonical.jsonl"],
    ),
]


def run(claim: dict):
    # Deployment-governor claims (paper sec:7) are cited to the paper, not gated
    # offline here: they are calibrated on the n=126 all-configs-attempted universe,
    # while the committed cells are scored at n=150, so the offline figures differ
    # from the paper's by design. The headline claims below are gated offline.
    if claim.get("paper_referenced"):
        return "PAPER-REF", "", []
    p = subprocess.run(claim["cmd"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    out = p.stdout + p.stderr
    missing = [m for m in claim["markers"] if m not in out]
    status = "PASS" if (p.returncode == 0 and not missing) else "FAIL"
    return status, out, missing


def main() -> int:
    results = []
    npass = 0
    print("=" * 72)
    print("Code-Capsules claim verification (offline; no Docker / API / model)")
    print("=" * 72)
    for c in CLAIMS:
        status, out, missing = run(c)
        if status == "PASS":
            npass += 1
        results.append({
            "id": c["id"], "title": c["title"], "section": c["section"],
            "paper": c["paper"], "data": c["data"], "cmd": " ".join(c["cmd"]),
            "status": status, "missing_markers": missing,
            # The load-bearing numbers checked, and the scorer's reproduced output
            # (the in-browser proof for the evidence explorer). Capped to keep
            # claims_results.json small.
            "markers": c["markers"],
            "reproduced": out if len(out) <= 8000 else out[:4000] + "\n...\n" + out[-3500:],
        })
        print(f"[{status}] {c['id']:<4} {c['title']}")
        if missing:
            print(f"        missing markers: {missing}")
    gated = [c for c in CLAIMS if not c.get("paper_referenced")]
    n_ref = len(CLAIMS) - len(gated)
    out_path = ROOT / "benchmarks" / "claims_results.json"
    out_path.write_text(json.dumps(
        {"claims": results,
         "summary": {"pass": npass, "gated": len(gated), "paper_referenced": n_ref}}, indent=2))
    print("-" * 72)
    print(f"{npass}/{len(gated)} headline claims reproduce offline  ->  {out_path.relative_to(ROOT)}")
    print(f"{n_ref} deployment-governor claims (run-both lever, regression "
          f"gate, escalation gate, value-of-resolve) are detailed in the paper (sec:7).")
    return 0 if npass == len(gated) else 1


if __name__ == "__main__":
    sys.exit(main())

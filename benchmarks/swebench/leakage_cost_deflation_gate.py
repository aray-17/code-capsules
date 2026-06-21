#!/usr/bin/env python3
"""Claim 7 scorer -- evaluation-gated leakage cost deflation.

Paper claim (paper.tex):
  An earlier configuration of the two-pass critique variant triggered its
  second pass from the benchmark's held-out evaluation verdict, skipping the
  second pass whenever the verdict reported the first pass had already
  resolved the instance. Consulting the held-out verdict inside the control
  flow silently deflated the variant's measured per-attempt cost. The
  corrected configuration runs the second pass unconditionally.

  Retired (leaky, evaluation-gated): "$0.308" / attempt   (prose "~$0.31")
        cross-vendor Table, paper.tex:2085 -- 102/150 @ $0.308
  Corrected (honest, always-run):    "$0.408" / attempt   (headline "$0.41")
        appendix paper.tex:2067 -- 98/150 @ $0.408; headline L694/L1209
  Stated deflation factor:           "a factor of $1.32$"
        abstract; sec:negative_leakage; sec:limitations; appendix

This scorer reproduces the two per-instance mean costs and their ratio
OFFLINE from committed eval JSONL data. No Docker / API / model.

The leak mechanism is independently verifiable in the raw data (NOT
hardcoded):
  * leaky file: 56/150 rows carry escalation_reason
    "already resolved at budget=10" AND stage2_turns==0 -- the second pass
    was skipped because the held-out evaluation verdict said pass 1 already
    resolved the instance. Those rows pay only the stage-1 cost (mean
    ~$0.146 vs ~$0.405 for rows that ran both passes).
  * honest file: all 150 rows carry escalation_reason
    "force_stage2 (always run both passes; no in-loop oracle)" with
    stage2_turns>=2 -- the second pass runs unconditionally.

Resolved-count cross-check ties each file to its exact paper row:
  leaky 102/150 (Table cross_vendor_variants), honest 98/150 (appendix).

This scorer reproduces both per-attempt costs ($0.308 and $0.408) and
their deflation ratio. The honest/leaky cost ratio is 1.3232 -> 1.32,
matching the paper's stated "a factor of 1.32" (the paper's own rounded
costs agree: 0.41/0.31 = 1.3226 -> 1.32, 0.408/0.308 = 1.3247 -> 1.32).
HISTORICAL NOTE: earlier paper drafts printed "1.31", an off-by-one typo
in the second decimal; corrected to 1.32 on 2026-06-14. The mismatch
branch below is retained as a regression guard.
"""

import json
from pathlib import Path

# Repo-relative root: this file lives at benchmarks/swebench/<name>.py
ROOT = Path(__file__).resolve().parents[2]
LEAKY = ROOT / "evals" / "p10_sonnet_twopass_b10p15_n150_20260516T152721.jsonl"
HONEST = ROOT / "evals" / "leakfree" / "tb_forcestage2_first150.jsonl"

# Paper-stated targets (paper.tex).
PAPER_LEAKY_COST = 0.308     # "$0.308" appendix/Table; "~$0.31" prose
PAPER_HONEST_COST = 0.408    # "$0.408" appendix; "$0.41" headline
PAPER_RATIO = 1.32           # "a factor of 1.32" (abstract/sec5/sec-limits/appendix)
PAPER_LEAKY_RESOLVED = 102   # Table cross_vendor_variants: 102/150
PAPER_HONEST_RESOLVED = 98   # appendix / headline: 98/150


def load(path):
    """Return (n_rows, costs, escalation_reason_counts, resolved_count,
    n_pass2_skipped)."""
    costs = []
    reasons = {}
    resolved = 0
    pass2_skipped = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            costs.append(float(row["cost_usd"]))
            if row.get("resolved"):
                resolved += 1
            if (row.get("stage2_turns") or 0) == 0:
                pass2_skipped += 1
            r = row.get("escalation_reason", "") or ""
            if "already resolved" in r:
                key = "already_resolved_skip_pass2"
            elif "force_stage2" in r:
                key = "force_stage2_always_run"
            else:
                key = "other/none"
            reasons[key] = reasons.get(key, 0) + 1
    return len(costs), costs, reasons, resolved, pass2_skipped


def mean(xs):
    return sum(xs) / len(xs)


def main():
    n_l, c_l, r_l, res_l, skip_l = load(LEAKY)
    n_h, c_h, r_h, res_h, skip_h = load(HONEST)
    m_l = mean(c_l)
    m_h = mean(c_h)
    ratio = m_h / m_l

    print("=" * 72)
    print("CLAIM 7 -- evaluation-gated leakage cost deflation")
    print("=" * 72)
    print()
    print("LEAKY  (evaluation-gated; pass 2 skipped when verdict says resolved)")
    print(f"  file:  {LEAKY.relative_to(ROOT)}")
    print(f"  rows:  {n_l}")
    print(f"  escalation_reason buckets: {r_l}")
    print(f"  pass-2-skipped (stage2_turns==0) rows: {skip_l}")
    print(f"  resolved: {res_l}/{n_l}   (paper: {PAPER_LEAKY_RESOLVED}/150)")
    print(f"  mean cost_usd/instance: {m_l:.6f}"
          f"   (paper: ${PAPER_LEAKY_COST:.3f} / '~$0.31')")
    print(f"  total cost_usd:         {sum(c_l):.4f}")
    print()
    print("HONEST (force_stage2; second pass runs unconditionally)")
    print(f"  file:  {HONEST.relative_to(ROOT)}")
    print(f"  rows:  {n_h}")
    print(f"  escalation_reason buckets: {r_h}")
    print(f"  pass-2-skipped (stage2_turns==0) rows: {skip_h}")
    print(f"  resolved: {res_h}/{n_h}   (paper: {PAPER_HONEST_RESOLVED}/150)")
    print(f"  mean cost_usd/instance: {m_h:.6f}"
          f"   (paper: ${PAPER_HONEST_COST:.3f} / '$0.41')")
    print(f"  total cost_usd:         {sum(c_h):.4f}")
    print()
    print("DEFLATION FACTOR  honest / leaky")
    print(f"  computed: {ratio:.4f}   (paper-stated: {PAPER_RATIO:.2f})")
    print(f"  computed rounds to {round(ratio, 2):.2f}")
    print(f"  paper rounded costs 0.41/0.31 = {0.41/0.31:.4f} -> {round(0.41/0.31,2):.2f}")
    print(f"  paper appendix 0.408/0.308 = {0.408/0.308:.4f} -> {round(0.408/0.308,2):.2f}")
    print()

    leaky_cost_ok = round(m_l, 3) == round(PAPER_LEAKY_COST, 3)
    honest_cost_ok = round(m_h, 3) == round(PAPER_HONEST_COST, 3)
    leaky_res_ok = res_l == PAPER_LEAKY_RESOLVED
    honest_res_ok = res_h == PAPER_HONEST_RESOLVED
    ratio_ok = round(ratio, 2) == round(PAPER_RATIO, 2)

    print("MATCH SUMMARY")
    print(f"  leaky  resolved {res_l}/150 vs paper {PAPER_LEAKY_RESOLVED}/150"
          f"  -> {'MATCH' if leaky_res_ok else 'MISMATCH'}")
    print(f"  honest resolved {res_h}/150 vs paper {PAPER_HONEST_RESOLVED}/150"
          f"  -> {'MATCH' if honest_res_ok else 'MISMATCH'}")
    print(f"  leaky  mean  ${m_l:.3f} vs paper ${PAPER_LEAKY_COST:.3f}"
          f"  -> {'MATCH' if leaky_cost_ok else 'MISMATCH'}")
    print(f"  honest mean  ${m_h:.3f} vs paper ${PAPER_HONEST_COST:.3f}"
          f"  -> {'MATCH' if honest_cost_ok else 'MISMATCH'}")
    print(f"  ratio  {round(ratio,2):.2f} vs paper-stated {PAPER_RATIO:.2f}"
          f"  -> {'MATCH' if ratio_ok else 'MISMATCH'}")
    print()

    costs_and_counts_ok = (leaky_cost_ok and honest_cost_ok
                           and leaky_res_ok and honest_res_ok)
    if costs_and_counts_ok and not ratio_ok:
        print("REGRESSION GUARD: the per-attempt costs and resolved counts")
        print(f"reproduce, but the deflation ratio {round(ratio,2):.2f} no longer")
        print(f"matches the paper-stated {PAPER_RATIO:.2f}. The paper was corrected")
        print("to 1.32 on 2026-06-14; if this fires, the data or the figure changed.")
        print()

    # Costs, counts, and the deflation ratio all reproduce against the paper.
    print("DATA REPRODUCTION (costs + counts): "
          f"{'PASS' if costs_and_counts_ok else 'FAIL'}")
    print(f"PAPER RATIO {PAPER_RATIO:.2f} REPRODUCES: "
          f"{'YES' if ratio_ok else f'NO (true={round(ratio,2):.2f})'}")
    return costs_and_counts_ok and ratio_ok


if __name__ == "__main__":
    main()
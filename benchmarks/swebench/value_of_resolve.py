"""Value-of-resolve decision rule (Phase 12 #3). Pure reanalysis, zero new compute.

Formalizes the qualitative economics in the paper's cascade section into an explicit rule:
model-tier escalation of the agreement-doomed set is worth it iff the value of a resolved
instance V exceeds that escalation's cost-per-recovered-resolve. Uses the same offline
loaders + canonical doomed sets as benchmarks/swebench/agreement_signal.py; reproduces the
paper's $1.40 / $2.20 / $3.07 / $7.51 figures.

  python3 benchmarks/swebench/value_of_resolve.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

# Anchor eval globs on the repo root so the scorer reproduces from any CWD
# (verify_criteria.py runs it with cwd=ROOT; a bare standalone invocation from
# elsewhere would otherwise silently get empty results).
ROOT = Path(__file__).resolve().parents[2]


def _L(pattern):
    fs = [x for x in glob.glob(str(ROOT / pattern)) if "bak" not in x and "invalid" not in x]
    if not fs:
        return None
    f = max(fs, key=lambda p: sum(1 for _ in open(p)))
    d = {}
    for line in open(f):
        line = line.strip()
        if line:
            r = json.loads(line)
            d[r["instance_id"]] = dict(res=(r.get("resolved") is True),
                                       cost=float(r.get("cost_usd") or 0.0))
    return d


def main():
    sf = _L("evals/p10_sonnet_floor_n150_*.jsonl")
    ss = _L("evals/p10_sonnet_siginject_b10x3_n150_*.jsonl")
    of = _L("evals/p9_opus_floor_n150_combined_20260527.jsonl")
    osig = _L("evals/opus_siginject_agreementfail65.jsonl")
    hf = _L("evals/p10_haiku_floor_n150_*.jsonl")
    hi = _L("evals/p10_haiku_implicit40_n150_*.jsonl")

    # base operating cost: the 2-config diverse-select (floor+siginject), keep either resolver
    ids = sorted(set(sf) & set(ss))
    base_cost = sum(sf[i]["cost"] + ss[i]["cost"] for i in ids)
    base_res = sum(1 for i in ids if sf[i]["res"] or ss[i]["res"])
    base_per = base_cost / base_res
    print(f"BASE (2-config diverse-select): {base_res}/{len(ids)} resolved at ${base_cost:.2f} "
          f"=> ${base_per:.2f} per resolve  (always worth it for value-of-resolve V > ${base_per:.2f})")

    rows = []  # (label, set_size, recovered, total_cost)
    # the Sonnet agreement-doomed set (both floor+siginject fail at b10)
    s_doomed = [i for i in ids if not sf[i]["res"] and not ss[i]["res"]]
    if of:
        rec = [i for i in s_doomed if of.get(i, {}).get("res")]
        rows.append(("Sonnet-doomed -> Opus (1 config)", len(s_doomed), len(rec),
                     sum(of.get(i, {}).get("cost", 0) for i in s_doomed)))
    if of and osig:
        rec = [i for i in s_doomed if of.get(i, {}).get("res") or osig.get(i, {}).get("res")]
        rows.append(("Sonnet-doomed -> Opus (2 configs)", len(s_doomed), len(rec),
                     sum(of.get(i, {}).get("cost", 0) + osig.get(i, {}).get("cost", 0) for i in s_doomed)))
    # the Haiku agreement-doomed set escalated to Opus (more headroom from a weaker start)
    if hf and hi and of:
        hids = sorted(set(hf) & set(hi) & set(of))
        h_doomed = [i for i in hids if not hf[i]["res"] and not hi[i]["res"]]
        rec = [i for i in h_doomed if of.get(i, {}).get("res")]
        rows.append(("Haiku-doomed -> Opus (1 config)", len(h_doomed), len(rec),
                     sum(of.get(i, {}).get("cost", 0) for i in h_doomed)))

    print(f"\n{'escalation option':<34} {'recovers':>14} {'$ total':>9} {'$/resolve':>10}  escalate iff V >=")
    for label, n, rec, cost in rows:
        per = cost / rec if rec else float("inf")
        thr = f"${per:.2f}" if rec else "never"
        print(f"  {label:<32} {rec:>3}/{n:<3} ({rec/n:>3.0%})   ${cost:>6.2f}   ${per:>7.2f}   {thr}")

    print(f"\nDECISION RULE: run the base everywhere (V > ${base_per:.2f}); escalate the "
          f"agreement-doomed set to a stronger tier ONLY when value-of-resolve V exceeds that "
          f"option's $/resolve (${min(c/r for _,_,r,c in rows if r):.2f} cheapest to "
          f"${max(c/r for _,_,r,c in rows if r):.2f} richest). The 3-tier+harness core never recovers: abandon at any V.")


if __name__ == "__main__":
    main()

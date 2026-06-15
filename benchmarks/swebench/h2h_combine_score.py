#!/usr/bin/env python3
"""
Combine the two additive Agentless baseline halves (first-150 + second-150) into
a single full-300 file and score it, per the plan in evals/h2h_baselines.md.

Reporting basis (decided 2026-05-30):
  - patch_selection = ORACLE for both halves (rerank was infeasible on the Linux
    server; we cite published Agentless + report the oracle equivalent as the
    diligence number). Both halves must share the same selection mode or the
    combine is apples-to-oranges -- the script asserts this.
  - Harness artifacts (e.g. the 300-token file-localization truncation that
    yielded found_files=[] for some sphinx instances) are counted as GENUINE
    FAILURES, not excluded. Denominator stays the full 300. They are listed
    separately for transparency only.

Usage:
  python3 benchmarks/swebench/h2h_combine_score.py                      # uses defaults below
  python3 benchmarks/swebench/h2h_combine_score.py --first A.jsonl --second B.jsonl --out C.jsonl
  python3 benchmarks/swebench/h2h_combine_score.py --no-write           # score only, don't write combined
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # benchmarks/swebench/X.py -> repo root
EVALS = ROOT / "evals"

DEFAULT_FIRST = EVALS / "h2h_agentless_sonnet_n150_run1_20260528.jsonl"
# sympy-corrected second-150 (the canonical full-300 Agentless half). The earlier
# _run1_20260530 file scored sympy under the pytest eval-bug (0/77) and is superseded.
DEFAULT_SECOND = EVALS / "h2h_agentless_sonnet_150_300_sympyfixed_20260601T003717.jsonl"

# Code-Capsules full-300 = the honest force_stage2 two-pass-critique re-run (the
# leakage-free deployable two-pass, which is the paper's headline H2H cell). The earlier
# p10 twopass first-150 file was the evaluation-gated (leaky) cell at $0.31 / 70-of-150
# and is superseded; the force_stage2 files are sympy-eval-fixed and carry honest cost.
# This makes the shipped scorer reproduce the paper's 136/300 @ $0.436 exactly.
CC_FIRST = EVALS / "leakfree" / "tb_forcestage2_first150.jsonl"    # 66/150, honest
CC_SECOND = EVALS / "leakfree" / "tb_forcestage2_second150.jsonl"  # 70/150, honest, sympy-fixed

# Known harness artifacts: counted as failures, listed for transparency.
KNOWN_ARTIFACTS = {
    "sphinx-doc__sphinx-8474",
    "sphinx-doc__sphinx-8627",
}


def load(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def repo_of(rec: dict) -> str:
    return rec.get("repo") or rec["instance_id"].split("__")[0].replace("-", "/", 1)


def build_cc_full300() -> list[dict] | None:
    """Assemble the Code-Capsules two-pass full-300 from the honest force_stage2
    re-run: first-150 + second-150 (sympy-eval-fixed). This is the paper's headline
    H2H cell. Returns None if any source file is missing."""
    for p in (CC_FIRST, CC_SECOND):
        if not p.exists():
            return None
    return load(CC_FIRST) + load(CC_SECOND)


def per_repo_map(rows: list[dict]) -> dict[str, list[int]]:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        a = agg[repo_of(r)]
        a[1] += 1
        if r.get("resolved") is True:
            a[0] += 1
    return agg


def summarize(rows: list[dict], label: str) -> dict:
    n = len(rows)
    resolved = sum(1 for r in rows if r.get("resolved") is True)
    no_patch = sum(1 for r in rows if not r.get("has_patch"))
    errored = sum(1 for r in rows if r.get("error"))
    none_verdict = sum(1 for r in rows if r.get("resolved") is None)
    cost = sum((r.get("cost_usd") or 0.0) for r in rows)
    rate = (resolved / n * 100) if n else 0.0
    print(f"\n=== {label} ===")
    print(f"  records:        {n}")
    print(f"  resolved:       {resolved}  ({rate:.1f}%)")
    print(f"  no patch:       {no_patch}")
    print(f"  errored:        {errored}")
    print(f"  no verdict:     {none_verdict}  (resolved is None)")
    print(f"  cost:           ${cost:.2f}")
    return {"n": n, "resolved": resolved, "rate": rate, "cost": cost}


def per_repo(rows: list[dict]) -> None:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # repo -> [resolved, total]
    for r in rows:
        a = agg[repo_of(r)]
        a[1] += 1
        if r.get("resolved") is True:
            a[0] += 1
    print("\n=== per-repo (full set) ===")
    print(f"  {'repo':28} {'resolved':>9} {'total':>6} {'rate':>7}")
    for repo in sorted(agg):
        res, tot = agg[repo]
        print(f"  {repo:28} {res:>9} {tot:>6} {res/tot*100:>6.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=Path, default=DEFAULT_FIRST)
    ap.add_argument("--second", type=Path, default=DEFAULT_SECOND)
    ap.add_argument("--out", type=Path, default=None,
                    help="combined output path (default: auto-timestamped n300_combined)")
    ap.add_argument("--no-write", action="store_true",
                    help="score only; do not write the combined file")
    ap.add_argument("--expect", type=int, default=150,
                    help="expected records per half (warn if mismatch)")
    args = ap.parse_args()

    for p in (args.first, args.second):
        if not p.exists():
            print(f"ERROR: missing input: {p}", file=sys.stderr)
            return 2

    first = load(args.first)
    second = load(args.second)

    # --- integrity checks -----------------------------------------------------
    problems = []
    if len(first) != args.expect:
        problems.append(f"first half has {len(first)} records (expected {args.expect})")
    if len(second) != args.expect:
        problems.append(f"second half has {len(second)} records (expected {args.expect}) "
                        f"-- run may still be in flight")

    ids_first = [r["instance_id"] for r in first]
    ids_second = [r["instance_id"] for r in second]
    dup_first = [k for k, v in Counter(ids_first).items() if v > 1]
    dup_second = [k for k, v in Counter(ids_second).items() if v > 1]
    if dup_first:
        problems.append(f"duplicate instance_ids in first half: {dup_first}")
    if dup_second:
        problems.append(f"duplicate instance_ids in second half: {dup_second}")
    overlap = set(ids_first) & set(ids_second)
    if overlap:
        problems.append(f"halves are NOT disjoint -- {len(overlap)} shared ids: "
                        f"{sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}")

    sel = {r.get("h2h_patch_selection") for r in first + second}
    sel_clean = {s.split("_")[0] for s in sel if s}  # collapse oracle_empty_fallback_*
    if sel_clean - {"oracle"}:
        problems.append(f"mixed/non-oracle patch_selection across halves: {sorted(sel)}")

    print("=== integrity checks ===")
    if problems:
        for p in problems:
            print(f"  WARNING: {p}")
    else:
        print("  OK: 150+150, disjoint, no dups, oracle selection on both halves")

    # --- per-half + combined summaries ---------------------------------------
    summarize(first, f"first-150  ({args.first.name})")
    summarize(second, f"second-150 ({args.second.name})")

    combined = first + second
    full = summarize(combined, "FULL-300 (combined)")
    per_repo(combined)

    # --- artifact transparency line ------------------------------------------
    present_artifacts = [r["instance_id"] for r in combined
                         if r["instance_id"] in KNOWN_ARTIFACTS]
    if present_artifacts:
        print(f"\n=== harness artifacts (counted as failures, NOT excluded) ===")
        for iid in present_artifacts:
            rec = next(r for r in combined if r["instance_id"] == iid)
            print(f"  {iid}: resolved={rec.get('resolved')} "
                  f"has_patch={rec.get('has_patch')} -- 300-tok file-loc truncation")

    # --- Code-Capsules full-300 (computed from the recipe, not hardcoded) -----
    cc = build_cc_full300()
    print("\n" + "=" * 60)
    print(f"AGENTLESS (oracle) full-300 baseline: "
          f"{full['resolved']}/{full['n']} = {full['rate']:.1f}%  @ ${full['cost']/full['n']:.4f}/inst")
    if cc is None:
        print("  Code-Capsules full-300: recipe files missing -- cannot score CC side")
    else:
        cc_n = len(cc)
        cc_res = sum(1 for r in cc if r.get("resolved") is True)
        cc_cost = sum((r.get("cost_usd") or 0.0) for r in cc)
        cc_rate = cc_res / cc_n * 100 if cc_n else 0.0
        dpp = cc_rate - full["rate"]
        cheaper = (1 - (cc_cost / cc_n) / (full["cost"] / full["n"])) * 100
        # per-repo CC >= Agentless check
        ccm, agm = per_repo_map(cc), per_repo_map(combined)
        violations = [r for r in ccm if ccm[r][0] < agm.get(r, [0, 0])[0]]
        print(f"  CODE-CAPSULES (two-pass) full-300:    "
              f"{cc_res}/{cc_n} = {cc_rate:.1f}%  @ ${cc_cost/cc_n:.4f}/inst")
        print(f"  -> CC {'+' if dpp >= 0 else ''}{dpp:.1f}pp resolve AND "
              f"{cheaper:.0f}% cheaper/instance"
              f"{'  [PARETO]' if dpp >= 0 and cheaper >= 0 else ''}")
        print(f"  -> CC >= Agentless on every repo: "
              f"{'YES' if not violations else 'NO -- ' + ', '.join(violations)}")
    print(f"  total modeled cost (Agentless both halves): ${full['cost']:.2f}")
    print("=" * 60)

    # --- write combined -------------------------------------------------------
    if not args.no_write:
        out = args.out
        if out is None:
            stamp = dt.datetime.now().strftime("%Y%m%d")
            out = EVALS / f"h2h_agentless_sonnet_n300_combined_oracle_{stamp}.jsonl"
        with open(out, "w") as f:
            for r in combined:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote combined -> {out}  ({len(combined)} records)")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

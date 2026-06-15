#!/usr/bin/env python3
"""Offline scorer for the sympy eval-bug recovery claim (paper Claim 10).

PAPER CLAIM (paper.tex):
  - Section "Evaluation correctness (sympy)" (~L2261-2272):
      "... scores all 77 sympy instances as unresolved for EVERY patch ...
       With the correct runner ... the Agentless oracle@k [resolves] 31/77."
  - Section "Workload capability ceiling" (~L2351-2358):
      "... the Agentless baseline resolves 31/77 ... A 0/77 reading arises
       only under an incorrect test runner (pytest in place of sympy's
       bin/test) ..."

  i.e. the SAME Agentless sympy patches go from 0/77 (scored under the
  pytest eval bug) to 31/77 once scored with sympy's bin/test harness.

WHAT THIS SCORER DOES (fully offline, no Docker / no API / no model):
  Both halves of the claim are reproduced from committed eval JSONLs that
  already carry the per-instance `resolved` verdict and an `eval_note`
  tag identifying which runner produced it:

    PRE-FIX  (pytest bug path):  h2h_agentless_sonnet_150_300_run1_*.jsonl
        sympy rows tagged eval_note in {docker, docker timeout ...}
        -> every sympy patch errors before any test runs -> 0/77 resolved.

    POST-FIX (bin/test path):   h2h_agentless_sonnet_150_300_sympyfixed_*.jsonl
        sympy rows tagged eval_note == "docker(sympy)" (the fixed runner;
        see tools/rescore_sympy.py) -> 31/77 resolved.

  The post-fix file is a like-for-like RE-SCORE of the pre-fix file: same 77
  sympy instance ids, byte-identical Agentless patches (h2h_patch), only the
  test runner changed. The scorer asserts that invariant so the 0->31 gain
  cannot be attributed to anything but the eval fix.

Run:  PYTHONNOUSERSITE=1 PYTHONPATH=src python3 benchmarks/swebench/sympy_evalbug_gate.py
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

# Repo-relative root: this file lives at benchmarks/swebench/<name>.py
ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals"

# Paper targets (paper.tex).
TARGET_PREFIX_RESOLVED = 0    # "scores all 77 sympy instances as unresolved"
TARGET_POSTFIX_RESOLVED = 31  # "the Agentless baseline resolves 31/77"
TARGET_TOTAL = 77             # "all 77 sympy instances"

# Committed data files.
PREFIX_FILE = EVALS / "h2h_agentless_sonnet_150_300_run1_20260530T174324.jsonl"
# The sympy-fixed re-score (glob the timestamp so the scorer is robust to it).
POSTFIX_GLOB = "h2h_agentless_sonnet_150_300_sympyfixed_*.jsonl"

# eval_note tags written by the fixed runner (tools/rescore_sympy.py routes
# sympy through `bin/test`; docker_eval tags those rows "docker(sympy)").
BINTEST_NOTE = "docker(sympy)"


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sympy_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["instance_id"].startswith("sympy__")]


def count_resolved(rows: list[dict]) -> int:
    return sum(1 for r in rows if r.get("resolved") is True)


def main() -> int:
    # ---- locate post-fix file ----
    matches = sorted(glob.glob(str(EVALS / POSTFIX_GLOB)))
    if not matches:
        raise SystemExit(f"no post-fix file matching {POSTFIX_GLOB} under {EVALS}")
    postfix_file = Path(matches[-1])

    pre_rows = load(PREFIX_FILE)
    post_rows = load(postfix_file)
    pre_sym = sympy_rows(pre_rows)
    post_sym = sympy_rows(post_rows)

    pre_n, post_n = len(pre_sym), len(post_sym)
    pre_resolved = count_resolved(pre_sym)
    post_resolved = count_resolved(post_sym)

    # ---- like-for-like invariants (so 0->31 is attributable to the eval fix only) ----
    pre_by_id = {r["instance_id"]: r for r in pre_sym}
    post_by_id = {r["instance_id"]: r for r in post_sym}
    same_id_set = set(pre_by_id) == set(post_by_id)
    identical_patch = sum(
        1
        for i in pre_by_id
        if i in post_by_id
        and (pre_by_id[i].get("h2h_patch") or "") == (post_by_id[i].get("h2h_patch") or "")
    )
    flips = [
        i
        for i in pre_by_id
        if i in post_by_id
        and pre_by_id[i].get("resolved") is not True
        and post_by_id[i].get("resolved") is True
    ]
    regressions = [
        i
        for i in pre_by_id
        if i in post_by_id
        and pre_by_id[i].get("resolved") is True
        and post_by_id[i].get("resolved") is not True
    ]

    # runner provenance via eval_note
    pre_notes = Counter(r.get("eval_note") for r in pre_sym)
    post_notes = Counter(r.get("eval_note") for r in post_sym)
    n_bintest = sum(1 for r in post_sym if r.get("eval_note") == BINTEST_NOTE)

    # ---- report ----
    print("=" * 72)
    print("Claim 10 -- sympy eval-bug recovery (Agentless oracle@k)")
    print("=" * 72)
    print(f"pre-fix  file : {PREFIX_FILE.name}")
    print(f"post-fix file : {postfix_file.name}")
    print()

    print("PRE-FIX (pytest bug runner):")
    print(f"  sympy rows           : {pre_n}        (paper: {TARGET_TOTAL})")
    print(f"  resolved             : {pre_resolved}/{pre_n}   "
          f"(paper: {TARGET_PREFIX_RESOLVED}/{TARGET_TOTAL})  "
          f"[{'MATCH' if pre_resolved == TARGET_PREFIX_RESOLVED and pre_n == TARGET_TOTAL else 'MISMATCH'}]")
    print(f"  eval_note tags       : {dict(pre_notes)}")
    print()

    print(f"POST-FIX (bin/test runner, eval_note='{BINTEST_NOTE}'):")
    print(f"  sympy rows           : {post_n}        (paper: {TARGET_TOTAL})")
    print(f"  resolved             : {post_resolved}/{post_n}  "
          f"(paper: {TARGET_POSTFIX_RESOLVED}/{TARGET_TOTAL})  "
          f"[{'MATCH' if post_resolved == TARGET_POSTFIX_RESOLVED and post_n == TARGET_TOTAL else 'MISMATCH'}]")
    print(f"  bin/test-scored rows : {n_bintest}")
    print(f"  eval_note tags       : {dict(post_notes)}")
    print()

    print("LIKE-FOR-LIKE INVARIANTS (recovery attributable to eval fix only):")
    print(f"  same sympy id set         : {same_id_set}")
    print(f"  byte-identical patches    : {identical_patch}/{pre_n}")
    print(f"  flips unresolved->resolved: {len(flips)}   (== post-fix resolved count: "
          f"{len(flips) == post_resolved})")
    print(f"  regressions resolved->unre: {len(regressions)}")
    print()

    print(f"RECOVERY: {pre_resolved}/{TARGET_TOTAL} -> {post_resolved}/{TARGET_TOTAL}   "
          f"(paper: {TARGET_PREFIX_RESOLVED}/{TARGET_TOTAL} -> "
          f"{TARGET_POSTFIX_RESOLVED}/{TARGET_TOTAL})")

    ok = (
        pre_n == TARGET_TOTAL
        and post_n == TARGET_TOTAL
        and pre_resolved == TARGET_PREFIX_RESOLVED
        and post_resolved == TARGET_POSTFIX_RESOLVED
        and same_id_set
        and identical_patch == pre_n
        and len(flips) == post_resolved
        and len(regressions) == 0
    )
    print()
    print("RESULT:", "PASS -- reproduces paper 0/77 -> 31/77" if ok
          else "FAIL -- see mismatches above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
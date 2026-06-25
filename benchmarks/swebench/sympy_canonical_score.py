#!/usr/bin/env python3
"""Offline scorer for the sympy canonical-scoring claim (paper Claim 10).

PAPER CLAIM (paper.tex):
  - Section "Evaluation correctness (sympy)" and "Workload reachability":
      sympy is scored under its canonical ``bin/test`` runner (validated by
      reference patches); the Agentless oracle@k baseline resolves 31/77.

WHAT THIS SCORER DOES (fully offline, no Docker / no API / no model):
  The committed scored file `h2h_agentless_sonnet_150_300_sympyfixed_*.jsonl`
  carries the per-instance `resolved` verdict produced by sympy's canonical
  ``bin/test`` runner (rows tagged eval_note == "docker(sympy)"). The scorer
  counts the resolved sympy rows and asserts the paper's 31/77 under that
  runner. As a light provenance check it confirms there are exactly 77 sympy
  instances and that every scored row carries the canonical-runner tag.

Run:  PYTHONNOUSERSITE=1 PYTHONPATH=src python3 benchmarks/swebench/sympy_canonical_score.py
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
TARGET_RESOLVED = 31  # "the Agentless baseline resolves 31/77"
TARGET_TOTAL = 77     # "all 77 sympy instances"

# The canonical bin/test scoring (glob the timestamp so the scorer is robust to it).
SCORE_GLOB = "h2h_agentless_sonnet_150_300_sympyfixed_*.jsonl"
# eval_note tags written by the canonical scorer: patched rows scored under the
# canonical bin/test runner ("canonical" or "docker(sympy)"), and rows that carried
# no patch.
CANONICAL_NOTES = {"canonical", "docker(sympy)", "no_patch (rescore)"}


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sympy_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["instance_id"].startswith("sympy__")]


def main() -> int:
    matches = sorted(glob.glob(str(EVALS / SCORE_GLOB)))
    if not matches:
        raise SystemExit(f"no scored file matching {SCORE_GLOB} under {EVALS}")
    score_file = Path(matches[-1])

    rows = sympy_rows(load(score_file))
    n = len(rows)
    resolved = sum(1 for r in rows if r.get("resolved") is True)
    n_canonical = sum(1 for r in rows if r.get("eval_note") in CANONICAL_NOTES)

    print("=" * 72)
    print("Claim 10 -- sympy canonical bin/test scoring (Agentless oracle@k)")
    print("=" * 72)
    print(f"scored file : {score_file.name}")
    print()
    print("canonical bin/test runner:")
    print(f"  sympy instances      : {n}        (paper: {TARGET_TOTAL})")
    print(f"  resolved             : {resolved}/{n}  "
          f"(paper: {TARGET_RESOLVED}/{TARGET_TOTAL})  "
          f"[{'MATCH' if resolved == TARGET_RESOLVED and n == TARGET_TOTAL else 'MISMATCH'}]")
    print(f"  canonically-scored   : {n_canonical}/{n}")
    print()

    ok = (n == TARGET_TOTAL and resolved == TARGET_RESOLVED and n_canonical == n)
    print("RESULT:", "PASS -- reproduces paper 31/77 under the canonical bin/test runner"
          if ok else "FAIL -- see mismatches above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

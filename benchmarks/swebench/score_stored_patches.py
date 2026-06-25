#!/usr/bin/env python3
"""Score stored SWE-bench patches offline with the runtime scorer (no model call).

WHY THIS EXISTS
  As long as an eval row archived the model's git diff, its verdict can be
  computed offline by applying that exact patch in the SWE-bench Docker
  container and grading it -- no model call, no API key, $0.

  This is why every runtime-produced eval row carries `model_patch` (see
  swe_bench_adapter.run_result_to_legacy_jsonl and the runtime RunResult.patch):
  a row that archived its patch can be graded offline from committed data alone.

WHAT IT DOES (no model, no API; Docker only)
  For each row that carries a non-empty patch field, apply the patch via the
  runtime's canonical docker_eval and record `resolved_scored` /
  `eval_note_scored`. By default only rows not already `resolved is True` are
  graded; pass --score-all to grade every row.

USAGE (Docker required; no API key needed):
  PYTHONNOUSERSITE=1 PYTHONPATH=src python3 benchmarks/swebench/score_stored_patches.py \
      --in evals/<cell>.jsonl --out evals/<cell>_SCORED.jsonl \
      [--patch-field model_patch] [--workers 4] [--score-all]

The patch field is auto-detected (model_patch, then h2h_patch) unless given.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from code_capsules.evaluation.docker_eval import docker_eval

_PATCH_FIELDS = ("model_patch", "h2h_patch", "patch")


def _load_instances() -> dict:
    """SWE-bench Lite test split, keyed by instance_id (FAIL/PASS lists decoded)."""
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    out = {}
    for row in ds:
        for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            val = row.get(key, [])
            if isinstance(val, str):
                row[key] = json.loads(val)
        out[row["instance_id"]] = row
    return out


def _detect_patch_field(rows: list[dict], override: str | None) -> str:
    if override:
        return override
    for f in _PATCH_FIELDS:
        if any((r.get(f) or "").strip() for r in rows):
            return f
    raise SystemExit(
        f"no patch field found (looked for {_PATCH_FIELDS}); rows carry only "
        "has_patch booleans and cannot be graded offline (no stored patch). "
        "Newer runs archive model_patch; see this file's docstring."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch-field", default=None,
                    help="patch field to score (auto: model_patch|h2h_patch|patch)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--score-all", dest="score_all", action="store_true",
                    help="also grade rows currently marked resolved (default "
                         "grades only rows not already resolved)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    pf = _detect_patch_field(rows, args.patch_field)
    insts = _load_instances()

    def needs(r):
        if not (r.get(pf) or "").strip():
            return False
        return args.score_all or r.get("resolved") is not True

    todo = [r for r in rows if needs(r)]
    print(f"rows={len(rows)}  patch_field={pf}  to_score={len(todo)}", flush=True)

    def work(r):
        iid = r["instance_id"]
        inst = insts.get(iid)
        if inst is None:
            return iid, None
        return iid, docker_eval(r.get(pf) or "", inst, timeout=args.timeout)

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for fut in as_completed(futs):
            iid, res = fut.result()
            if res is not None:
                results[iid] = res
                print(f"  {iid}  resolved={res.get('resolved')}  note={res.get('note')}",
                      flush=True)

    gained, lost = [], []
    for r in rows:
        iid = r["instance_id"]
        if iid in results:
            new = results[iid].get("resolved")
            old = r.get("resolved")
            r["resolved_scored"] = new
            r["eval_note_scored"] = results[iid].get("note")
            if new is True and old is not True:
                gained.append(iid)
            elif old is True and new is not True:
                lost.append(iid)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def eff(r):
        return r["resolved_scored"] if "resolved_scored" in r else r.get("resolved")

    old_n = sum(1 for r in rows if r.get("resolved") is True)
    new_n = sum(1 for r in rows if eff(r) is True)
    print(f"\nnewly resolved : {len(gained)}  {gained}", flush=True)
    if lost:
        print(f"no longer resolved : {len(lost)}  {lost}", flush=True)
    print(f"resolved: {old_n} -> {new_n} / {len(rows)}  (wrote {args.out})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

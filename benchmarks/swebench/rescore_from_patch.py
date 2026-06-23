#!/usr/bin/env python3
"""Model-free re-score of stored SWE-bench patches with the runtime scorer.

WHY THIS EXISTS
  A scorer change (bug fix, runner correction, harness upgrade) must NOT force
  an expensive online re-run of the models. As long as the eval row archived
  the model's git diff, the verdict can be recomputed offline by re-applying
  that exact patch in the SWE-bench Docker container and re-grading it.

  This is why every runtime-produced eval row carries `model_patch` (see
  swe_bench_adapter.run_result_to_legacy_jsonl and the runtime RunResult.patch):
  a row that archived its patch can be re-graded offline for $0, with no model
  call, so this utility is all you need to verify a score from committed data.

WHAT IT DOES (no model, no API; Docker only)
  For each row whose stored verdict is not `resolved is True` and which carries a
  non-empty patch field, re-apply the patch via the runtime's canonical
  docker_eval and record `resolved_rescored` / `eval_note_rescored`. Rows already
  `resolved is True` are kept by default; pass --rescore-all to re-grade every
  row regardless of its stored verdict.

USAGE (Docker required; no API key needed):
  PYTHONNOUSERSITE=1 PYTHONPATH=src python3 benchmarks/swebench/rescore_from_patch.py \
      --in evals/<cell>.jsonl --out evals/<cell>_RESCORED.jsonl \
      [--patch-field model_patch] [--workers 4] [--rescore-all]

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
        "has_patch booleans and cannot be re-scored offline -- an online re-run "
        "is required. Newer runs archive model_patch; see this file's docstring."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch-field", default=None,
                    help="patch field to re-score (auto: model_patch|h2h_patch|patch)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--rescore-all", action="store_true",
                    help="also re-score rows currently marked resolved (checks "
                         "for false-POSITIVE scorer changes; default re-scores "
                         "only non-resolved rows, the false-negative case)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    pf = _detect_patch_field(rows, args.patch_field)
    insts = _load_instances()

    def needs(r):
        if not (r.get(pf) or "").strip():
            return False
        return args.rescore_all or r.get("resolved") is not True

    todo = [r for r in rows if needs(r)]
    print(f"rows={len(rows)}  patch_field={pf}  to_rescore={len(todo)}", flush=True)

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

    flips, regressions = [], []
    for r in rows:
        iid = r["instance_id"]
        if iid in results:
            new = results[iid].get("resolved")
            old = r.get("resolved")
            r["resolved_rescored"] = new
            r["eval_note_rescored"] = results[iid].get("note")
            if new is True and old is not True:
                flips.append(iid)
            elif old is True and new is not True:
                regressions.append(iid)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def eff(r):
        return r["resolved_rescored"] if "resolved_rescored" in r else r.get("resolved")

    old_n = sum(1 for r in rows if r.get("resolved") is True)
    new_n = sum(1 for r in rows if eff(r) is True)
    print(f"\nflips false-neg -> resolved : {len(flips)}  {flips}", flush=True)
    if regressions:
        print(f"REGRESSIONS resolved -> not  : {len(regressions)}  {regressions}", flush=True)
    print(f"resolved: {old_n} -> {new_n} / {len(rows)}  (wrote {args.out})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

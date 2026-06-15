#!/usr/bin/env python3
"""Claim 9 scorer -- workload capability ceiling (oracle union).

Paper claim (paper.tex):
  L233-238: "capability ceiling on SWE-bench Lite first 150 is 103/150
  (68.7%), with 47 instances (31.3%) universally unreachable. The
  unreachable set is dominated by matplotlib (13/23, 57%) and django
  (33/114, 29%)".
  L2281-2287 (Section: Workload capability ceiling): "Taking the union
  of resolved instances across all four vendors evaluated (Anthropic
  Haiku 4.5, Sonnet 4.6, Opus 4.7, and OpenAI gpt-5-codex) and every
  shipped variant on every tier, the absolute workload capability
  ceiling is 103/150 (68.7%). That union spans 5 Opus cells, 10 Sonnet
  cells, 11 Haiku cells, and 3 codex cells (29 configurations in
  total). The complementary set (47 instances, 31.3%) is not resolved
  by any tested configuration on any vendor."
  L1470/1472: "across the five Opus cells (any one resolves) is
  101/150" and "shipped-variant configurations on four model families
  is 103/150 (68.7%)".

Method: per-instance ORACLE UNION over the committed offline eval
JSONLs for all four vendors (Haiku, Sonnet, Opus, gpt-5-codex) at
n=150. An instance counts as resolvable if ANY config on ANY vendor
has resolved == True. No Docker, no API, no model; computed by hand.

Notes on the file set:
  * The oracle UNION is monotone: extra files can only add to it,
    never remove. So the union is invariant to including the Sonnet
    variance replicates (implicit40_var2/var3, twopass_var2/var3) and
    the multiple codex two-pass runs -- they add no new resolved
    instance. We verify this explicitly (see the 29-cell vs file-set
    cross-check printed below); the result is 103/150 either way.
  * The aborted 3-row phaseStaged stub (...T034932) is excluded; the
    canonical 150-row run is ...T035438. Including the stub still
    yields 103/150 (it adds nothing), so the exclusion is hygiene, not
    a load-bearing filter.
"""
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals"

# Aborted 3-row stub; the canonical 150-row phaseStaged run is ...T035438.
STUB = str(EVALS / "p10_sonnet_phaseStaged_b35_n150_20260517T034932.jsonl")

SONNET = [f for f in sorted(glob.glob(str(EVALS / "p10_sonnet_*_n150_*.jsonl")))
          if f != STUB]
HAIKU = sorted(glob.glob(str(EVALS / "p10_haiku_*_n150_*.jsonl")))
OPUS = sorted(glob.glob(str(EVALS / "p9_opus_*_n150_combined_20260527.jsonl")))
CODEX = sorted(glob.glob(str(EVALS / "swe_codex_*_n150_*.jsonl")))
FOUR_FAMILY = SONNET + HAIKU + OPUS + CODEX

SONNET_SIGINJECT = str(EVALS / "p10_sonnet_siginject_b10x3_n150_20260517T035438.jsonl")


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def oracle_union(files):
    """Return (resolved_union, all_ids, repo_of) over the file set."""
    resolved = set()
    ids = set()
    repo_of = {}
    for f in files:
        for d in load(f):
            iid = d["instance_id"]
            ids.add(iid)
            repo = (d.get("repo") or iid.split("__")[0]).split("/")[0]
            repo_of[iid] = repo
            if d.get("resolved") is True:
                resolved.add(iid)
    return resolved, ids, repo_of


def repo_breakdown(ids, doomed, repo_of):
    tot = defaultdict(int)
    bad = defaultdict(int)
    for i in ids:
        tot[repo_of[i]] += 1
    for i in doomed:
        bad[repo_of[i]] += 1
    return tot, bad


def main():
    print("=" * 70)
    print("CLAIM 9 -- workload capability ceiling (oracle union)")
    print("Paper target: union 103/150 (68.7%); doomed 47 (31.3%);")
    print("              matplotlib 13/23 (57%); django 33/114 (29%);")
    print("              Opus alone 101/150")
    print("=" * 70)

    resolved, ids, repo_of = oracle_union(FOUR_FAMILY)
    doomed = ids - resolved
    u, n = len(resolved), len(ids)
    print(f"\n[FOUR-VENDOR ORACLE UNION] files={len(FOUR_FAMILY)} "
          f"(Sonnet {len(SONNET)} + Haiku {len(HAIKU)} + Opus {len(OPUS)} "
          f"+ codex {len(CODEX)})")
    print(f"  window={n}  union={u}/{n} ({u / n * 100:.1f}%)  "
          f"doomed={len(doomed)} ({len(doomed) / n * 100:.1f}%)")

    tot, bad = repo_breakdown(ids, doomed, repo_of)
    print("\n  Per-repo doomed breakdown:")
    for r in sorted(tot):
        pct = bad[r] / tot[r] * 100 if tot[r] else 0.0
        print(f"    {r:14s} doomed {bad[r]:3d}/{tot[r]:3d} ({pct:.0f}%)")
    print(f"  doomed sum check: {sum(bad.values())}  "
          f"(union {u} + doomed {len(doomed)} = {u + len(doomed)})")

    # Cross-check 1: Opus alone covers 101; the 2 it misses are the
    # Sonnet signal-injection niche (paper L1470, L2287).
    opus_res, _, _ = oracle_union(OPUS)
    opus_covered = len(opus_res & resolved)
    missed = sorted(resolved - opus_res)
    sig_res, _, _ = oracle_union([SONNET_SIGINJECT])
    print(f"\n  Opus alone covers {opus_covered}/{u} reachable instances.")
    print(f"  Reachable instances NOT resolved by any Opus cell: {missed}")
    for iid in missed:
        print(f"    {iid}: resolved by Sonnet signal-injection? "
              f"{iid in sig_res}")

    # Cross-check 2: union is invariant to the variance/extra-run files.
    sonnet_canon = [f for f in SONNET if "var2" not in f and "var3" not in f]
    codex_canon = [f for f in CODEX
                   if not (("twoPass" in f) and ("run3" not in f))]
    canon = sonnet_canon + HAIKU + OPUS + codex_canon
    res_canon, ids_canon, _ = oracle_union(canon)
    print(f"\n  29-cell (paper) union={len(res_canon)}/{len(ids_canon)}  "
          f"vs file-set union={u}/{n}  same set? {res_canon == resolved}")

    print("\n" + "=" * 70)
    print("MATCH CHECK (paper value vs computed)")
    print("=" * 70)
    mpl = bad["matplotlib"]
    dja = bad["django"]
    checks = [
        ("union", "103/150", f"{u}/{n}", u == 103 and n == 150),
        ("doomed", "47", f"{len(doomed)}", len(doomed) == 47),
        ("pct resolved", "68.7%", f"{u / n * 100:.1f}%",
         abs(u / n * 100 - 68.7) < 0.05),
        ("doomed pct", "31.3%", f"{len(doomed) / n * 100:.1f}%",
         abs(len(doomed) / n * 100 - 31.3) < 0.05),
        ("matplotlib doomed", "13/23", f"{mpl}/{tot['matplotlib']}",
         mpl == 13 and tot["matplotlib"] == 23),
        ("django doomed", "33/114", f"{dja}/{tot['django']}",
         dja == 33 and tot["django"] == 114),
        ("Opus alone", "101/150", f"{len(opus_res)}/{n}",
         len(opus_res) == 101),
    ]
    all_ok = True
    for name, paper_v, comp_v, ok in checks:
        flag = "MATCH" if ok else "MISMATCH"
        all_ok = all_ok and ok
        print(f"  {name:20s} paper={paper_v:10s} computed={comp_v:10s} [{flag}]")
    print("\nRESULT:", "ALL MATCH" if all_ok else "MISMATCH(ES) PRESENT")


if __name__ == "__main__":
    main()
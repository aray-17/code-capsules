#!/usr/bin/env python3
"""Build the evidence index that powers the web explorer's "Eval runs" catalog.

Scans the eval-run files (evals/*.jsonl, evals/leakfree/*.jsonl,
evals/heldout/*.jsonl, evals/scopeC/**/*.jsonl, evals/*.csv -- NOT the
per-instance stream archives under evals/streams/), computes a one-line
summary per file (rows, resolved fraction, mean cost, repositories/providers,
schema), gives each a human-readable label, and maps each file to the paper
claim(s) it backs by resolving the data globs in benchmarks/claims_results.json.
The result, benchmarks/explorer/evidence_index.json, lets a reader browse every
eval run in the browser, see what it is and which claim it supports, and drill
into the raw rows -- without poring over the files by hand.

Run (after verify_criteria.py, which writes claims_results.json):
    python3 benchmarks/build_evidence_index.py
"""
from __future__ import annotations

import csv
import glob
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_JSON = ROOT / "benchmarks" / "claims_results.json"
OUT = ROOT / "benchmarks" / "explorer" / "evidence_index.json"

SOURCES = ["evals/*.jsonl", "evals/leakfree/*.jsonl", "evals/heldout/*.jsonl",
           "evals/scopeC/*.jsonl", "evals/scopeC/menu/*.jsonl", "evals/scopeC/opus/*.jsonl",
           "evals/*.csv"]
SKIP = ("bak", "invalid", "contaminated", "ratelimited", "raw_with_errors",
        "credit_exhausted", "disk_corrupted", "preeval_bug", "dropped", ".preeval")

RESOLVED_KEYS = ("resolved", "shipped_gold_resolved", "gold_resolved", "canonical_resolved")
COST_KEYS = ("cost_usd", "lever_cost_usd")

# Filename-token -> readable phrase, scanned in groups so the label reads
# "<tier> . <config> . <benchmark/scope> . <experiment>".
TIER = {"sonnet": "Sonnet", "opus": "Opus", "haiku": "Haiku",
        "codex": "gpt-5-codex", "gpt5": "gpt-5-codex", "gemini": "Gemini"}
CONFIG = {"forcestage2": "two-pass critique (force-stage2)", "twopass": "two-pass critique",
          "siginject": "signal-injection", "p8c": "relevance ranker", "ranker": "relevance ranker",
          "planfirst": "plan-then-execute", "plan_first": "plan-then-execute",
          "toolalloc": "per-tool-class hint", "phase": "phase-staged",
          "implicit20": "implicit b=20", "implicit40": "implicit b=40",
          "signaled10": "signaled b=10", "floor100": "unbounded floor (b=100)",
          "unbounded": "unbounded budget", "floor": "unbounded floor", "lever": "diverse-pair lever",
          "implicit": "implicit budget", "signaled": "signaled budget"}
BENCH = {"humaneval": "HumanEval", "mbpp": "MBPP"}
EXPER = {"h2h": "head-to-head", "agentless": "Agentless baseline", "cross_vendor": "cross-vendor",
         "warmstart": "warm-start", "mixed_workload": "mixed-workload routing",
         "compound": "compound routing", "cc_sympy": "sympy re-run", "sympy": "sympy",
         "runtime_demo": "runtime demo", "e2e_runtime_smoke": "runtime e2e smoke",
         "controller_validation": "controller validation", "oracle": "oracle replay"}


def _is_skipped(name: str) -> bool:
    low = name.lower()
    return any(s in low for s in SKIP)


def _label(rel: str) -> str:
    name = Path(rel).name.lower()
    parts: list[str] = []

    def pick(table):
        for tok, phrase in table.items():
            if tok in name and phrase not in parts:
                parts.append(phrase)
                return

    pick(TIER)
    pick(CONFIG)
    pick(BENCH)
    pick(EXPER)
    # scope (sample size / split)
    for tok, phrase in (("second150", "held-out (second 150)"), ("150_300", "instances 150-300"),
                        ("first150", "first 150"), ("n300", "n=300"), ("n164", "n=164"),
                        ("n500", "n=500"), ("n150", "n=150"), ("n50", "n=50"), ("n30", "n=30")):
        if tok in name:
            parts.append(phrase)
            break
    return " . ".join(parts) if parts else Path(rel).stem


def _rows(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        rows = []
        for r in csv.DictReader(io.StringIO(path.read_text(errors="replace"))):
            d = dict(r)
            if "resolved" in d:
                d["resolved"] = str(d["resolved"]).strip().lower() in ("true", "1", "yes")
            for k in COST_KEYS:
                if d.get(k) not in (None, ""):
                    try:
                        d[k] = float(d[k])
                    except (TypeError, ValueError):
                        pass
            rows.append(d)
        return rows
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if s:
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return rows


def _summary(path: Path) -> dict:
    rows = _rows(path)
    n = len(rows)
    cols = sorted(rows[0].keys()) if rows else []

    rkey = next((k for k in RESOLVED_KEYS if any(k in r for r in rows)), None)
    resolved = None
    if rkey:
        resolved = sum(1 for r in rows if r.get(rkey) is True or str(r.get(rkey)).upper() == "RESOLVED")
    if resolved is None and any("outcome" in r for r in rows):
        rkey = "outcome"
        resolved = sum(1 for r in rows if str(r.get("outcome")).upper() == "RESOLVED")

    ckey = next((k for k in COST_KEYS if any(isinstance(r.get(k), (int, float)) for r in rows)), None)
    mean_cost = None
    if ckey:
        vals = [r[ckey] for r in rows if isinstance(r.get(ckey), (int, float))]
        mean_cost = round(sum(vals) / len(vals), 4) if vals else None

    repos = sorted({
        (r.get("repo") or r.get("provider") or str(r.get("instance_id", "")).split("__")[0])
        for r in rows if r.get("repo") or r.get("provider") or r.get("instance_id")
    } - {""})

    return {
        "n_rows": n,
        "resolved": resolved,
        "resolved_pct": round(100 * resolved / n, 1) if (resolved is not None and n) else None,
        "resolved_key": rkey,
        "mean_cost": mean_cost,
        "repos": repos,
        "columns": cols,
        "size_bytes": path.stat().st_size,
    }


def _claim_index() -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {}
    if not CLAIMS_JSON.exists():
        return idx
    for c in json.loads(CLAIMS_JSON.read_text()).get("claims", []):
        tag = {"id": c["id"], "title": c["title"], "section": c["section"]}
        for pattern in c.get("data", []):
            for f in glob.glob(str(ROOT / pattern)):
                idx.setdefault(str(Path(f).relative_to(ROOT)), []).append(tag)
    return idx


def main() -> int:
    claim_idx = _claim_index()
    files, seen = [], set()
    for pattern in SOURCES:
        for f in sorted(glob.glob(str(ROOT / pattern))):
            rel = str(Path(f).relative_to(ROOT))
            if rel in seen or _is_skipped(rel):
                continue
            seen.add(rel)
            entry = {"path": rel, "label": _label(rel), "format": Path(rel).suffix.lstrip(".")}
            entry.update(_summary(Path(f)))
            backing = claim_idx.get(rel, [])
            entry["claims"] = backing
            entry["kind"] = "claim-evidence" if backing else "supporting"
            files.append(entry)

    files.sort(key=lambda e: (e["kind"] != "claim-evidence", e["path"]))

    claims_meta = []
    if CLAIMS_JSON.exists():
        claims_meta = [{"id": c["id"], "title": c["title"], "section": c["section"], "status": c["status"]}
                       for c in json.loads(CLAIMS_JSON.read_text()).get("claims", [])]

    OUT.write_text(json.dumps({
        "note": "Eval-run catalog for the evidence explorer. Generated by "
                "benchmarks/build_evidence_index.py from evals/*.jsonl + "
                "evals/leakfree/*.jsonl + evals/heldout/*.jsonl + "
                "evals/scopeC/**/*.jsonl + evals/*.csv; backing claims from claims_results.json.",
        "n_files": len(files),
        "n_claim_evidence": sum(1 for e in files if e["kind"] == "claim-evidence"),
        "claims": claims_meta,
        "files": files,
    }, indent=1))
    backed = sum(1 for e in files if e["kind"] == "claim-evidence")
    print(f"indexed {len(files)} eval-run files ({backed} claim-evidence, "
          f"{len(files) - backed} supporting) -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

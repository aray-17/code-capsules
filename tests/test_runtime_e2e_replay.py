"""End-to-end runtime replay: the SHIPPED runtime reproduces the paper's governor
anchors over real logged eval data.

Unlike test_governor_replay.py (which feeds hand-built signal dicts into
CrossSampleAgreementCascade.decision in isolation), this drives the full public
entry point — CodeCapsulesRunner.run(sampler, grade_fn, regression_fn) ->
run_controller -> run_round -> select_by_verifier + agreement_reading + the
cascade — over the committed leak-free EXP-4 lever data, with the runtime
COMPUTING the agreement signals itself from per-candidate grades (not reading
them pre-baked). It proves the deployable runtime a user actually calls
reproduces the numbers in paper §"A regression-suite gate restores the wrong
abandons" (sec:agreement_regression_gate): first-150 (64, 70, 77) and held-out
(76, 77, 87) for (agreement, pure_regok, hybrid_regok).

Note on pure_regok: the runtime's `no_signal` band ships the no-reading set
(ship_fallback, governor-independent), which the scorer's simplified `pure`
formula (ship iff selected.regression_ok is True) omits. So the runtime's
default pure_regok is +2 on first-150 (72); with no_signal="abandon" it matches
the scorer's 70. hybrid_regok (the shipped default) and agreement match exactly
either way because their formulas already account for the fallback.
"""
import json
from pathlib import Path

import pytest

from code_capsules.controller.runtime import CodeCapsulesRunner, RunnerPolicy

ROOT = Path(__file__).resolve().parents[1]
# Exact paths (NOT globs): evals/leakfree/ also holds a quarantined
# *.RATELIMITED-MIXED.jsonl that a broad *.jsonl glob would pick up alongside
# the clean held-out files.
FIRST150 = ROOT / "evals/leakfree/exp4_lever_floor100_siginject.jsonl"
HELD_OUT = ROOT / "evals/leakfree/exp4_lever_second150.jsonl"


def _load(path):
    if not path.exists():
        pytest.skip(f"leak-free eval data not present: {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _governed_resolved(rows, governor, no_signal="ship_fallback"):
    """Drive the FULL runtime per instance over the logged candidates; count
    instances the runtime ships (outcome RESOLVED) that are gold-resolved."""
    pol = RunnerPolicy(
        configs=("floor", "siginject"), tiers=("default",), min_samples=2,
        governor=governor, ship_gate="repro_or_regok", no_signal=no_signal,
    )
    runner = CodeCapsulesRunner(pol)
    n = 0
    for r in rows:
        cands = r["candidates"]

        def sampler(config, tier, _c=cands, _id=r["instance_id"]):
            # real patch token per config; empty string when the config produced no patch
            return f"{_id}|{config}" if _c.get(config, {}).get("has_patch") else ""

        def _cfg(patch):
            return patch.split("|", 1)[1] if patch else None

        def grade_fn(patch, _c=cands):
            return _c.get(_cfg(patch), {}).get("grade", "NOPATCH")

        def regression_fn(patch, _c=cands):
            return _c.get(_cfg(patch), {}).get("regression_ok")

        run = runner.run(sampler, grade_fn, regression_fn)
        if run.outcome == "RESOLVED" and r.get("shipped_gold_resolved"):
            n += 1
    return n


def test_full_runtime_reproduces_default_governor_anchors():
    """hybrid_regok (the shipped default) and agreement reproduce the paper
    anchors EXACTLY through the full CodeCapsulesRunner.run path, both splits."""
    first, held = _load(FIRST150), _load(HELD_OUT)
    assert len(first) == 150 and len(held) == 150
    # hybrid_regok = the shipped default governor (RunnerPolicy default)
    assert _governed_resolved(first, "hybrid_regok") == 77
    assert _governed_resolved(held, "hybrid_regok") == 87
    # agreement (repro-only) governor
    assert _governed_resolved(first, "agreement") == 64
    assert _governed_resolved(held, "agreement") == 76


def test_full_runtime_reproduces_pure_regok_anchor():
    """pure_regok matches the paper anchor (70 first / 77 held) when the
    no-reading band is modeled like the scorer (no_signal='abandon')."""
    first, held = _load(FIRST150), _load(HELD_OUT)
    assert _governed_resolved(first, "pure_regok", no_signal="abandon") == 70
    assert _governed_resolved(held, "pure_regok", no_signal="abandon") == 77


def test_no_signal_ship_fallback_ships_the_no_reading_band():
    """Documents the runtime's default no_signal behavior: ship_fallback ships
    +2 gold-resolved no-reading instances on first-150 that the abandon band
    declines (72 vs 70). Governor-independent; the deployable default."""
    first = _load(FIRST150)
    assert _governed_resolved(first, "pure_regok", no_signal="ship_fallback") == 72
    assert _governed_resolved(first, "pure_regok", no_signal="abandon") == 70

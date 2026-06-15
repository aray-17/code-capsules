"""Flagship example: the DEPLOYABLE LEVER.

The lever is the framework's three-component cost/quality controller:

    SOLVERS        a diverse pair of execution variants generate candidate patches
    REPRO-VERIFIER a deployable, NON-GOLD verifier (a generated reproduction +
                   a regression check) selects the resolver among the candidates
    GOVERNOR       when the diverse configs AGREE on failure, the instance is
                   doomed at this tier -> ABANDON (don't grind), else SELECT the winner

You compose it with two callables passed to `runner.run(sampler, grade_fn)`:
    sampler(config, tier) -> patch     run a named solver at a model tier
    grade_fn(patch)       -> grade     the DEPLOYABLE verifier (never a gold gate)

The shipped `policy.yaml` controller block supplies the solver pair, the tier
ladder, and the governor's abandon-vs-escalate default.

Run this file for a no-Docker MOCK smoke demo:  python examples/deployable_lever.py
The production sketch below shows the real (Docker-backed) wiring.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from code_capsules import CodeCapsulesRunner                       # the lever
from code_capsules.controller.verifier import RESOLVED, FLAT, NOPATCH

POLICY = str(Path(__file__).parent.parent / "policy.yaml")


# ── 1. MOCK smoke demo (runs anywhere; no Docker, no model) ──────────────────
def mock_demo() -> None:
    runner = CodeCapsulesRunner.from_policy_file(POLICY)   # solver pair + governor

    # A sampler that returns a canned patch per solver config.
    patches = {"floor": "patch-from-floor", "siginject": "patch-from-siginject"}
    sampler = lambda config, tier: patches.get(config, "")

    # A DEPLOYABLE grade_fn: a PURE function of the patch (a stand-in for
    # "run the generated reproduction + regression check"). NEVER reads gold.
    repro_passes = {"patch-from-siginject"}               # pretend only this one fixes it
    def grade_fn(patch: str) -> str:
        if not patch.strip():
            return NOPATCH
        return RESOLVED if patch in repro_passes else FLAT

    run = runner.run(sampler, grade_fn)
    print(f"[resolve case]  outcome={run.outcome}  tier={run.final_tier}  "
          f"selection={getattr(run.selection, 'config', None)}")

    # Now make the verifier reject both -> the governor abandons.
    run2 = runner.run(sampler, lambda p: FLAT)
    print(f"[abandon case]  outcome={run2.outcome}  (diverse configs agreed on failure)")


# ── 2. PRODUCTION sketch (real wiring; needs Docker + the SWE-bench harness) ──
def production_sketch() -> None:
    """How the lever is wired for real. Not executed by default (needs Docker).

    The SWE-bench runtime adapters, repo cache, and Docker-backed reproduction
    verifier live in the operational harness (not distributed; see CLAIMS.md
    "Operational data not in this repository"). Provide your own equivalents:
    """
    # from your_harness import make_swebench_adapters   # (instance, cache) -> (sampler, gold_grade)
    # from your_harness import RepoCacheManager
    # from your_harness import repro_verifier as rv      # generates fail-on-base reproductions
    #
    # cache = RepoCacheManager()
    # for instance in instances:
    #     sampler, _gold = make_swebench_adapters(instance, cache)   # gold grade is SCORING-ONLY
    #
    #     # The DEPLOYABLE grade_fn: generate a fail-on-base reproduction, run it on
    #     # the candidate patch, and require the repo's existing tests to stay green.
    #     # This reads NO held-out/gold verdict -- it is what a deployment actually has.
    #     repros = rv.validated_repros(instance, k=4)
    #     def grade_fn(patch, _repros=repros, _inst=instance):
    #         if not patch.strip():
    #             return NOPATCH
    #         passed = all(rv._run_repro(_inst, patch, code)[0] == rv.PASS for code in _repros)
    #         reg_ok, _ = rv._run_regression(_inst, patch)
    #         return RESOLVED if (_repros and passed and reg_ok is not False) else FLAT
    #
    #     run = runner.run(sampler, grade_fn)         # SELECT on success, ABANDON on agreement-fail
    #     # Score the final outcome with gold docker_eval SEPARATELY (scoring only).
    raise SystemExit("production_sketch is documentation; see the commented wiring above.")


if __name__ == "__main__":
    mock_demo()

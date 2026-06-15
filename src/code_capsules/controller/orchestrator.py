"""Runtime orchestrator: composes the validated controller primitives into the
adaptive-compute loop, independent of any benchmark.

Until now the decision primitives (select_by_verifier, agreement_reading,
CrossSampleAgreementCascade, VerifierGate, the VOI formula) lived in the runtime
but the loop that USES them lived entirely in the harness (tools/run_swebench_*).
This module is that loop, so the runtime can run the controller given two injected
adapters:

  sampler(config, tier) -> patch     run one diverse config at a model tier
  grade_fn(patch)       -> grade     the deployable verifier grade (e.g. repro+
                                     regression); the SAME grade drives BOTH levers

The validated VOI policy (generalizability eval 2026-06-03):
  diverse-sample K configs -> grade each ONCE ->
    select_by_verifier  (QUALITY lever: keep the best candidate)
    agreement_reading   (COST lever: did the diverse samples AGREE on failure?)
  -> CrossSampleAgreementCascade.decision():
       STOP (a sample resolved) / GATHER_SAMPLES (K too small to trust doom) /
       ABANDON (within-tier doomed -- the validated, economical action) /
       ESCALATE_TIER (opt-in only; cross-model escalation is marginal -- see the
                      cascade docstring + the Phase 8/9 negative).

Benchmark specifics (docker, repro generation, SWE-bench scoring) stay in the
sampler/grade_fn adapters; this module owns only the policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from code_capsules.controller.selector import (
    Candidate, Selection, select_by_verifier, agreement_reading,
)
from code_capsules.controller.cascade_triggers import CrossSampleAgreementCascade
from code_capsules.controller.escalation_policy import (
    EscalationStage, stage_trigger_fires,
)
from code_capsules.controller.verifier import NOPATCH


@dataclass(frozen=True)
class ControllerResult:
    decision: str                      # STOP / GATHER_SAMPLES / ABANDON / ESCALATE_TIER
    selection: Optional[Selection]     # best candidate (quality lever); None if no candidates
    tier: int
    n_samples: int
    n_resolved: int
    resolved: bool                     # a diverse sample graded RESOLVED
    signals: Optional[dict] = None     # the signals dict the cascade decided on
                                       # (agreement counts + failure_kind + the
                                       # regression channel) -- the escalation
                                       # ladder's triggers read it


def run_round(
    configs: Sequence[str],
    sampler: Callable[[str, int], str],
    grade_fn: Callable[[str], str],
    *,
    tier: int = 1,
    cascade: Optional[CrossSampleAgreementCascade] = None,
    regression_fn: Optional[Callable[[str], Optional[bool]]] = None,
) -> ControllerResult:
    """One controller round at `tier`: diverse-sample the `configs`, grade each
    candidate ONCE (grading is the expensive op -- e.g. docker), then read both
    levers off the shared grades and return the cascade decision. The caller acts
    on `.decision` (ship `.selection`, gather more, abandon, or escalate tier).

    regression_fn(patch) -> bool|None is the OPTIONAL regression channel (e.g.
    make_repro_verifier(...).regression): the round threads the SELECTED
    candidate's verdict (`selected_regression_ok`) plus the count over all
    candidates (`n_regression_ok`) into the cascade signals, where the shipped
    hybrid_regok governor reads them. Run once per distinct patch; empty patches
    are never run (verdict None). Absent -> signals carry None/0 and the cascade
    behaves exactly as before."""
    cascade = cascade or CrossSampleAgreementCascade()
    cands = [Candidate(c, sampler(c, tier)) for c in configs]

    # grade each distinct patch ONCE; both levers read the same grades
    _cache: dict[str, str] = {}

    def graded(patch: str) -> str:
        if patch not in _cache:
            _cache[patch] = grade_fn(patch)
        return _cache[patch]

    # regression channel: run each distinct non-empty patch ONCE (it is as
    # expensive as grading); None-safe when no regression_fn is wired
    _reg_cache: dict[str, Optional[bool]] = {}

    def regression(patch: str) -> Optional[bool]:
        if regression_fn is None or not (patch or "").strip():
            return None
        if patch not in _reg_cache:
            _reg_cache[patch] = regression_fn(patch)
        return _reg_cache[patch]

    sel = select_by_verifier(cands, graded)
    sig = agreement_reading(cands, graded)
    sig["selected_regression_ok"] = regression(sel.patch) if sel is not None else None
    sig["n_regression_ok"] = sum(1 for c in cands if regression(c.patch) is True)
    decision = cascade.decision(sig, current_tier=tier)
    return ControllerResult(
        decision=decision, selection=sel, tier=tier,
        n_samples=sig["n_samples"], n_resolved=sig["n_resolved"],
        resolved=(sig["n_resolved"] > 0), signals=sig,
    )


def decide_escalation(
    *, resolved: bool, has_patch: bool, n_fail_to_pass: int,
    eval_result: dict, session, signals,
    start_budget: int, target_budget: int, cap_pressure_threshold: float,
    verifier_gate=None, always_escalate: bool = False, force_stage2: bool = False,
):
    """The escalation-decision dispatch (Phase 11: moved from the SWE-bench harness
    into the framework). Four modes, each returning a framework EscalationDecision:

      force_stage2    -> deployment-realistic two-pass: run stage 2 UNCONDITIONALLY,
                         never consulting the held-out FAIL_TO_PASS outcome. Used to
                         measure two-pass cost/quality without the in-loop oracle that
                         always_escalate and the verifier RESOLVED grade both read.
      verifier_gate   -> execution-verifier grade gate (stop RESOLVED, abandon the
                         confident-doomed set, escalate the ambiguous middle) -- EXP-1.
                         NB: the SWE-bench verifier shim grades RESOLVED from the gold
                         `resolved` field, so its stop-for-success decision is NOT
                         oracle-free; use force_stage2 for the leakage-free measurement.
      always_escalate -> two-pass critique (escalate unless stage-1 already resolved)
      else            -> the default heuristic should_escalate (cap-pressure + signals)

    The individual deciders (should_escalate, VerifierGate) already live in the
    framework; this is the dispatch that selects among them -- now framework-owned
    too, so the harness only supplies the SWE-bench inputs. Logic reproduced exactly
    from the previous harness inline block (characterized by test_escalation_parity)."""
    from code_capsules.controller.escalation import EscalationDecision, should_escalate
    if force_stage2:
        return EscalationDecision(
            escalate=True,
            reason="force_stage2 (always run both passes; no in-loop oracle)",
            next_budget=target_budget, next_prompt_budget_hint=target_budget)
    if verifier_gate is not None:
        from code_capsules.controller.verifier import from_swebench_eval
        from code_capsules.controller.cascade_triggers import VerifierGate
        grade = from_swebench_eval({**eval_result, "has_patch": has_patch}, n_fail_to_pass)
        esc = VerifierGate(abandon=tuple(verifier_gate)).should_escalate(
            {"resolved": bool(resolved), "verifier_grade": grade}, current_tier=1)
        return EscalationDecision(
            escalate=esc, reason=f"verifier_gate(grade={grade}, escalate={esc})",
            next_budget=target_budget if esc else None,
            next_prompt_budget_hint=target_budget if esc else None)
    if always_escalate and not resolved:
        return EscalationDecision(
            escalate=True, reason="always_escalate (V1 two-pass critique)",
            next_budget=target_budget, next_prompt_budget_hint=target_budget)
    return should_escalate(
        resolved=bool(resolved), actual_turns=session.num_turns,
        current_budget=start_budget, signals=signals,
        target_budget=target_budget, cap_pressure_threshold=cap_pressure_threshold)


@dataclass(frozen=True)
class TierStep:
    tier_index: int          # 1-based position in the tier ladder
    tier: str                # the tier id (e.g. "sonnet", "opus")
    decision: str
    n_samples: int
    n_resolved: int


@dataclass(frozen=True)
class ControllerRun:
    outcome: str             # RESOLVED / ABANDONED / EXHAUSTED
    final_tier: str
    selection: Optional[Selection]   # best candidate produced -- carried on EVERY
                                     # outcome (incl. ABANDONED/EXHAUSTED) so the
                                     # caller can always LOG/score the patch; the
                                     # `outcome` field (not None-ness) is what says
                                     # whether to SHIP it. None only if no config
                                     # produced any candidate.
    steps: tuple             # tuple[TierStep] -- the per-tier trajectory
    decline_reason: Optional[str] = None
                             # WHY a decline-to-ship declined (outcome ABANDONED):
                             #   trigger_not_fired -- a stage remained but its
                             #       escalation trigger did not fire
                             #   top_of_ladder     -- doomed at the last rung of a
                             #       multi-stage ladder (nowhere left to escalate)
                             #   declined_doomed   -- the single-stage doom call
                             #       (no escalation ladder configured at all)
                             #   no_signal_abandon -- the round had NO usable
                             #       execution reading and the cascade's
                             #       no_signal policy is 'abandon' (or
                             #       'escalate' at the top of the ladder):
                             #       never-ship-unverified, NOT a doom call
                             # None on every other outcome.


def _as_stages(
    tiers: Optional[Sequence[str]],
    configs: Optional[Sequence[str]],
    stages: Optional[Sequence[EscalationStage]],
    cascade: Optional[CrossSampleAgreementCascade],
) -> tuple:
    """Normalize the two run_controller signatures onto one stage ladder.

    Back-compat conversion (tiers= + configs=): every tier runs the SAME
    configs, and the trigger comes from the cascade's escalate_on_agreement
    ("always" when True -- the old unconditional climb; "never" when False --
    the old abandon-at-tier-1)."""
    if stages is not None:
        if tiers is not None or configs is not None:
            raise ValueError("pass either stages= or tiers=+configs=, not both")
        return tuple(stages)
    if tiers is None or configs is None:
        raise ValueError("run_controller needs tiers=+configs= or stages=")
    esc = getattr(cascade, "escalate_on_agreement", True) if cascade is not None else True
    trigger = "always" if esc else "never"
    return tuple(EscalationStage(tier=t, configs=tuple(configs), trigger=trigger)
                 for t in tiers)


def run_controller(
    tiers: Optional[Sequence[str]] = None,
    configs: Optional[Sequence[str]] = None,
    sampler: Callable[[str, str], str] = None,
    grade_fn: Callable[[str], str] = None,
    *,
    cascade: Optional[CrossSampleAgreementCascade] = None,
    regression_fn: Optional[Callable[[str], Optional[bool]]] = None,
    stages: Optional[Sequence[EscalationStage]] = None,
    p_source=None,
) -> ControllerRun:
    """Multi-stage VOI loop that HONORS ESCALATE_TIER by routing to a stronger model.

    Two signatures:
      run_controller(tiers, configs, sampler, grade_fn)   -- back-compat: every
          tier runs the SAME configs; escalation is unconditional ("always")
          when the cascade escalates on agreement (the default here), "never"
          otherwise. Behavior is unchanged from the pre-ladder loop.
      run_controller(sampler=..., grade_fn=..., stages=[EscalationStage(...)])
          -- the escalation LADDER: per-stage configs + per-stage triggers
          (escalation_policy.TRIGGERS). E.g. the validated destination shapes:
          GOV-V2 escalate-regok=1-to-Opus = 84/150 @ $1.60/res (Opus recoveries
          20/21 in regok=1, 3/17 in regok=0):
              stages=[EscalationStage("sonnet", ("floor", "siginject")),
                      EscalationStage("opus", ("floor",), trigger="regok_true")]
          cascade signaled10->opus_floor on no-patch = 86 @ $0.805
          (django-concentrated caveat; single-config tier-1 needs a
          min_samples=1 cascade so K=1 forms the failure call):
              stages=[EscalationStage("s10", ("signaled10",)),
                      EscalationStage("opus", ("floor",), trigger="has_patch")]

    `sampler(config, tier)` runs one diverse config at a model tier. At each
    stage run_round() reads both levers off one grading pass:
      STOP           -> ship the selection -> outcome RESOLVED (a sample passed
                        its repros, the governor rescued on the regression
                        channel, or the no-signal ship-fallback fired)
      GATHER_SAMPLES -> K<min_samples; the stage's full ensemble already ran, so
                        treat as exhausted-at-stage -> advance to the next stage
      ABANDON / ESCALATE_TIER (the agreement-failure pair) -> the ladder
          arbitrates:
            failure_kind == "no_signal" and a stage remains -> ESCALATE
                (no usable execution reading is NEVER a doom call -- the
                default policy, independent of the next stage's trigger)
            the NEXT stage's trigger fires on the signals -> ESCALATE
            otherwise -> the governor takes its terminal decision with no tier
                left to escape to: hybrid_regok may still rescue (ship), the
                no-signal band ship-falls-back at the top of the ladder
                (mirrors the EXP-4 harness NO_REPRO_FALLBACK), else outcome
                ABANDONED with `decline_reason` recorded (trigger_not_fired /
                top_of_ladder / declined_doomed -- see ControllerRun).
    Out of stages -> EXHAUSTED (no doom call was ever formed; carry the patch).

    For cost-optimal deployment the validated finding says tier-escalation is
    MARGINAL (~$2.42/resolve vs $1.40 base) UNLESS gate-positive: escalate
    regok=1 abandons (GOV-V2 above) or the no-patch case (the signaled10
    cascade) -- the only settings where a stronger tier earned its cost.
    Caveat (G4): regok=0 is NOT doom off-django (P(gold|regok=0)=0.37
    held-out); a non-fired regok_true trigger declines with a recorded reason,
    it never asserts confirmed doom.

    regression_fn(patch) -> bool|None is the optional regression channel,
    forwarded to every run_round so the cascade's hybrid_regok governor and the
    regok_true trigger can read it (see run_round / CrossSampleAgreementCascade).

    p_source: optional pluggable VOI belief (value_of_information.PSource),
    forwarded to the ladder's `voi` triggers (see stage_trigger_fires). None ->
    the EXP-1 per-grade belief table (prior behavior). Ships disabled at the
    policy surface (`p_source: null`).

    The cascade's `no_signal` knob is honored ladder-wide: 'ship_fallback'
    (default) climbs/ships as described above; 'abandon' declines the
    no-reading band without climbing; 'escalate' climbs but declines (instead
    of shipping) at the top of the ladder -- declines on this band record
    decline_reason='no_signal_abandon' (never a doom call)."""
    ladder = _as_stages(tiers, configs, stages, cascade)
    cascade = cascade or CrossSampleAgreementCascade(
        max_tier=len(ladder), escalate_on_agreement=True)
    n_stages = len(ladder)
    steps = []
    last_selection: Optional[Selection] = None
    for idx, stage in enumerate(ladder, start=1):
        # run_round passes its integer `tier` (the cascade's current_tier) to the
        # sampler; wrap so the sampler instead receives this stage's STRING tier id.
        res = run_round(stage.configs, lambda c, _idx, _t=stage.tier: sampler(c, _t),
                        grade_fn, tier=idx, cascade=cascade,
                        regression_fn=regression_fn)
        last_selection = res.selection      # keep the best candidate seen, any outcome
        steps.append(TierStep(tier_index=idx, tier=stage.tier, decision=res.decision,
                              n_samples=res.n_samples, n_resolved=res.n_resolved))
        if res.decision == cascade.STOP:
            return ControllerRun("RESOLVED", stage.tier, res.selection, tuple(steps))
        if res.decision in (cascade.ABANDON, cascade.ESCALATE_TIER):
            sig = res.signals or {}
            remaining = idx < n_stages
            no_signal_round = sig.get("failure_kind") == "no_signal"
            if remaining:
                # No usable execution reading is never a doom call: climb,
                # regardless of the next stage's trigger (the default policy
                # of the APPROVED no_signal split) -- UNLESS the cascade's
                # no_signal policy is 'abandon' (never spend escalation on
                # the no-reading band; fall to terminal arbitration).
                if no_signal_round:
                    if getattr(cascade, "no_signal", "ship_fallback") != "abandon":
                        continue
                elif stage_trigger_fires(ladder[idx], sig, res.selection,
                                         p_source=p_source):
                    continue
            # Terminal arbitration: the governor's final word with no tier left
            # to escape to (current_tier=max_tier disables the escalate branch,
            # so hybrid_regok's rescue and the no-signal ship-fallback apply).
            final = cascade.decision(sig, current_tier=cascade.max_tier)
            if final == cascade.STOP:
                return ControllerRun("RESOLVED", stage.tier, res.selection, tuple(steps))
            # Decline-to-ship, with the reason recorded. Carry the selection:
            # the governor says DON'T SHIP (outcome=ABANDONED), but the caller
            # still needs the best failed patch to log/score it. The SHIP
            # decision keys on `outcome`, never on selection being None.
            if no_signal_round:
                reason = "no_signal_abandon"     # policy decline, NOT a doom call
            elif remaining:
                reason = "trigger_not_fired"
            elif n_stages > 1:
                reason = "top_of_ladder"
            else:
                reason = "declined_doomed"
            return ControllerRun("ABANDONED", stage.tier, res.selection, tuple(steps),
                                 decline_reason=reason)
        # GATHER_SAMPLES -> evidence too weak for any call: climb unconditionally
    # Ladder exhausted (incl. the degenerate single-config GATHER_SAMPLES case, which
    # is a plain "run one variant" with no doom call): carry the last selection so the
    # patch is never lost.
    return ControllerRun("EXHAUSTED", ladder[-1].tier if ladder else "",
                         last_selection, tuple(steps))


__all__ = ["ControllerResult", "run_round", "TierStep", "ControllerRun", "run_controller",
           "EscalationStage", "decide_escalation"]

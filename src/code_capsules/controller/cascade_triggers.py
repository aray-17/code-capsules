"""
Built-in CascadeTrigger implementations.

Three shipped defaults:

- HeuristicCascade: the original signal-based logic. Escalates iff cap pressure
  ≥ threshold AND any quality signal fired. Doesn't escalate if the
  model self-terminated below the cap (clean give-up).

- AlwaysEscalate: forces escalation when not resolved. Used by the two-pass
  critique variant (always run stage 2). Bypasses heuristic.

- NeverEscalate: forces no escalation. Used to A/B against AlwaysEscalate
  or to disable cascade entirely for cost-bounded deployments.

User-defined triggers conform to the CascadeTrigger Protocol from
controller.api.
"""
from __future__ import annotations

from typing import Any


class HeuristicCascade:
    """The original signal-based escalation logic.

    Escalation decision tree:
      1. If signals contains 'resolved' True → don't escalate
      2. If cap_pressure < threshold → don't escalate
         (model self-terminated; extra turns unlikely to help)
      3. If any signal fired (file_thrash / test_failure / traceback /
         patch_attempt_failed) → escalate
      4. Else clean give-up → don't escalate

    Threshold default 0.7 (must have used 70% of budget to be eligible).
    """

    name = "heuristic"

    def __init__(self, cap_pressure_threshold: float = 0.7):
        self.cap_pressure_threshold = cap_pressure_threshold

    def should_escalate(self, signals: dict[str, Any], current_tier: int) -> bool:
        if signals.get("resolved"):
            return False
        cap = signals.get("cap_pressure", 0.0)
        if isinstance(cap, (int, float)) and cap < self.cap_pressure_threshold:
            return False
        # Any quality signal fired?
        for key in ("file_thrash", "test_failure", "traceback", "patch_attempt_failed"):
            if signals.get(key):
                return True
        return False


class AlwaysEscalate:
    """Forces escalation when not resolved. Tier-aware: only fires while
    current_tier < some max (default 2, i.e. one escalation allowed)."""

    name = "always_escalate"

    def __init__(self, max_tier: int = 2):
        self.max_tier = max_tier

    def should_escalate(self, signals: dict[str, Any], current_tier: int) -> bool:
        if signals.get("resolved"):
            return False
        if current_tier >= self.max_tier:
            return False
        return True


class NeverEscalate:
    """Never escalates. Useful for cost-bounded deployments or A/B controls."""

    name = "never_escalate"

    def should_escalate(self, signals: dict[str, Any], current_tier: int) -> bool:
        return False


class VerifierGate:
    """Execution-verifier-grounded escalation gate (validated 2026-06-01).

    Unlike HeuristicCascade (which keys on self-reported quality signals that
    a controlled study showed do NOT separate winnable from doomed), this gate keys on the
    *execution* verifier grade (controller.verifier), which separates the
    confident tails cleanly:

      RESOLVED -> stop-for-success (already passes; escalation wastes spend)
      FLAT     -> confident doomed (ran, nothing moved) -> drop
      NOPATCH / PARTIAL / BROKEN / UNKNOWN -> ambiguous middle -> ESCALATE

    The calibration study (n=50 django): dropping only FLAT and stopping
    RESOLVED kept 17/17 resolves (ZERO lost) while cutting escalations. The naive symmetric gate
    (also dropping NOPATCH/BROKEN) is catastrophic because NOPATCH at a low
    floor budget is 43% winnable ("no patch yet" = needs more turns). So this
    gate acts ONLY on the tails and defaults to escalate on the middle.

    `abandon` is the confident-doomed set to drop. Default {"FLAT"} is the
    zero-resolve-loss point; {"FLAT", "BROKEN"} saves more at ~1 lost resolve.
    Reads signals["verifier_grade"] (see controller.verifier); if
    absent, falls back to escalate-when-unresolved (AlwaysEscalate behavior),
    so it degrades safely when no execution reading is available.
    """

    name = "verifier_gate"

    def __init__(self, abandon: tuple[str, ...] = ("FLAT",), max_tier: int = 2):
        from code_capsules.controller.verifier import RESOLVED
        self._RESOLVED = RESOLVED
        self.abandon = set(abandon)
        self.max_tier = max_tier

    def should_escalate(self, signals: dict[str, Any], current_tier: int) -> bool:
        grade = signals.get("verifier_grade")
        # Already done (by the success criterion or a RESOLVED reading): stop.
        if signals.get("resolved") or grade == self._RESOLVED:
            return False
        if current_tier >= self.max_tier:
            return False
        # Confident-doomed tail: drop (don't spend stage-2).
        if grade in self.abandon:
            return False
        # No reading available -> safe fallback: escalate-when-unresolved.
        # Ambiguous middle (NOPATCH/PARTIAL/BROKEN/UNKNOWN) -> escalate.
        return True


class CrossSampleAgreementCascade:
    """Cross-sample diverse-agreement cascade: the validated VOI cost lever
    (generalizability eval 2026-06-03).

    Run K DIVERSE configs at the current tier, then key on their *agreement*:

      any sample resolved        -> STOP            (verifier-select already has a
                                                     winner; the quality lever's output)
      K < min_samples, all failed -> GATHER_SAMPLES  (a single failure is only ~85%
                                                     precise; run one more cheap
                                                     same-tier diverse config to reach
                                                     the ~98% K>=2 agreement, don't
                                                     abandon on weak evidence)
      K >= min_samples, all failed -> ABANDON         (the within-tier DOOMED signal;
                                                     the validated, economical action:
                                                     stop spending -- see below)
                                  OR -> ESCALATE_TIER  (ONLY if opt-in escalate_on_agreement
                                                     AND a stronger tier remains)

    Empirical basis (Sonnet first-150 n=150): when 2 diverse configs (floor +
    siginject) both fail at b10, the instance is doomed at ~98% in-family precision
    and **0% recovered by a different same-tier harness** (Agentless) -- so more
    same-tier compute, in any form, does not help. ABANDON is therefore the
    validated win, and it replicates cross-model (Sonnet floor+siginject 98%,
    Haiku floor+implicit40 97%; the partner config is model-specific).

    ESCALATE_TIER is OFF by default and carries a standing caveat: a *stronger
    model* (Opus) recovered only ~18% of the doomed set, at ~$2.42/resolve = ~1.7x
    the base $1.40/resolve rate. This matches the project's PRIOR negative finding
    (the cross-model cost-cascade was uneconomical, "+18% cost for +1
    pass"). So tier-escalation is a marginal, more-expensive tail -- enable it
    (escalate_on_agreement=True) ONLY when a resolve is worth ~1.7x the base cost.
    When enabled, the escalation axis is the MODEL TIER, not the turn budget
    (escalation_kind="tier"); a budget-only harness must NOT wire it (it would spend
    the exact same-tier compute the eval showed is wasted).

    GOVERNOR (the ship/abandon gate; n=300 replay, 2026-06-10): the
    regression channel (signals['selected_regression_ok'], threaded by
    orchestrator.run_round from the verifier's .regression) rescues would-be
    abandons. Three modes:

      agreement    -> the original behavior above (repro agreement only)
      hybrid_regok -> DEFAULT. Before abandoning, if the SELECTED candidate's
                      regression channel is True, STOP and ship it instead --
                      the no-silent-abandon fix. Replay: first-150 64 -> 77
                      (= floor parity); held-out 76 -> 87 vs floor 90 (-3): it
                      recovers 11/14 wrong-abandons, NOT 14/14 -- ship it as the
                      strict improvement at zero added cost, claim no parity.
      pure_regok   -> ship iff selected_regression_ok is True, regardless of the
                      repro outcome (replay: 70 / 77 -- dominated by hybrid).

    regok=0 is NOT doom off-django (P(gold|regok=0)=0.37 held-out): no mode here
    treats a False regression verdict as abandon-evidence, and None (no verdict)
    never rescues. Any regok=0=>abandon policy must be re-justified per-domain.

    SHIP GATE (precision knob, OFF by default): ship_gate='repro_and_regok'
    additionally requires selected_regression_ok is not False for a repro-pass
    STOP, else the instance demotes to the escalate/decline path. It buys SHIP
    precision at a resolve cost (replay: first-150 58->78% at -4 resolves,
    held-out 67->77% at -8) -- an auto-merge confidence label, not a default.

    NO_SIGNAL BAND (the APPROVED zero-resolved split, sign-off 2026-06-10):
    when signals['failure_kind'] == 'no_signal' (selector.agreement_reading:
    every non-RESOLVED grade is UNKNOWN/NOPATCH -- no usable execution
    reading), the zero-resolved round is NOT treated as doomed. The
    `no_signal` knob (plan §4 item 6 policy surface) sets the action:

      ship_fallback -> DEFAULT (the prior hardcoded behavior): escalate if a
                       stronger stage remains, else STOP (ship the carried
                       selection) -- the harness NO_REPRO_FALLBACK
                       behavior, now in the framework.
      escalate      -> escalate if a stronger stage remains, else ABANDON:
                       never ship an unverified patch (decline at the top of
                       the ladder instead of ship-falling-back).
      abandon       -> ABANDON the no-reading band immediately (never ship
                       unverified, never spend escalation on it) -- the old
                       pre-split framework behavior, now opt-in only.

    Reads signals['n_resolved'] and signals['n_samples'] (K), plus the
    optional 'failure_kind' / 'selected_regression_ok'. Degrades safely
    (treats a bare resolved flag as STOP; absent counts -> GATHER_SAMPLES;
    absent failure_kind -> no no-signal branch; absent
    selected_regression_ok -> identical to governor='agreement').
    """

    name = "diverse_agreement"
    aliases = ("cross_sample_agreement",)   # back-compat: pre-paper internal name
    escalation_kind = "tier"   # IF escalation is enabled, it is TIER not budget

    GOVERNORS = ("agreement", "pure_regok", "hybrid_regok")
    SHIP_GATES = ("repro_or_regok", "repro_and_regok")
    NO_SIGNAL_MODES = ("ship_fallback", "escalate", "abandon")

    # decision constants (richer than should_escalate's bool)
    STOP, GATHER_SAMPLES, ESCALATE_TIER, ABANDON = (
        "STOP", "GATHER_SAMPLES", "ESCALATE_TIER", "ABANDON")

    def __init__(self, max_tier: int = 2, min_samples: int = 2,
                 escalate_on_agreement: bool = False,
                 governor: str = "hybrid_regok",
                 ship_gate: str = "repro_or_regok",
                 no_signal: str = "ship_fallback"):
        if governor not in self.GOVERNORS:
            raise ValueError(f"governor must be one of {self.GOVERNORS}, got {governor!r}")
        if ship_gate not in self.SHIP_GATES:
            raise ValueError(f"ship_gate must be one of {self.SHIP_GATES}, got {ship_gate!r}")
        if no_signal not in self.NO_SIGNAL_MODES:
            raise ValueError(
                f"no_signal must be one of {self.NO_SIGNAL_MODES}, got {no_signal!r}")
        self.max_tier = max_tier
        self.min_samples = min_samples
        self.escalate_on_agreement = escalate_on_agreement
        self.governor = governor
        self.ship_gate = ship_gate
        self.no_signal = no_signal

    def _resolved(self, signals: dict[str, Any]) -> bool:
        return bool(signals.get("resolved")) or signals.get("n_resolved", 0) > 0

    def decision(self, signals: dict[str, Any], current_tier: int) -> str:
        """STOP / GATHER_SAMPLES / ESCALATE_TIER / ABANDON."""
        regok = signals.get("selected_regression_ok")
        if self.governor == "pure_regok":
            # Ship on the regression channel alone, regardless of repro outcome
            # (None is no-verdict, never a rescue); else fall to escalate/decline.
            if regok is True:
                return self.STOP
        elif self._resolved(signals):
            # Repro-pass ship -- unless the opt-in AND-gate demotes it (a repro
            # pass with a BROKEN regression verdict falls to escalate/decline).
            if not (self.ship_gate == "repro_and_regok" and regok is False):
                return self.STOP
        # Weak evidence: not enough diverse samples to trust a doomed call.
        if signals.get("n_samples", 0) < self.min_samples:
            return self.GATHER_SAMPLES
        # APPROVED no_signal split (sign-off 2026-06-10): zero-resolved with NO
        # usable execution reading (every grade UNKNOWN/NOPATCH -- see
        # selector.agreement_reading's failure_kind) is NOT a doom call.
        # The `no_signal` knob arbitrates: ship_fallback (default) escalates
        # when a stronger stage remains, else SHIPS the carried selection --
        # mirroring the harness NO_REPRO_FALLBACK, which ships the
        # no-repro band the framework used to silently abandon (the live
        # framework/harness divergence this branch removes); 'escalate'
        # declines instead of shipping at the top of the ladder; 'abandon'
        # declines the band outright. Absent failure_kind (hand-built
        # signals, logged agreement dicts) -> branch inert.
        if signals.get("failure_kind") == "no_signal":
            if self.no_signal == "abandon":
                return self.ABANDON
            if self.escalate_on_agreement and current_tier < self.max_tier:
                return self.ESCALATE_TIER
            return self.STOP if self.no_signal == "ship_fallback" else self.ABANDON
        # Cross-sample agreement on failure = within-tier doomed (~98% precise).
        # Default: ABANDON (validated, economical). Tier-escalation only if opted in
        # AND a stronger tier remains (marginal economics -- see class docstring).
        if self.escalate_on_agreement and current_tier < self.max_tier:
            return self.ESCALATE_TIER
        # hybrid_regok: the no-silent-abandon rescue -- the selected candidate
        # keeps the existing suite green, so ship the carried patch instead of
        # abandoning (held-out replay recovers 11/14 wrong-abandons, not 14/14).
        if self.governor == "hybrid_regok" and regok is True:
            return self.STOP
        return self.ABANDON

    def should_escalate(self, signals: dict[str, Any], current_tier: int) -> bool:
        """Bool view for the CascadeTrigger Protocol. True only when tier-escalation
        is opted in and fires; the validated DEFAULT (abandon) returns False --
        more same-tier compute is wasted, so a budget-escalating harness should not
        escalate on this signal."""
        return self.decision(signals, current_tier) == self.ESCALATE_TIER


__all__ = ["HeuristicCascade", "AlwaysEscalate", "NeverEscalate", "VerifierGate",
           "CrossSampleAgreementCascade"]

"""controller/runtime.py — the deployable entry point (Phase 11).

CodeCapsulesRunner wraps the validated VOI controller (orchestrator.run_controller)
behind a clean API + a policy object, so the framework runs WITHOUT the SWE-bench
harness. The caller supplies two adapters; the Runner owns the policy:

  sampler(config, tier) -> patch     run a diverse config at a model tier
  grade_fn(patch)       -> grade     the deployable verifier (e.g. repro+regression)

This is the surface that makes the framework a runtime rather than a pile of
helper functions: a deployer constructs a Runner from a policy (eventually
policy.yaml), passes their own sampler/grade_fn, and gets the validated
diverse-sample -> select+agreement -> cascade behaviour. The SWE-bench harness
becomes one such caller (its adapters wrap run_one + the repro/gold grade).

NOTE: this is the runtime entry; wiring the SWE-bench harness to CALL it (so the
harness collapses to a thin adapter) is the next, attended Phase 11 step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from code_capsules.controller.orchestrator import run_controller, ControllerRun
from code_capsules.controller.cascade_triggers import CrossSampleAgreementCascade
from code_capsules.controller.escalation_policy import EscalationStage
from code_capsules.controller.verifier import RESOLVED, FLAT, NOPATCH

ESCALATION_MODES = ("off", "gate")
P_SOURCE_KINDS = ("resolvability_gbm",)
P_SOURCE_FEATURES = ("static", "early_dynamics")


@dataclass(frozen=True)
class PSourceSpec:
    """Validated `p_source` policy spec (plan §4 items 5+6). Parse-time only:
    holds the paths/knobs, NEVER touches sklearn -- the model is constructed
    lazily (tools.resolvability_adapter.ResolvabilityPSource) at run() time and
    loads its pipeline only at predict time (plan G9: sklearn.ensemble is
    broken in every install on this machine)."""

    kind: str                 # resolvability_gbm
    model: str                # joblib path (manifest expected alongside)
    features: str = "static"  # static | early_dynamics (transfer finding: static wins)


@dataclass(frozen=True)
class RunnerPolicy:
    """The deployable VOI policy. Validated defaults (generalizability eval
    2026-06-03/04; governor replay EXP-4 n=300, 2026-06-10): a 2-config diverse
    ensemble, a single tier, ABANDON on agreement-failure (tier-escalation is
    marginal -- ~$2.42/resolve vs $1.40 base; opt in via the `escalation`
    ladder, or the deprecated `escalate_on_agreement` alias), and the
    hybrid_regok governor (rescue a would-be abandon when the selected patch's
    regression channel is True -- replay 64->77 first-150 / 76->87 held-out at
    zero added cost).

    DEPRECATED ALIAS: `escalate_on_agreement: true` in the controller block is
    kept for back-compat and maps to `escalation: {mode: gate}` with an empty
    ladder -- every tier in `tiers` runs the same `configs` with an
    unconditional ("always") trigger, exactly the legacy climb. New policies
    should write the `escalation` block instead; when both are present the
    explicit `escalation` block WINS (mode: off silences the alias)."""

    configs: tuple = ("floor", "siginject")   # the diverse SELECT/agreement ensemble
    tiers: tuple = ("default",)               # model ladder, weakest -> strongest
    escalate_on_agreement: bool = False       # DEPRECATED alias -- see class docstring
    min_samples: int = 2                      # K>=2 for the ~98%-precise doomed call
    governor: str = "hybrid_regok"            # agreement | pure_regok | hybrid_regok
    ship_gate: str = "repro_or_regok"         # | repro_and_regok (precision knob, opt-in)
    no_signal: str = "ship_fallback"          # | escalate | abandon (no-reading band)
    escalation_mode: str = "off"              # off | gate (off -> single-stage, ladder inert)
    escalation_ladder: tuple = ()             # tuple[EscalationStage], rung 2 onward
    escalation_voi: Optional[tuple] = None    # (value_per_resolve, cost_per_escalation)
                                              # defaults for ladder rows w/ trigger=voi
    p_source: Optional[PSourceSpec] = None    # pluggable VOI belief; ships DISABLED
    prompt_includes_fail_to_pass: bool = True  # G2 disclosure flag (see prompt.build_prompt;
                                              # True matches ALL historical evals)

    @classmethod
    def from_dict(cls, d: dict) -> "RunnerPolicy":
        """Build from a plain dict (the policy.yaml-parsed `controller:` block);
        unknown keys ignored, missing keys take the validated defaults, unknown
        VALUES raise ValueError (governor / ship_gate / no_signal /
        escalation.mode / ladder trigger / p_source.kind|features)."""
        d = d or {}
        governor = str(d.get("governor", cls.governor))
        if governor not in CrossSampleAgreementCascade.GOVERNORS:
            raise ValueError(
                f"controller.governor must be one of "
                f"{CrossSampleAgreementCascade.GOVERNORS}, got {governor!r}")
        ship_gate = str(d.get("ship_gate", cls.ship_gate))
        if ship_gate not in CrossSampleAgreementCascade.SHIP_GATES:
            raise ValueError(
                f"controller.ship_gate must be one of "
                f"{CrossSampleAgreementCascade.SHIP_GATES}, got {ship_gate!r}")
        no_signal = str(d.get("no_signal", cls.no_signal))
        if no_signal not in CrossSampleAgreementCascade.NO_SIGNAL_MODES:
            raise ValueError(
                f"controller.no_signal must be one of "
                f"{CrossSampleAgreementCascade.NO_SIGNAL_MODES}, got {no_signal!r}")
        pifp = d.get("prompt_includes_fail_to_pass", cls.prompt_includes_fail_to_pass)
        if not isinstance(pifp, bool):
            raise ValueError(
                f"controller.prompt_includes_fail_to_pass must be a bool, got {pifp!r}")
        tiers = tuple(d.get("tiers", cls.tiers))
        mode, ladder, voi = cls._parse_escalation(
            d.get("escalation"), bool(d.get("escalate_on_agreement", False)))
        if ladder and len(tiers) > 1:
            raise ValueError(
                "controller.escalation.ladder with a multi-entry tiers list is "
                "ambiguous: the ladder defines the rungs after tiers[0] -- use a "
                "single-entry tiers (the base stage) plus the ladder")
        return cls(
            configs=tuple(d.get("configs", cls.configs)),
            tiers=tiers,
            escalate_on_agreement=(mode == "gate"),
            min_samples=int(d.get("min_samples", cls.min_samples)),
            governor=governor,
            ship_gate=ship_gate,
            no_signal=no_signal,
            escalation_mode=mode,
            escalation_ladder=ladder,
            escalation_voi=voi,
            p_source=cls._parse_p_source(d.get("p_source")),
            prompt_includes_fail_to_pass=pifp,
        )

    @staticmethod
    def _parse_escalation(block, alias: bool):
        """Parse the `escalation:` sub-block -> (mode, ladder, voi).

        Absent block + escalate_on_agreement alias True -> ({mode: gate},
        empty ladder) = the legacy unconditional tier climb (deprecated alias
        mapping -- see the class docstring; no runtime warning by design).
        Ladder rows construct EscalationStage objects (their __post_init__
        validates trigger names and voi requirements); the block-level `voi`
        numbers are the default C/V for rows that omit voi_cost/voi_value."""
        if block is None:
            return ("gate" if alias else "off"), (), None
        block = block or {}
        mode = block.get("mode", "off")
        # YAML 1.1 footgun: bare `off`/`on` parse as booleans -- normalize.
        if mode is False:
            mode = "off"
        elif mode is True:
            mode = "gate"
        mode = str(mode)
        if mode not in ESCALATION_MODES:
            raise ValueError(
                f"controller.escalation.mode must be one of {ESCALATION_MODES}, "
                f"got {mode!r}")
        voi = block.get("voi")
        voi_t = None
        if voi is not None:
            try:
                voi_t = (float(voi["value_per_resolve"]),
                         float(voi["cost_per_escalation"]))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(
                    "controller.escalation.voi needs numeric value_per_resolve "
                    f"and cost_per_escalation, got {voi!r}") from e
        ladder = []
        for row in (block.get("ladder") or ()):
            if not isinstance(row, dict) or "tier" not in row:
                raise ValueError(
                    f"controller.escalation.ladder rows need a tier, got {row!r}")
            ladder.append(EscalationStage(
                tier=str(row["tier"]),
                configs=tuple(row.get("configs") or ()),
                trigger=str(row.get("trigger", "always")),
                voi_cost=row.get("voi_cost", voi_t[1] if voi_t else None),
                voi_value=row.get("voi_value", voi_t[0] if voi_t else None),
            ))
        return mode, tuple(ladder), voi_t

    @staticmethod
    def _parse_p_source(ps) -> Optional[PSourceSpec]:
        """Validate the `p_source:` key. null/None (the shipped default --
        DISABLED until tracking §5D promotion criteria are met) -> None; a dict
        needs kind=resolvability_gbm + a model path. Pure validation: NEVER
        imports sklearn/joblib/pandas (plan G9)."""
        if ps is None:
            return None
        if not isinstance(ps, dict):
            raise ValueError(f"controller.p_source must be null or a mapping, got {ps!r}")
        kind = str(ps.get("kind", ""))
        if kind not in P_SOURCE_KINDS:
            raise ValueError(
                f"controller.p_source.kind must be one of {P_SOURCE_KINDS}, got {kind!r}")
        model = ps.get("model")
        if not model:
            raise ValueError("controller.p_source needs a model path (joblib)")
        features = str(ps.get("features", "static"))
        if features not in P_SOURCE_FEATURES:
            raise ValueError(
                f"controller.p_source.features must be one of {P_SOURCE_FEATURES}, "
                f"got {features!r}")
        return PSourceSpec(kind=kind, model=str(model), features=features)


class CodeCapsulesRunner:
    """Run the validated adaptive-compute controller, harness-free."""

    def __init__(self, policy: Optional[RunnerPolicy] = None, p_source: Any = None):
        self.policy = policy or RunnerPolicy()
        # Optional injected belief source (duck-typed: .p_resolve(evidence) -> float).
        # When provided, the runner uses it directly and never imports the optional
        # private resolvability adapter, keeping the package standalone. Default None
        # (p_source disabled, the shipped default).
        self._p_source = p_source

    @classmethod
    def from_policy_dict(cls, d: dict) -> "CodeCapsulesRunner":
        return cls(RunnerPolicy.from_dict(d))

    @classmethod
    def from_policy_file(cls, path, key: str = "controller") -> "CodeCapsulesRunner":
        """Load the controller policy from a YAML file's `controller:` block --
        making policy.yaml LOAD-BEARING (it controls the runtime, not just docs).
        Falls back to the validated defaults if the file or block is absent/unreadable."""
        import yaml
        from pathlib import Path as _Path
        try:
            raw = yaml.safe_load(_Path(path).read_text()) or {}
        except (OSError, yaml.YAMLError):
            return cls()
        return cls.from_policy_dict(raw.get(key) or {})

    def _cascade(self, max_tier: Optional[int] = None) -> CrossSampleAgreementCascade:
        pol = self.policy
        return CrossSampleAgreementCascade(
            max_tier=max_tier if max_tier is not None else len(pol.tiers),
            min_samples=pol.min_samples,
            escalate_on_agreement=(pol.escalate_on_agreement
                                   or pol.escalation_mode == "gate"),
            governor=pol.governor,
            ship_gate=pol.ship_gate,
            no_signal=pol.no_signal,
        )

    def _build_p_source(self):
        """Construct the opt-in VOI belief source from the policy's PSourceSpec
        -- LAZILY, at run() time only (policy parsing never reaches here, so it
        never imports the ML stack; ResolvabilityPSource itself defers the
        joblib/sklearn load to first predict -- plan G9). None when disabled
        (the shipped `p_source: null` default)."""
        if self._p_source is not None:
            return self._p_source
        spec = self.policy.p_source
        if spec is None:
            return None
        # spec-string path: load the OPTIONAL private resolvability adapter (not shipped
        # in the package). Public callers inject a PSource via
        # CodeCapsulesRunner(..., p_source=<obj>) instead of using a spec string.
        from tools.resolvability_adapter import ResolvabilityPSource
        return ResolvabilityPSource(spec.model)

    def run(
        self,
        sampler: Callable[[str, str], str],
        grade_fn: Callable[[str], str],
        regression_fn: Optional[Callable[[str], Optional[bool]]] = None,
    ) -> ControllerRun:
        """Drive the validated controller over the policy's tier ladder.
        sampler(config, tier)->patch and grade_fn(patch)->grade are the caller's
        (benchmark or production) adapters; the optional regression_fn(patch)->
        bool|None adapter is the regression channel the hybrid_regok governor
        reads (e.g. make_repro_verifier(...).regression) -- without it the
        governor degrades to plain agreement behavior.

        Escalation: with `escalation.mode: gate` and a non-empty ladder, the
        run is the STAGED loop -- base stage = (tiers[0], configs), then the
        policy's EscalationStage rungs with their per-stage triggers. With an
        empty ladder (incl. the deprecated escalate_on_agreement alias) every
        tier in `tiers` runs the same configs with the legacy unconditional
        trigger; with mode off (the default) the run is single-stage,
        identical to the pre-ladder behavior."""
        pol = self.policy
        p_source = self._build_p_source()
        if pol.escalation_mode == "gate" and pol.escalation_ladder:
            stages = ((EscalationStage(tier=pol.tiers[0], configs=pol.configs),)
                      + tuple(pol.escalation_ladder))
            return run_controller(
                sampler=sampler, grade_fn=grade_fn, stages=stages,
                cascade=self._cascade(max_tier=len(stages)),
                regression_fn=regression_fn, p_source=p_source,
            )
        return run_controller(
            self.policy.tiers, self.policy.configs, sampler, grade_fn,
            cascade=self._cascade(), regression_fn=regression_fn,
            p_source=p_source,
        )


def make_variant_sampler(task, config, *, gate=None, cost_model=None):
    """Turn ONE registered Variant into the (sampler, grade_fn) pair the runner drives.

    The convergence helper: a "single variant" is just a ONE-CONFIG policy on the one
    public runner, and "run that variant" is a sampler. This builds that sampler from a
    registered Variant + its VariantConfig and CAPTURES the full RunResult, so the caller
    can log cost/turns/tokens (the controller itself is patch-in / grade-out and stays
    domain-agnostic -- it never sees telemetry).

      sampler, grade_fn, captured = make_variant_sampler(task, cfg, gate=g, cost_model=cm)
      CodeCapsulesRunner(RunnerPolicy(configs=(cfg.name,), tiers=("default",))).run(sampler, grade_fn)
      result = captured["result"]            # the RunResult, for logging

    The grade_fn reads the captured RunResult.resolved -- the variant's OWN quality-gate
    verdict, computed during variant.run -- so the controller adds NO extra eval. For a
    one-config policy this yields RESOLVED (gate passed) or EXHAUSTED (no cross-sample
    doom call is possible at K=1); either way captured["result"] holds the RunResult.

    gate / cost_model: optional instances to register + splice into config.extra by name
    (mirrors how the former pipeline runner resolved them), so custom, non-registrable
    instances (e.g. a DockerEvalGate built around a callable) are found by the variant's
    orchestration via the registry.
    """
    from code_capsules.api.registry import get as _get, register as _register
    if gate is not None:
        _register("quality_gate", gate)
        config.extra.setdefault("quality_gate", gate.name)
    if cost_model is not None:
        _register("cost_model", cost_model)
        config.extra.setdefault("cost_model", cost_model.name)
    captured: dict = {"result": None}

    def sampler(config_name, tier):
        variant = _get("variant", config.name)
        result = variant.run(task, config)
        captured["result"] = result
        return result.patch or ""

    def grade_fn(patch):
        rr = captured["result"]
        if rr is not None and rr.resolved:
            return RESOLVED
        return NOPATCH if not (patch or "").strip() else FLAT

    return sampler, grade_fn, captured


__all__ = ["RunnerPolicy", "PSourceSpec", "CodeCapsulesRunner", "make_variant_sampler",
           "ESCALATION_MODES", "P_SOURCE_KINDS", "P_SOURCE_FEATURES"]

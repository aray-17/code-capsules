"""
CodeCapsulesPolicy: declarative policy for single-session coding-agent
execution.

A ``CodeCapsulesPolicy`` is the operator's primary handle on the framework:
it declares which variant to run, the turn budget that variant gets, the
escalation behavior on quality-gate failure, the routing class the policy
applies to, and the model tier it was calibrated on. Each policy is a
single per-cell deployment choice; the shipped ``policy.yaml`` is a table
of named policies keyed by ``(workload_class, tier, knee)`` so a deployer
selects from a Pareto frontier rather than constructing a policy by hand.

.. note::
    A ``CodeCapsulesPolicy`` selects ONE variant (a single solver). The
    **deployable lever** (a diverse *pair* of solvers with verifier-select
    (the quality lever) and the diverse-sample agreement governor (the cost /
    abandon lever)) is a different composition, expressed by
    :class:`code_capsules.controller.runtime.RunnerPolicy` and run by the
    package-root :class:`~code_capsules.CodeCapsulesRunner` (the ``controller:``
    block of ``policy.yaml``). This DSL covers the single-variant cells; the
    lever is the three-component cost/quality controller.

Same model as Agent Capsules' :class:`ControllerPolicy`: a dataclass with
sensible defaults and a public preset selector
(:func:`policy_for`). Most operators use a preset and override one or two
fields; advanced operators construct a policy from scratch when their
deployment falls outside the shipped calibration.

Quickstart
----------

Use a shipped preset (typed kwargs, no YAML):

.. code-block:: python

    from code_capsules import policy_for

    # Cost-axis knee (two-pass critique) on Sonnet for hard agent workloads.
    policy = policy_for(
        workload="hard_workload",
        tier="sonnet",
        knee="balanced",
    )

Construct a policy directly when calibrating a new cell:

.. code-block:: python

    from code_capsules import CodeCapsulesPolicy

    policy = CodeCapsulesPolicy(
        variant="two_pass_critique",
        turn_budget=10,
        escalating_target_budget=25,
        always_escalate=True,
        quality_gate="docker_eval",
        cascade_trigger="heuristic",
    )

Load from a YAML cell table:

.. code-block:: python

    from code_capsules import CodeCapsulesPolicy

    policy = CodeCapsulesPolicy.from_yaml(
        "policy.yaml",
        workload="hard_workload",
        tier="sonnet",
        knee="quality",
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover - hard dep
    raise ImportError(
        "PyYAML is required for the policy DSL. Install with: pip install pyyaml"
    ) from e


class PolicyError(ValueError):
    """Raised when a policy YAML file has invalid content."""


# ── Vocabulary for typed string fields ─────────────────────────────────────

_VALID_VARIANTS = {
    "signaled_budget",
    "implicit_budget",
    "unbounded_budget",     # budget-knob alias: implicit framing at an effectively
                            # unbounded turn cap (b=100); the paper's "unbounded budget"
                            # menu entry (Table: shipped presets, quality/ceiling cells)
    "plan_then_execute",
    "two_pass_critique",
    "relevance_ranker",
    "stuck_signal_injection",
    "per_tool_class_hint",
    "phase_staged",
    "tool_alloc",
}

_VALID_MODES = {"sequential", "escalating", "routed", "fine"}

_VALID_QUALITY_GATES = {
    "docker_eval",
    "binary_tests",
    "python_ast",
    "always_pass",
}

_VALID_CASCADE_TRIGGERS = {
    "heuristic",
    "always_escalate",
    "never_escalate",
    "verifier_gate",            # execution-verifier-grounded gate
    "diverse_agreement",        # validated VOI cost lever (regression-gated hybrid;
                                # escalate TIER on diverse-agreement-fail)
    "cross_sample_agreement",   # back-compat alias for diverse_agreement
}

_VALID_PROMPT_VARIANTS = {
    "default",
    "plan_first",
    "two_pass_critique",
    "stuck_signal_injection",
    "phase_staged",
    "per_tool_class_hint",
}


#: Budget-knob / alias framings -> the registered Variant that actually executes
#: them. The framings name a cost-quality cell in the menu (paper Table:
#: shipped presets) but are not themselves registered executors; to_variant_config
#: resolves them so a single-variant policy is directly runnable.
_VARIANT_EXECUTOR = {
    "implicit_budget": "signaled_budget",   # no in-prompt budget signal
    "unbounded_budget": "signaled_budget",  # no signal, large turn cap
    "tool_alloc": "per_tool_class_hint",    # the per-tool-class allocation variant
}


def _registered_names(kind: str) -> set:
    """Names registered for `kind` in the extension registry (custom impls).

    Lazy + defensive: imported only when a name is NOT in the shipped catalog,
    so the module-load-time construction of SHIPPED_PRESETS (which uses shipped
    names only) never triggers the registry bootstrap and there is no import
    cycle. This is what makes a user-registered custom variant / quality_gate /
    cascade_trigger selectable by name through CodeCapsulesPolicy (the headline
    one-line extensibility claim)."""
    try:
        from code_capsules.api import registry
        return set(registry.list_registered(kind))
    except Exception:                       # pragma: no cover - defensive
        return set()


@dataclass
class CodeCapsulesPolicy:
    """
    Declarative policy for a single coding-agent execution cell.

    All fields have sensible defaults; in practice operators either pick a
    shipped preset via :func:`policy_for` or construct a policy from the
    YAML calibration table. Constructing a policy by hand is the escape
    hatch for new cells outside the shipped calibration.

    Variant selection: which execution strategy the agent runs:

    Attributes:
        variant:
            Name of the execution variant to run. One of
            ``signaled_budget`` (single-pass with a turn-budget hint in the
            system prompt), ``implicit_budget`` (single-pass with no
            in-prompt budget signal), ``plan_then_execute`` (single-pass
            with a "produce a plan before any tool call" instruction),
            ``two_pass_critique`` (escalating; second pass with a critique
            prompt on quality-gate failure), ``relevance_ranker`` (single-
            pass with a BM25 file ranking prepended to the system prompt),
            ``stuck_signal_injection`` (single-pass with signal-triggered
            mid-session prompt extensions), ``per_tool_class_hint``
            (single-pass with a tool-class allocation hint), ``phase_staged``
            (single-pass with explicit phase markers in the prompt), or
            ``tool_alloc`` (single-pass with a tool-budget allocation).

        mode:
            Execution mode the framework runs the variant under. One of
            ``sequential`` (single attempt, no re-entry), ``escalating``
            (initial attempt under the start budget; on quality-gate
            failure, re-attempt under the target budget with a re-attempt
            prompt), ``routed`` (workload classifier chooses among shipped
            variants per task), or ``fine`` (legacy single-call mode used
            for HumanEval/MBPP). Most coding cells use ``sequential`` or
            ``escalating``.

    Budget: how many turns the agent gets:

    Attributes:
        turn_budget:
            Maximum number of agent turns in a single attempt. The agent's
            tool-call loop terminates when this is reached even if the
            quality gate would have continued. Default 20.
        prompt_budget_hint:
            When set, communicates the budget to the agent in its system
            prompt ("You have ``prompt_budget_hint`` turns left"). When
            ``None``, the agent is not told its budget; this is the
            ``implicit`` framing. The hint may differ from ``turn_budget``;
            for example, a 25-turn hard cap with a 20-turn hint asks the
            agent to pace for 20 while leaving 5 turns of headroom.

    Escalation budgets, only consulted when ``mode == "escalating"``:

    Attributes:
        escalating_start_budget:
            Turn budget for the initial attempt under
            ``mode="escalating"``. The first attempt runs to this cap;
            if the quality gate fails, a second attempt is launched.
        escalating_target_budget:
            Combined turn cap (start + extension) for the second
            attempt under ``mode="escalating"``. Must be
            ``>= escalating_start_budget``.
        always_escalate:
            When ``True``, the second attempt runs whenever the first did
            not pass the quality gate (``escalate = not resolved``). NOTE:
            in a benchmark this consults the GOLD gate, so it is NOT
            leakage-free and NOT deployable as-is: the gold verdict is
            unavailable at deployment time. For a leakage-free always-both
            measurement use the harness ``force_stage2`` mode; for a
            deployable gate use the execution-verifier cascade trigger.
            When ``False``, the cascade trigger decides from
            deployment-available signals.

    Mechanism selection: which Protocol implementations the framework
    uses to wrap this variant:

    Attributes:
        prompt_variant:
            Name of the prompt-shaping variant prepended to the system
            prompt. Values follow the same vocabulary as ``variant`` but
            are interpreted by the prompt-construction layer, not the
            execution-mode layer; usually equals ``variant`` but may
            differ for advanced compositions (e.g. ``two_pass_critique``
            execution with ``plan_first`` first-pass prompt).
        quality_gate:
            Name of the :class:`QualityGate` implementation that scores
            each attempt. ``docker_eval`` runs the SWE-bench gold tests
            under Docker; ``binary_tests`` runs pytest on the worktree;
            ``python_ast`` checks for compilable Python; ``always_pass``
            disables the gate (useful for purely cost-oriented sweeps).
        cascade_trigger:
            Name of the :class:`CascadeTrigger` implementation that
            decides whether to re-attempt after a gate failure. Only
            consulted when ``mode == "escalating"``. ``heuristic``
            (re-attempt if the gate failed AND the patch was non-empty),
            ``always_escalate`` (re-attempt unconditionally; equivalent
            to setting ``always_escalate=True``), or ``never_escalate``
            (single attempt; collapses ``escalating`` mode to
            ``sequential``).

    Deployment-table coordinates, used by :func:`policy_for` and by the
    YAML loader to select an entry from the shipped policy file:

    Attributes:
        workload_class:
            Logical workload class this policy was calibrated on. The
            shipped calibration ships
            ``hard_workload`` (SWE-bench-Lite-style bug fixes),
            ``scientific_workload`` (scientific codebases: astropy,
            scikit-learn, sympy), and
            ``saturated_workload`` (HumanEval/MBPP-style single-shot
            completion). Custom workload classes are accepted as free
            strings and dispatched by the workload classifier at runtime.
        tier:
            Model tier this policy is calibrated for: one of
            ``haiku``, ``sonnet``, or ``opus``. Used by
            :func:`policy_for` to select a cell from the policy table;
            advisory only at runtime (the framework's
            :class:`ModelClient` is configured independently).
        knee:
            Cost-quality knee this policy occupies on the cross-tier
            frontier. One of
            ``cost_min`` (cheapest credible cell, Sonnet),
            ``balanced`` (cost-axis knee, two-pass critique on Sonnet),
            ``quality`` (best Sonnet quality, unbounded budget),
            ``quality_max`` (Opus, dominates the Sonnet floor on both
            axes), or
            ``ceiling`` (Opus unbounded-budget upper bound).

    Measurement provenance, populated when a policy was selected from a
    shipped calibration table; both fields are advisory (the framework
    does not enforce that runtime measurements match these):

    Attributes:
        pass_rate:
            Resolved-instance fraction this policy achieved on the
            shipped calibration set (``None`` when not from a table).
        cost_per_task:
            Per-attempt cost in USD this policy achieved on the shipped
            calibration set under cache-aware accounting
            (``None`` when not from a table).
    """

    # ── Variant selection ─────────────────────────────────────────────────
    variant: str = "signaled_budget"
    mode: str = "sequential"

    # ── Budget ────────────────────────────────────────────────────────────
    turn_budget: int = 20
    prompt_budget_hint: int | None = None

    # ── Escalation budgets (mode="escalating" only) ───────────────────────
    escalating_start_budget: int = 10
    escalating_target_budget: int = 25
    always_escalate: bool = False

    # ── Mechanism selection ───────────────────────────────────────────────
    prompt_variant: str = "default"
    quality_gate: str = "docker_eval"
    cascade_trigger: str = "heuristic"

    # ── Deployable composition (the three-component lever) ─────────────────
    # Optional `controller:` block - the diverse-sample SELECT + agreement
    # governor that composes solvers + repro-verifier + governor into the
    # ship/abandon/escalate machine (Figure: deployable anatomy). Its shape is
    # the validated RunnerPolicy dict (configs, tiers, min_samples, governor,
    # ship_gate, escalation); `runner_policy()` materialises it. None = the
    # single-config path (variant/mode above) with no agreement governor.
    controller: dict | None = None

    # ── Deployment-table coordinates (advisory) ───────────────────────────
    workload_class: str = "hard_workload"
    tier: str = "sonnet"
    knee: str = "balanced"

    # ── Measurement provenance (populated by table loaders) ───────────────
    pass_rate: float | None = None
    cost_per_task: float | None = None

    def __post_init__(self) -> None:
        if self.variant not in _VALID_VARIANTS and self.variant not in _registered_names("variant"):
            raise ValueError(
                f"variant {self.variant!r} not in shipped catalog "
                f"{sorted(_VALID_VARIANTS)} and not registered. Register a custom "
                f"variant via code_capsules.api.registry.register('variant', impl)."
            )
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, "
                f"got {self.mode!r}"
            )
        if self.turn_budget < 1:
            raise ValueError(
                f"turn_budget must be >= 1, got {self.turn_budget}"
            )
        if (
            self.prompt_budget_hint is not None
            and self.prompt_budget_hint < 1
        ):
            raise ValueError(
                f"prompt_budget_hint must be >= 1 when set, "
                f"got {self.prompt_budget_hint}"
            )
        if self.escalating_start_budget < 1:
            raise ValueError(
                f"escalating_start_budget must be >= 1, "
                f"got {self.escalating_start_budget}"
            )
        if self.escalating_target_budget < self.escalating_start_budget:
            raise ValueError(
                f"escalating_target_budget ({self.escalating_target_budget}) "
                f"must be >= escalating_start_budget "
                f"({self.escalating_start_budget})"
            )
        if self.prompt_variant not in _VALID_PROMPT_VARIANTS:
            raise ValueError(
                f"prompt_variant must be one of "
                f"{sorted(_VALID_PROMPT_VARIANTS)}, "
                f"got {self.prompt_variant!r}"
            )
        if (self.quality_gate not in _VALID_QUALITY_GATES
                and self.quality_gate not in _registered_names("quality_gate")):
            raise ValueError(
                f"quality_gate {self.quality_gate!r} not in shipped catalog "
                f"{sorted(_VALID_QUALITY_GATES)} and not registered. Register a custom "
                f"gate via code_capsules.api.registry.register('quality_gate', impl)."
            )
        if (self.cascade_trigger not in _VALID_CASCADE_TRIGGERS
                and self.cascade_trigger not in _registered_names("cascade_trigger")):
            raise ValueError(
                f"cascade_trigger {self.cascade_trigger!r} not in shipped catalog "
                f"{sorted(_VALID_CASCADE_TRIGGERS)} and not registered. Register a custom "
                f"trigger via code_capsules.api.registry.register('cascade_trigger', impl)."
            )
        if self.pass_rate is not None and not 0.0 <= self.pass_rate <= 1.0:
            raise ValueError(
                f"pass_rate must be in [0, 1] when set, got {self.pass_rate}"
            )
        if self.cost_per_task is not None and self.cost_per_task < 0:
            raise ValueError(
                f"cost_per_task must be >= 0 when set, "
                f"got {self.cost_per_task}"
            )
        if self.controller is not None:
            if not isinstance(self.controller, dict):
                raise ValueError(
                    f"controller must be a dict (the RunnerPolicy block), "
                    f"got {type(self.controller).__name__}"
                )
            # Validate by parsing into the typed RunnerPolicy now (raises
            # ValueError on bad governor/ship_gate/no_signal/escalation/etc.),
            # so a bad controller block fails at construction, not at run time.
            self.runner_policy()

    def runner_policy(self):
        """Materialise the optional ``controller:`` block as a typed RunnerPolicy.

        Returns the validated :class:`~code_capsules.controller.runtime.RunnerPolicy`
        for this policy's deployable composition, or ``None`` when no
        ``controller`` block is set (the single-config path). Lazy import keeps
        the policy dataclass free of a runtime dependency.
        """
        if self.controller is None:
            return None
        from code_capsules.controller.runtime import RunnerPolicy
        return RunnerPolicy.from_dict(self.controller)

    def to_variant_config(self):
        """Materialise this single-variant policy as an executable ``VariantConfig``.

        The runner consumes a ``VariantConfig`` whose ``name`` is a REGISTERED
        variant. This resolves the budget-knob framings to their executors
        (``implicit_budget`` / ``unbounded_budget`` -> the ``signaled_budget``
        executor with no in-prompt budget; ``tool_alloc`` -> ``per_tool_class_hint``)
        and carries the budget / escalation / prompt fields, so a policy pulled
        from the shipped menu is directly runnable:

            cfg = policy_for(knee="balanced").to_variant_config()
            sampler, grade_fn, captured = make_variant_sampler(task, cfg, gate=...)
            CodeCapsulesRunner(
                RunnerPolicy(configs=(cfg.name,), tiers=("default",))
            ).run(sampler, grade_fn)
        """
        from code_capsules.api import VariantConfig
        executor = _VARIANT_EXECUTOR.get(self.variant, self.variant)
        implicit = self.variant in ("implicit_budget", "unbounded_budget")
        escalating = self.mode == "escalating"
        return VariantConfig(
            name=executor,
            mode=self.mode,
            turn_budget=self.turn_budget,
            prompt_budget_hint=None if implicit else self.prompt_budget_hint,
            prompt_variant=None if self.prompt_variant == "default" else self.prompt_variant,
            escalating_start_budget=self.escalating_start_budget if escalating else None,
            escalating_target_budget=self.escalating_target_budget if escalating else None,
            always_escalate=self.always_escalate,
        )

    # ── YAML loaders ──────────────────────────────────────────────────────

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        workload: str | None = None,
        tier: str | None = None,
        knee: str | None = None,
    ) -> "CodeCapsulesPolicy":
        """
        Load a policy from a YAML calibration table.

        The YAML file is expected to follow the
        ``deployment_defaults[workload][tier][knee]`` schema shipped
        with the framework. When ``workload``, ``tier``, or ``knee`` is
        ``None``, falls back to the dataclass defaults
        (``hard_workload``, ``sonnet``, ``balanced``).

        Args:
            path: Path to the YAML file.
            workload: Workload class to look up
                (default: ``"hard_workload"``).
            tier: Model tier to look up (default: ``"sonnet"``).
            knee: Cost-quality knee to look up (default: ``"balanced"``).

        Returns:
            A populated :class:`CodeCapsulesPolicy`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            PolicyError: If the YAML is malformed, the named entry is
                missing, or an entry field is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise PolicyError(f"Invalid YAML in {path}: {exc}") from exc

        wl = workload or cls.__dataclass_fields__["workload_class"].default
        tr = tier or cls.__dataclass_fields__["tier"].default
        kn = knee or cls.__dataclass_fields__["knee"].default

        defaults = raw.get("deployment_defaults", {})
        try:
            entry = defaults[wl][tr][kn]
        except KeyError as exc:
            raise PolicyError(
                f"{path}: no deployment_defaults entry for "
                f"workload={wl!r} tier={tr!r} knee={kn!r} "
                f"(missing key: {exc})"
            ) from exc

        return cls._from_entry(entry, wl=wl, tr=tr, kn=kn, source=str(path))

    @classmethod
    def from_yaml_string(
        cls,
        text: str,
        *,
        workload: str = "hard_workload",
        tier: str = "sonnet",
        knee: str = "balanced",
    ) -> "CodeCapsulesPolicy":
        """Load a policy from a YAML string (testing and embedding)."""
        try:
            raw = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise PolicyError(f"Invalid YAML: {exc}") from exc
        defaults = raw.get("deployment_defaults", {})
        try:
            entry = defaults[workload][tier][knee]
        except KeyError as exc:
            raise PolicyError(
                f"no deployment_defaults entry for "
                f"workload={workload!r} tier={tier!r} knee={knee!r} "
                f"(missing key: {exc})"
            ) from exc
        return cls._from_entry(
            entry, wl=workload, tr=tier, kn=knee, source="<string>"
        )

    def to_yaml_entry(self) -> dict[str, Any]:
        """
        Serialise this policy to a ``deployment_defaults`` entry dict.

        The returned dict is suitable for nesting under
        ``deployment_defaults[workload][tier][knee]`` in a YAML policy
        file. ``workload_class``, ``tier``, and ``knee`` are omitted
        from the entry (the cell coordinates are the keys above the
        entry, not fields inside it). ``pass_rate`` and
        ``cost_per_task`` are emitted when set.
        """
        out: dict[str, Any] = {
            "variant": self.variant,
            "mode": self.mode,
            "turn_budget": self.turn_budget,
        }
        if self.prompt_budget_hint is not None:
            out["prompt_budget_hint"] = self.prompt_budget_hint
        if self.mode == "escalating":
            out["escalating_start_budget"] = self.escalating_start_budget
            out["escalating_target_budget"] = self.escalating_target_budget
            out["always_escalate"] = self.always_escalate
        if self.prompt_variant != "default":
            out["prompt_variant"] = self.prompt_variant
        if self.quality_gate != "docker_eval":
            out["quality_gate"] = self.quality_gate
        if self.cascade_trigger != "heuristic":
            out["cascade_trigger"] = self.cascade_trigger
        if self.pass_rate is not None:
            out["pass_rate"] = self.pass_rate
        if self.cost_per_task is not None:
            out["cost_per_task"] = self.cost_per_task
        return out

    #: Non-typed keys a shipped calibration entry may carry as provenance or
    #: harness-execution detail. They are not :class:`CodeCapsulesPolicy` fields
    #: (the typed surface is variant/mode/budgets/mechanisms), so the loader
    #: tolerates them rather than rejecting the whole file, while a genuinely
    #: unknown key (a typo / wrong field) still raises.
    _IGNORED_ENTRY_FIELDS = frozenset({
        "config",                    # legacy alias for `variant` (mapped below)
        "model",                     # redundant with the tier key; provenance
        "source", "note",           # calibration provenance
        "n", "resolved", "usd_per_resolve", "registry_key",  # menu provenance
        "preselect_top_n",           # harness-execution knob (ranker top-N)
        "escalating_stage2_mode",    # harness-execution knob (stage-2 prompt)
        "force_stage2",              # harness-execution knob (deployable two-pass)
    })

    @classmethod
    def _from_entry(
        cls, entry: dict, *, wl: str, tr: str, kn: str, source: str
    ) -> "CodeCapsulesPolicy":
        # Build kwargs from the entry, accepting only known fields and
        # forwarding the table coordinates as workload_class/tier/knee.
        known = {f.name for f in fields(cls)}
        unknown = set(entry) - known - cls._IGNORED_ENTRY_FIELDS
        if unknown:
            raise PolicyError(
                f"{source}: deployment_defaults entry "
                f"[{wl}][{tr}][{kn}] contains unknown field(s) "
                f"{sorted(unknown)}. Valid fields: {sorted(known)}; "
                f"tolerated provenance/harness fields: {sorted(cls._IGNORED_ENTRY_FIELDS)}"
            )
        # 'config' is the legacy alias for 'variant'; accept either.
        kwargs = {k: v for k, v in entry.items() if k in known}
        if "variant" not in kwargs and "config" in entry:
            kwargs["variant"] = entry["config"]
        kwargs["workload_class"] = wl
        kwargs["tier"] = tr
        kwargs["knee"] = kn
        return cls(**kwargs)


# ── Shipped deployment presets ─────────────────────────────────────────────
#
# Same shape as Agent Capsules' SENSITIVITY_PRESETS: a name → policy table
# that ships with the framework, anchored on per-tier measurement. The
# presets here are the four cost-quality knees on Sonnet 4.6 for
# hard agent workloads (SWE-bench Lite first 150, cache-aware accounting).
# Add presets for other (workload, tier) cells by extending this dict in
# your deployment or by loading from a custom policy.yaml.

SHIPPED_PRESETS: dict[tuple[str, str, str], CodeCapsulesPolicy] = {
    # Hard agent workloads (SWE-bench Lite first 150, cache-aware accounting).
    # The validated cross-tier cost-quality menu; the knee names and numbers
    # mirror policy.yaml's deployment_defaults and the paper's shipped-presets
    # table. cost_min/balanced/quality run on Sonnet; quality_max/ceiling on Opus.
    # cost_per_task is the cache-aware $/attempt on the calibration set.
    ("hard_workload", "sonnet", "cost_min"): CodeCapsulesPolicy(
        variant="signaled_budget",
        mode="sequential",
        turn_budget=10,
        prompt_budget_hint=10,
        workload_class="hard_workload",
        tier="sonnet",
        knee="cost_min",
        pass_rate=0.560,        # 84/150, the cheapest credible cell (dominates
                                # the relevance ranker on Sonnet)
        cost_per_task=0.174,
    ),
    ("hard_workload", "sonnet", "balanced"): CodeCapsulesPolicy(
        variant="two_pass_critique",
        mode="escalating",
        turn_budget=10,
        escalating_start_budget=10,
        escalating_target_budget=25,
        prompt_variant="two_pass_critique",
        workload_class="hard_workload",
        tier="sonnet",
        knee="balanced",
        pass_rate=0.440,        # 66/150, leak-free (deploy via force_stage2)
        cost_per_task=0.41,
    ),
    ("hard_workload", "sonnet", "quality"): CodeCapsulesPolicy(
        variant="unbounded_budget",
        mode="sequential",
        turn_budget=100,
        workload_class="hard_workload",
        tier="sonnet",
        knee="quality",
        pass_rate=0.513,        # 77/150, best Sonnet quality (dominated by quality_max)
        cost_per_task=0.47,
    ),
    ("hard_workload", "opus", "quality_max"): CodeCapsulesPolicy(
        variant="implicit_budget",
        mode="sequential",
        turn_budget=20,
        workload_class="hard_workload",
        tier="opus",
        knee="quality_max",
        pass_rate=0.613,        # 92/150, dominates the Sonnet floor on both axes
        cost_per_task=0.48,
    ),
    ("hard_workload", "opus", "ceiling"): CodeCapsulesPolicy(
        variant="unbounded_budget",
        mode="sequential",
        turn_budget=100,
        workload_class="hard_workload",
        tier="opus",
        knee="ceiling",
        pass_rate=0.640,        # 96/150, 93% of the 103/150 capability cap
        cost_per_task=0.60,
    ),
}


def policy_for(
    *,
    workload: str = "hard_workload",
    tier: str = "sonnet",
    knee: str = "balanced",
) -> CodeCapsulesPolicy:
    """
    Return the shipped :class:`CodeCapsulesPolicy` for a given cell.

    The framework ships a calibration table covering the per-tier
    cost-quality knees on hard agent workloads
    (Sonnet 4.6, SWE-bench Lite first 150, cache-aware accounting). Use
    this factory to pull a pre-calibrated policy out of the table by
    cell coordinates; override the returned policy's fields with
    :func:`dataclasses.replace` for per-customer or per-deployment
    customization.

    Args:
        workload: Workload class. Default ``"hard_workload"``.
        tier: Model tier. Default ``"sonnet"``.
        knee: Cost-quality knee. One of ``"cost_min"``, ``"balanced"``,
            ``"quality"`` (Sonnet), ``"quality_max"``, or ``"ceiling"``
            (Opus). Default ``"balanced"``.

    Returns:
        A pre-configured :class:`CodeCapsulesPolicy`.

    Raises:
        ValueError: If the ``(workload, tier, knee)`` cell is not in
            the shipped calibration table. Custom cells are loaded
            from a per-deployment ``policy.yaml`` via
            :meth:`CodeCapsulesPolicy.from_yaml`.

    Example:

    .. code-block:: python

        from code_capsules import policy_for
        from dataclasses import replace

        # Cost-axis knee (two-pass critique) on Sonnet for hard agent workloads
        policy = policy_for(knee="balanced")

        # Same knee, but override the turn budget for a tighter deployment
        policy = replace(policy_for(knee="balanced"), turn_budget=15)
    """
    key = (workload, tier, knee)
    if key not in SHIPPED_PRESETS:
        keys = sorted(SHIPPED_PRESETS)
        raise ValueError(
            f"No shipped preset for workload={workload!r} tier={tier!r} "
            f"knee={knee!r}. Shipped presets: {keys}. "
            f"For custom cells, load from a policy.yaml via "
            f"CodeCapsulesPolicy.from_yaml(...)."
        )
    return SHIPPED_PRESETS[key]


__all__ = [
    "CodeCapsulesPolicy",
    "PolicyError",
    "SHIPPED_PRESETS",
    "policy_for",
]

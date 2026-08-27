# code-capsules

A calibrated **cost-quality configuration framework** for single-session
coding agents. You declare a policy once; the runtime runs a calibrated
solver configuration for the workload, keeps the patch an execution
verifier confirms, and stops spending on instances a deployable signal
marks unsolvable at this tier.

> **From the paper?** This codebase is the artifact for *The Economics of
> Coding Agents: Calibrating the Cost-Quality Frontier*. The exact code
> state cited in the paper is tagged
> [`v1.0-arxiv`](https://github.com/aray-17/code-capsules/releases/tag/v1.0-arxiv);
> `main` may have evolved since. Every load-bearing number reproduces
> offline - see [Reproducing the paper's claims](#reproducing-the-papers-claims).

## The framework is three composable components

- **Solvers** - a catalog of execution variants (signaled budget,
  plan-then-execute, relevance ranker, signal injection, per-tool-class
  hint, phase-staged, two-pass critique, and more) that generate
  candidate patches.
- **Repro-verifier** (the *quality* lever) - keep the patch a generated
  reproduction confirms, scored without consulting held-out tests.
- **Governor** (the *cost* lever) - stop spending on instances a
  deployable signal (a failed reproduction together with a broken
  regression suite) marks doomed at this tier.

**Two things ship**, both calibrated and checked on a held-out split:
(1) a per-workload **menu** of single configurations on the cost-quality
Pareto frontier, and (2) a **regression-suite governor** that recovers a
majority of wrongly-abandoned resolves, narrowing the gap to the
unbounded floor. The single-tier run-both-and-abandon composition is
reported as a **first-class negative**: it adds zero resolves over its
best single member at roughly twice the cost, and an escalation gate
loses its selectivity once the strong tier runs.

## Headline results

Same models, same data split, same scorer; cost as one modeled
per-instance surface.

| Comparison | Result |
|---|---|
| vs. Agentless (head-to-head, SWE-bench Lite, 300 instances) | **172/300 resolved (57.3%) @ \$0.445/inst** vs. Agentless 152/300 (50.7%) @ \$0.456, more resolved at lower cost (Pareto; McNemar p=0.0055, 13.8% lower cost per resolve) |
| Calibrated cross-tier menu (resolved/150) | cost-min 84 @ \$0.17 · balanced 98 @ \$0.41 · quality 111 @ \$0.47 · quality-max 128 @ \$0.48 · ceiling 138 @ \$0.60 |
| Regression-suite governor | recovers a majority of wrongly-abandoned resolves (narrowing the gap to the unbounded floor) |

Full methodology, per-claim evidence, and the negative results are in
[`paper/paper.pdf`](paper/paper.pdf) and [`CLAIMS.md`](CLAIMS.md).

## Quickstart

```bash
pip install -e ".[dev]"
```

```python
from code_capsules import policy_for, CodeCapsulesRunner

# 1. Pick a calibrated preset from the shipped cross-tier menu.
policy = policy_for(workload="hard_workload", tier="sonnet", knee="balanced")
print(policy.variant, policy.turn_budget, f"${policy.cost_per_task:.2f}/attempt")

# 2. Or load the deployable lever (diverse solver pair + repro-verifier + governor).
lever = CodeCapsulesRunner.from_policy_file("policy.yaml")
# lever.run(sampler, grade_fn)   # wire your model sampler + grader; see examples/deployable_lever.py
```

The configuration layer is fully offline (no API keys). Running the
lever against real models requires a model sampler and a grader - see
[`examples/deployable_lever.py`](examples/deployable_lever.py).

## Reproducing the paper's claims

The headline claims reproduce **offline** (no Docker, no API keys, no
model calls) from the committed evaluation data in [`evals/`](evals/),
including the deployment-governor analysis (the diverse-sample
agreement signal, the regression-suite gate, the escalation gate, and
the value-of-resolve rule):

```bash
python3 benchmarks/verify_criteria.py     # PASS/FAIL per claim; every claim reproduces offline
```

Or browse them interactively in the **evidence explorer** - a
dependency-free static page that shows each claim with its paper value,
status, and reproduce-command, and lets you drill into any evaluation
run (per-instance rows, resolved / cost, aggregates):

```bash
bash benchmarks/explore.sh                # builds the data, serves localhost, opens the browser
```

Each claim maps to a scorer and its evidence files in
[`CLAIMS.md`](CLAIMS.md); the scorers live in
[`benchmarks/`](benchmarks/).

## Examples

- [`examples/quickstart_policy.py`](examples/quickstart_policy.py) - the configuration DSL (presets + custom configs)
- [`examples/deployable_lever.py`](examples/deployable_lever.py) - run the 3-component controller end to end
- [`examples/workload_routing.py`](examples/workload_routing.py) - route by workload class
- [`examples/advanced/custom_quality_gate.py`](examples/advanced/custom_quality_gate.py) - plug in your own quality gate
- [`examples/calibrate_deployment_defaults.py`](examples/calibrate_deployment_defaults.py) - how the shipped presets were calibrated

The cross-vendor HumanEval/MBPP cost study (claim C8) is reproduced offline by
[`benchmarks/cross_vendor/score.py`](benchmarks/cross_vendor/score.py) over the
committed CSVs - see [`CLAIMS.md`](CLAIMS.md).

## Extending

The framework's reusable contribution is its **extension surface**:
eight Protocols you implement against your own domain (the SWE-bench /
HumanEval / MBPP numbers calibrate the shipped defaults; you derive your
own for your workload). The primitives - `WorkloadClassifier`,
`RoutingStrategy`, `Variant`, `Signal`, `QualityGate`, `CascadeTrigger`,
`CostModel`, `ModelClient` - are defined in
[`src/code_capsules/api/protocols.py`](src/code_capsules/api/protocols.py)
and registered by name. Concrete classes need not inherit from anything
(PEP 544 structural typing). See
[`examples/advanced/custom_quality_gate.py`](examples/advanced/custom_quality_gate.py) and
[`examples/workload_routing.py`](examples/workload_routing.py) for
worked extensions.

## Tests

```bash
pytest -m "not integration and not slow and not benchmark"
```

The offline suite uses scripted adapters - no API keys required. Live
evaluation and the SWE-bench Docker path are reserved for separate
benchmarking and are not part of CI.

## Citing

```bibtex
@article{ray2026codecapsules,
  title  = {The Economics of Coding Agents: Calibrating the Cost-Quality Frontier},
  author = {Ray, Aninda},
  year   = {2026},
  note   = {Working paper.},
  url    = {https://github.com/aray-17/code-capsules}
}
```

See also [`CITATION.cff`](CITATION.cff).

## Issues and pull requests

This is a single-maintainer research project. Bug reports are triaged
within ~2 weeks; pull requests are reviewed within ~3 weeks. Best
effort, not SLA. See [CONTRIBUTING.md](CONTRIBUTING.md) for what falls
in scope. Forks are welcome - Apache 2.0 explicitly permits forking and
divergence.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Independent author. Correspondence: research@anindaray.com.
ORCID: [0009-0007-3029-8265](https://orcid.org/0009-0007-3029-8265).

# Contributing to Code Capsules

Code Capsules is a single-maintainer research project. Bug reports
are triaged within ~2 weeks; pull requests are reviewed within
~3 weeks. Best effort, not SLA. Forks are welcome — Apache 2.0
explicitly permits forking and divergence without coordination.

## Scope

### Welcome — please send a PR

- Bug fixes with a reproduction (failing test or minimal script)
- Documentation, typo, and example fixes
- New tests for existing behavior
- Additional model adapters that follow the existing adapter
  contract in `src/code_capsules/adapters/`
- New implementations of an extension primitive (a `WorkloadClassifier`,
  `RoutingStrategy`, `Variant`, `Signal`, `QualityGate`, `CascadeTrigger`,
  `CostModel`, or `ModelClient`) registered against the existing registry

### Open an issue first — wait for acknowledgment before sending a PR

- API changes (anything visible to policy authors)
- New framework features
- Changes that touch the controller, the repro verifier (quality gate),
  the cost governor, or the calibrated policy defaults. These behaviors
  back specific claims in the paper and require careful review.
- New evaluation methodology or new benchmarks

### Out of scope — will be closed

- Changes that would invalidate published claims without a clear
  upgrade path or revised evaluation
- Dependencies on private datasets or paid APIs in the test path
- Production infrastructure unrelated to the framework's programming model

## Tests

PRs must keep the offline test suite green, and the claim-reproduction
gate must stay at 12/12:

```bash
pip install -e ".[dev]"
pytest -m "not integration and not slow and not benchmark"
python3 benchmarks/verify_criteria.py
```

The offline suite uses scripted adapters — no API keys required. Live
evaluation against real models, and the SWE-bench Docker path, are
reserved for separate benchmarking and are not part of CI.

## Merging

Commits and tags on `main` are signed (SSH) and show as **Verified** on
GitHub. To keep `main` fully verified, merge pull requests with **Squash
and merge** or **Create a merge commit** — GitHub signs the resulting
commit automatically. Avoid **Rebase and merge**: it re-creates the
commit without a signature, leaving an unverified commit on `main`.

## Evaluation methodology and operational data

A larger body of operational evaluation work — per-instance stream
archives, overnight harnesses, gap audits, and multi-week eval logs —
is intentionally maintained outside this repository. If your work
depends on understanding *how* the paper's numbers were produced
(rather than verifying *that* they reproduce), contact
**research@anindaray.com**.

## Citing the paper

If your contribution is in the context of academic work, please cite
the paper this framework is described in. See `CLAIMS.md` for the
citation block.

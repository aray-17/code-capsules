"""Regression gate: gold-patch smoke test for the SWE-bench Docker eval.

For every new benchmark integration we should be running the benchmark's own
reference solutions through our eval. If a known-correct solution doesn't
resolve, the eval is broken (not the solution). Running the benchmark's own
reference solutions through the eval is the cheapest guard against this.

These tests are slow (each spins up a Docker container ~20-40s) and have an
external dependency on the SWE-bench Docker images being pullable. They run
under the `slow` marker so a developer can include them on-demand with
`pytest -m slow` but they don't block ordinary CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# These tests are slow + require Docker + require image pulls. Marked so they
# don't block fast CI but can be invoked with `pytest -m slow`.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def harness():
    from tools.run_swebench_docker import load_instances, docker_eval, ensure_image
    return load_instances, docker_eval, ensure_image


def _gold_resolves(harness, start: int) -> bool:
    """Run the gold patch for instance at `start` and assert it resolves."""
    load_instances, docker_eval, ensure_image = harness
    instances = load_instances(start=start, n=1)
    assert len(instances) == 1
    inst = instances[0]
    assert ensure_image(inst["instance_id"], timeout=300), \
        f"Failed to pull image for {inst['instance_id']}"
    res = docker_eval(inst.get("patch", ""), inst, timeout=300)
    assert res.get("resolved") is True, (
        f"Gold patch for {inst['instance_id']} did NOT resolve - eval harness "
        f"is broken. note={res.get('note')} stdout_tail={(res.get('stdout') or '')[-300:]!r}"
    )
    return True


def _gold_resolves_for_repo(harness, repo_prefix: str) -> bool:
    """Run the gold patch for the first instance whose id starts with repo_prefix.

    Robust to dataset ordering (unlike a hardcoded index) - used for repos that
    don't sit at the front of SWE-bench Lite (e.g. sympy lives at index ~150+).
    """
    load_instances, docker_eval, ensure_image = harness
    instances = load_instances(start=0, n=300)
    matches = [i for i in instances if i["instance_id"].startswith(repo_prefix)]
    assert matches, f"no instance with prefix {repo_prefix!r} in SWE-bench Lite"
    inst = matches[0]
    assert ensure_image(inst["instance_id"], timeout=600), \
        f"Failed to pull image for {inst['instance_id']}"
    res = docker_eval(inst.get("patch", ""), inst, timeout=600)
    assert res.get("resolved") is True, (
        f"Gold patch for {inst['instance_id']} did NOT resolve - eval harness "
        f"is broken. note={res.get('note')} stdout_tail={(res.get('stdout') or '')[-300:]!r}"
    )
    return True


def _gold_resolves_by_id(harness, instance_id: str) -> bool:
    """Run the gold patch for an exact instance_id and assert it resolves.

    Pins specific instances (matplotlib/pytest/django) regardless of dataset
    ordering - these three cover the scorer's three code paths (pytest-by-file,
    pytest with a parametrize-heavy suite, and Django runtests-by-exit-code) and
    are the trio the runtime scorer (code_capsules.evaluation.docker_eval) was
    Docker-verified against when it moved out of the private harness.
    """
    load_instances, docker_eval, ensure_image = harness
    instances = load_instances(instance_ids=[instance_id])
    assert len(instances) == 1, f"{instance_id!r} not found in SWE-bench Lite"
    inst = instances[0]
    assert ensure_image(instance_id, timeout=600), \
        f"Failed to pull image for {instance_id}"
    res = docker_eval(inst.get("patch", ""), inst, timeout=600)
    assert res.get("resolved") is True, (
        f"Gold patch for {instance_id} did NOT resolve - eval harness "
        f"is broken. note={res.get('note')} stdout_tail={(res.get('stdout') or '')[-300:]!r}"
    )
    return True


def test_gold_patch_matplotlib_23314_resolves(harness):
    """matplotlib-23314 gold patch must resolve (pytest-by-file path)."""
    _gold_resolves_by_id(harness, "matplotlib__matplotlib-23314")


def test_gold_patch_pytest_11143_resolves(harness):
    """pytest-11143 gold patch must resolve (pytest path, self-hosted suite)."""
    _gold_resolves_by_id(harness, "pytest-dev__pytest-11143")


def test_gold_patch_django_16527_resolves(harness):
    """django-16527 gold patch must resolve (runtests.py exit-code path)."""
    _gold_resolves_by_id(harness, "django__django-16527")


def test_gold_patch_astropy_resolves(harness):
    """Reference patch from an astropy instance must resolve under our eval."""
    _gold_resolves(harness, start=0)  # first 6 are astropy in SWE-bench Lite


def test_gold_patch_django_resolves(harness):
    """Reference patch from a Django instance must resolve under our eval.

    This is the regression guarded here: before the Django scorer fix,
    Django gold patches failed because (a) test_patch wasn't applied and
    (b) `_django_dotted()` produced unimportable dotted labels.
    """
    _gold_resolves(harness, start=6)  # first django instance


def test_gold_patch_sympy_resolves(harness):
    """Reference patch from a sympy instance must resolve under our eval.

    sympy's testbed has no pytest, so a `python -m pytest` path scores every
    sympy patch 0 (`No module named pytest`); sympy must be routed through
    `bin/test -C --verbose` + a verbose-log parser. If this test fails, sympy
    is unscorable.
    """
    _gold_resolves_for_repo(harness, "sympy__")

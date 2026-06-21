"""Gap B parity gate: the self-contained fallback scorer MUST agree with
swebench's canonical grader.

``code_capsules.evaluation.docker_eval`` ships two backends behind one public
``docker_eval``: the canonical one (delegates to swebench's own
``make_test_spec().eval_script`` + ``MAP_REPO_TO_PARSER`` - the grader of record
for the paper's numbers) and a stdlib-only fallback used when swebench is not
installed. The original scorer bug was a hand-rolled grader diverging from
canonical; this test makes that divergence impossible to ship by running BOTH
backends on the SWE-bench reference (gold) patches and asserting they agree.

The gold trio covers the three resolution code paths:
  - matplotlib: pytest, resolve-by-parsed-status (pytest-by-file)
  - pytest-dev/pytest: pytest, parametrize-heavy self-hosted suite
  - django: runtests.py, resolve-by-exit-code
plus a sympy instance (bin/test verbose-log path).

Slow: each case spins up Docker containers (one per backend) and may pull
images. Marked ``slow`` so it doesn't block fast CI; run with ``pytest -m slow``.
Skipped entirely when swebench is not importable (no canonical backend to
compare against).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import code_capsules.evaluation.docker_eval as de

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not de._HAVE_SWEBENCH,
                       reason="swebench not installed -> no canonical backend to compare against"),
]

# Pinned instance ids covering all three resolution code paths + sympy.
PARITY_IDS = [
    "matplotlib__matplotlib-23314",   # pytest-by-file
    "pytest-dev__pytest-11143",       # pytest, parametrize-heavy
    "django__django-16527",           # runtests.py exit-code
    "sympy__sympy-20212",             # bin/test verbose-log
]


def _load(instance_id: str) -> dict:
    from tools.run_swebench_docker import load_instances, ensure_image
    matches = load_instances(instance_ids=[instance_id])
    assert len(matches) == 1, f"{instance_id!r} not found in SWE-bench Lite"
    inst = matches[0]
    assert ensure_image(instance_id, timeout=900), f"could not pull image for {instance_id}"
    return inst


@pytest.mark.parametrize("instance_id", PARITY_IDS)
def test_canonical_and_selfcontained_agree_on_gold(instance_id):
    """Both backends must RESOLVE the gold patch, and must agree with each other."""
    inst = _load(instance_id)
    gold = inst.get("patch", "")
    assert gold.strip(), f"no gold patch for {instance_id}"

    canonical = de._canonical_docker_eval(gold, inst, timeout=1200)
    fallback = de._selfcontained_docker_eval(gold, inst, timeout=1200)

    # The gold patch is the benchmark's own reference fix: it MUST resolve.
    assert canonical["resolved"] is True, (
        f"canonical backend did not resolve gold {instance_id}: "
        f"note={canonical['note']} tail={(canonical['stdout'] or '')[-300:]!r}"
    )
    # Parity: the stdlib fallback must reach the SAME verdict as canonical.
    assert fallback["resolved"] == canonical["resolved"], (
        f"PARITY BREAK on {instance_id}: canonical={canonical['resolved']} "
        f"fallback={fallback['resolved']} (fallback note={fallback['note']})"
    )


def test_both_backends_false_on_nonapplying_patch():
    """A patch that cannot apply is a genuine not-resolved (False) on BOTH
    backends - never None (the container did run) and never a crash."""
    inst = _load(PARITY_IDS[0])
    bogus = (
        "diff --git a/this_path_does_not_exist.py b/this_path_does_not_exist.py\n"
        "--- a/this_path_does_not_exist.py\n"
        "+++ b/this_path_does_not_exist.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-nonexistent original line\n"
        "+nonexistent replacement line\n"
    )
    canonical = de._canonical_docker_eval(bogus, inst, timeout=600)
    fallback = de._selfcontained_docker_eval(bogus, inst, timeout=600)
    assert canonical["resolved"] is False, f"canonical: {canonical['note']}"
    assert fallback["resolved"] is False, f"fallback: {fallback['note']}"

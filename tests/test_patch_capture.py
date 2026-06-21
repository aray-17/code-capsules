"""Patch archival is the verify-without-rerun guarantee.

A scorer change must be checkable by a MODEL-FREE re-score of the stored patch
(see benchmarks/swebench/rescore_from_patch.py). That only works if every eval
row carries the model's git diff. These gold-free tests lock that in:

  1. run_result_to_legacy_jsonl persists RunResult.patch as `model_patch`.
  2. the re-score utility auto-detects the patch field and refuses (loudly) to
     pretend it can re-score rows that carry only a `has_patch` boolean.
"""
from code_capsules.api.types import RunResult
from code_capsules.evaluation.swe_bench_adapter import run_result_to_legacy_jsonl
from benchmarks.swebench.rescore_from_patch import _detect_patch_field

import pytest

PATCH = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y\n"
INSTANCE = {"instance_id": "astropy__astropy-1", "repo": "astropy/astropy"}


def test_legacy_jsonl_persists_model_patch():
    rr = RunResult(task_id="astropy__astropy-1", resolved=False, patch=PATCH)
    row = run_result_to_legacy_jsonl(rr, INSTANCE, mode="sequential")
    assert row["model_patch"] == PATCH, "model_patch must carry the diff for offline re-score"
    assert row["has_patch"] is True


def test_legacy_jsonl_empty_patch_is_none():
    rr = RunResult(task_id="astropy__astropy-1", resolved=None, patch="")
    row = run_result_to_legacy_jsonl(rr, INSTANCE, mode="sequential")
    assert row["model_patch"] is None
    assert row["has_patch"] is False


def test_detect_patch_field_prefers_model_patch():
    rows = [{"model_patch": PATCH, "h2h_patch": "other"}]
    assert _detect_patch_field(rows, None) == "model_patch"


def test_detect_patch_field_falls_back_to_h2h():
    rows = [{"h2h_patch": PATCH}]
    assert _detect_patch_field(rows, None) == "h2h_patch"


def test_detect_patch_field_refuses_boolean_only_rows():
    # rows that recorded only has_patch (the pre-fix CC failure mode) cannot be
    # re-scored offline -- the utility must error, not silently no-op.
    rows = [{"has_patch": True, "resolved": False}]
    with pytest.raises(SystemExit):
        _detect_patch_field(rows, None)

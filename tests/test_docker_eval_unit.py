"""Fast, gold-free unit tests for the runtime SWE-bench scorer's log/label parsers.

These exercise the two most error-prone, pure-Python pieces of
``code_capsules.evaluation.docker_eval`` WITHOUT Docker:

  - ``_parse_log_pytest``: turns a pytest ``-rA`` summary into a
    {nodeid: STATUS} map (resolution is decided from this map, NOT the exit code).
  - ``_django_dotted``: turns a SWE-bench Django FAIL_TO_PASS id into the
    canonical runtests.py dotted label, handling both id formats without
    doubling the method.

No Docker, no network, no `slow` marker - these run in ordinary CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import code_capsules.evaluation.docker_eval as de
from code_capsules.evaluation.docker_eval import (
    _parse_log_pytest,
    _django_dotted,
    _selfcontained_docker_eval,
    _RAN_SENTINEL,
)


class _FakeProc:
    """Stand-in for subprocess.CompletedProcess (no Docker)."""

    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def test_parse_log_pytest_basic_statuses():
    """PASSED/FAILED lines, parametrize ids, and the ' - <msg>' tail on FAILED."""
    log = "\n".join(
        [
            "some preamble that should be ignored",
            "PASSED a.py::t1",
            "FAILED a.py::t2 - AssertionError",
            "PASSED a.py::t3[x-y]",
            "trailing noise",
        ]
    )
    status = _parse_log_pytest(log)
    assert status == {
        "a.py::t1": "PASSED",
        "a.py::t2": "FAILED",      # ' - AssertionError' stripped, only the nodeid kept
        "a.py::t3[x-y]": "PASSED", # parametrize node id preserved verbatim
    }


def test_django_dotted_two_part_class_form():
    """'method (mod.Class)' -> 'mod.Class.method'."""
    assert _django_dotted("m (mod.Class)") == "mod.Class.m"


def test_django_dotted_method_already_in_paren_not_doubled():
    """'method (mod.Class.method)' -> 'mod.Class.method' (method NOT appended twice)."""
    assert _django_dotted("m (mod.Class.m)") == "mod.Class.m"


def test_django_dotted_malformed_returns_none():
    """A docstring-style label with no paren is skipped (returns None)."""
    assert _django_dotted("Regression for #9362") is None


# ── Gap A: never grade a run that did not execute ────────────────────────────
# A failed image pull / daemon error means the eval never ran. That MUST surface
# as resolved=None (retryable), NOT a false-negative False. This was the bug
# behind the 2026-06 autotrigger misfire. These exercise the self-contained
# backend's run-sentinel gate by monkeypatching subprocess (no Docker).

_PYTEST_INSTANCE = {
    "instance_id": "foo__bar-1",
    "repo": "foo/bar",
    "FAIL_TO_PASS": ["t/test_x.py::test_a"],
    "PASS_TO_PASS": ["t/test_x.py::test_b"],
}


def test_selfcontained_none_when_container_did_not_run(monkeypatch):
    """Sentinel absent (image pull failed) -> resolved is None, not False."""
    def fake_run(*a, **k):
        return _FakeProc(
            "Unable to find image 'swebench/...' locally\n"
            "docker: manifest unknown: manifest unknown.\n",
            returncode=125,
        )
    monkeypatch.setattr(de.subprocess, "run", fake_run)
    res = _selfcontained_docker_eval("diff --git a/x b/x\n", _PYTEST_INSTANCE, timeout=10)
    assert res["resolved"] is None
    assert res["note"].startswith("eval-did-not-run")


def test_selfcontained_resolves_when_all_targets_pass(monkeypatch):
    """Sentinel present + every target PASSED -> resolved True."""
    def fake_run(*a, **k):
        out = f"{_RAN_SENTINEL}\nPASSED t/test_x.py::test_a\nPASSED t/test_x.py::test_b\n"
        return _FakeProc(out, returncode=0)
    monkeypatch.setattr(de.subprocess, "run", fake_run)
    res = _selfcontained_docker_eval("diff --git a/x b/x\n", _PYTEST_INSTANCE, timeout=10)
    assert res["resolved"] is True
    assert res["note"] == "docker"


def test_selfcontained_false_when_target_missing_but_ran(monkeypatch):
    """Sentinel present but a target never reported -> False (genuinely not
    resolved), crucially NOT None: the container DID run."""
    def fake_run(*a, **k):
        out = f"{_RAN_SENTINEL}\nPASSED t/test_x.py::test_b\n"  # test_a absent
        return _FakeProc(out, returncode=1)
    monkeypatch.setattr(de.subprocess, "run", fake_run)
    res = _selfcontained_docker_eval("diff --git a/x b/x\n", _PYTEST_INSTANCE, timeout=10)
    assert res["resolved"] is False


def test_selfcontained_empty_patch_is_false_not_none():
    """An empty patch is a genuine not-resolved (no eval attempted)."""
    res = _selfcontained_docker_eval("   ", _PYTEST_INSTANCE, timeout=10)
    assert res["resolved"] is False
    assert res["note"] == "empty patch"

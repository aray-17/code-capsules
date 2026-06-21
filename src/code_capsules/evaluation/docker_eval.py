"""Canonical SWE-bench Docker scorer.

This module is the single source of truth for grading a model patch against a
SWE-bench Lite instance: it applies the patch (plus the instance's own
``test_patch``) inside the official SWE-bench Docker container, runs the
FAIL_TO_PASS / PASS_TO_PASS targets, and reports whether the instance is
resolved.

It is shipped inside the runtime (rather than the private benchmarking harness)
so the public package can score and evaluate without the private ``tools/``
harness. The runtime adapters (``swe_bench_adapter.make_docker_eval_gate`` and
``quality_gates_adaptors.DockerEvalGate``) default to this implementation; the
private harness re-exports ``docker_eval`` from here so its CLI keeps working.

Two backends, one public entry point (``docker_eval``):

  1. CANONICAL (preferred, used whenever ``swebench`` is importable). Delegates
     grading to swebench's OWN machinery: ``make_test_spec().eval_script`` builds
     the canonical test command (reset test files, apply test_patch, run the
     repo's canonical runner) and ``MAP_REPO_TO_PARSER`` parses the result. This
     is the EXACT grader used to produce this project's published numbers, so the
     public package reproduces the paper's verdicts bit-for-bit. swebench is an
     OPTIONAL dependency — import it only if you score SWE-bench.

  2. SELF-CONTAINED FALLBACK (used only when swebench is not installed). A
     stdlib-only reimplementation of the same resolution semantics, kept so the
     package can score without pulling in swebench. ``tests/test_docker_eval_parity.py``
     gates it against the canonical backend on gold patches so the two cannot
     silently drift (a hand-rolled grader diverging from canonical was the
     original scorer bug).

Either backend NEVER grades a run that did not actually execute: a failed image
pull / daemon error means the eval never ran, which MUST surface as ``resolved:
None`` (retryable), never a false-negative ``False``. (Gap that caused the
2026-06 autotrigger misfire.)

This module must NOT import from ``tools/`` (swebench, an external package, is
fine). Both backends return the same dict shape:
``{"resolved": bool|None, "returncode": int, "stdout": str, "note": str}``.

Resolution semantics (do not change without re-verifying against Docker):
  - pytest repos: resolve from the parsed per-test status map (-rA summary),
    NOT pytest's exit code (pytest exits non-zero whenever ANY collected test
    fails, but only the FAIL_TO_PASS / PASS_TO_PASS targets matter).
  - Django / unittest repos: run runtests.py with canonical dotted labels;
    resolution is by exit code (correct labels give a reliable code).
  - sympy: no pytest in the testbed; run ``bin/test -C --verbose`` and parse
    per-test status from the verbose log.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile

# swebench's own grading machinery. OPTIONAL: present for anyone scoring
# SWE-bench, absent for a HumanEval/MBPP-only install. When present, docker_eval
# delegates grading to it (the canonical, paper-grade path); when absent, the
# self-contained stdlib fallback below is used. Guarded so merely importing this
# module never fails for a swebench-less install.
try:
    from swebench.harness.test_spec import make_test_spec as _make_test_spec
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER as _MAP_REPO_TO_PARSER
    from swebench.harness.constants import TestStatus as _TestStatus
    _HAVE_SWEBENCH = True
except Exception:  # ImportError, or any swebench import-time error
    _make_test_spec = None
    _MAP_REPO_TO_PARSER = None
    _TestStatus = None
    _HAVE_SWEBENCH = False

# Printed as the FIRST line inside the container by the self-contained backend.
# Its presence in the output is the reliable "the eval actually ran" signal: a
# failed image pull / daemon error means the container never executed the command
# and the sentinel is absent -> return None (retryable) instead of a false
# negative. (The canonical backend uses git-apply markers for the same purpose.)
_RAN_SENTINEL = "__CC_EVAL_RAN__"


def _image_name(instance_id: str) -> str:
    img_id = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{img_id}:latest"


def _is_sympy(instance: dict) -> bool:
    repo = instance.get("repo", "") or ""
    return repo == "sympy/sympy" or instance.get("instance_id", "").startswith("sympy__")


# sympy's testbed has NO pytest; the canonical SWE-bench spec runs it via
# `bin/test -C --verbose` and parses per-test status from the verbose log
# (TEST_SYMPY in swebench constants; parse_log_sympy in swebench log_parsers).
# Our docker_eval otherwise decides resolution by exit code, which is wrong for
# sympy because bin/test runs the whole file and exits non-zero if ANY test in
# it fails (not just the FAIL_TO_PASS targets). So sympy needs output parsing.
SYMPY_TIMEOUT = 600  # sympy suites are slow; 180s timed out ~14% of instances


def _parse_log_sympy(log: str) -> dict:
    """Map test-name -> 'PASSED'|'FAILED'|'ERROR' from sympy's bin/test verbose log.

    Faithful port of swebench.harness.log_parsers.parse_log_sympy: verbose lines
    look like `test_name ... ok` / ` F` / ` E`; failures also emit a
    `____ path.py:test ____` separator banner.
    """
    status: dict[str, str] = {}
    for m in re.findall(r"(_*) (.*)\.py:(.*) (_*)", log):
        status[f"{m[1]}.py:{m[2]}"] = "FAILED"
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                status[line.split()[0]] = "ERROR"
            elif line.endswith(" F"):
                status[line.split()[0]] = "FAILED"
            elif line.endswith(" ok"):
                status[line.split()[0]] = "PASSED"
    return status


def _parse_log_pytest(log: str) -> dict:
    """Map nodeid -> 'PASSED'|'FAILED'|'ERROR'|'SKIPPED' from pytest -rA output.

    Faithful port of swebench.harness.log_parsers.parse_log_pytest. The -rA
    summary section emits one line per test starting with the status word
    followed by the nodeid (FAILED lines use ' - ' before the message, which we
    strip). Resolution is decided from this map, NOT from pytest's exit code:
    pytest exits non-zero whenever ANY collected test fails, but we only care
    about the FAIL_TO_PASS and PASS_TO_PASS targets.
    """
    status: dict[str, str] = {}
    for line in log.split("\n"):
        if line.startswith(("PASSED", "FAILED", "ERROR", "SKIPPED")):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                status[parts[1]] = parts[0]
    return status


def _pytest_test_files(*id_lists: list[str]) -> list[str]:
    """Unique pytest test files (the path before '::') across the given lists."""
    files: list[str] = []
    for ids in id_lists:
        for tid in ids:
            f = tid.split("::")[0]
            if f and f not in files:
                files.append(f)
    return files


def _sympy_test_files(test_patch: str) -> list[str]:
    """Test files for bin/test, taken from the SWE-bench test_patch diff headers."""
    files: list[str] = []
    for _a, b in re.findall(r"^diff --git a/(\S+) b/(\S+)", test_patch, re.M):
        if b not in files:
            files.append(b)
    return files


def _django_dotted(tid: str) -> str | None:
    """Canonical Django runtests dotted label from a SWE-bench FAIL_TO_PASS id.

    Module-level (and unit-tested) because Django label construction is the most
    error-prone part of the scorer. Handles BOTH SWE-bench id formats:
      - "method (mod.Class)"          -> "mod.Class.method"
      - "method (mod.Class.method)"   -> "mod.Class.method"  (NOT doubled)
    """
    if " (" not in tid:
        return None   # malformed / docstring - skip
    method, cls_path = tid.split(" (", 1)
    cls_path = cls_path.rstrip(")")
    # Use the FULL dotted path: app.[tests_dir.]test_module.Class.method
    # SWE-bench Lite FAIL_TO_PASS for Django is full-dotted form, e.g.
    #   "test_X (forms_tests.tests.test_media.FormsMediaTestCase)"
    # Previously this code collapsed to 3-part `app.Class.method` to
    # work around namespace-package import issues, but that broke import
    # for the common case where the test lives in `app/tests/test_*.py`
    # - Django's `runtests.py` then tries to interpret the second
    # component as a submodule name and ImportError-s. Full dotted path
    # is the canonical form Django's test loader accepts.
    # Newer SWE-bench django ids already embed the method inside the
    # paren: "test_X (module.Class.test_X)". In that case cls_path is
    # the complete dotted path; appending .method again would yield
    # ...Class.test_X.test_X and runtests would report a missing
    # attribute. Detect this and return cls_path as-is.
    if cls_path.endswith(f".{method}"):
        return cls_path
    return f"{cls_path}.{method}"


# ── Public entry point ───────────────────────────────────────────────────────

def docker_eval(patch: str, instance: dict, timeout: int = 180) -> dict:
    """Grade a model patch against a SWE-bench Lite instance inside Docker.

    Delegates to swebench's canonical grader when ``swebench`` is installed (the
    exact, paper-grade path), otherwise to the self-contained stdlib fallback.
    Container is removed automatically via --rm. Returns
    ``{"resolved": bool|None, "returncode": int, "stdout": str, "note": str}``.
    ``resolved is None`` means the eval did not run (image/daemon unavailable, or
    timeout) and is RETRYABLE - never treat it as a not-resolved.
    """
    if not (patch or "").strip():
        return {"resolved": False, "returncode": -1, "stdout": "", "note": "empty patch"}
    if _is_sympy(instance):
        timeout = max(timeout, SYMPY_TIMEOUT)
    if _HAVE_SWEBENCH:
        return _canonical_docker_eval(patch, instance, timeout=timeout)
    return _selfcontained_docker_eval(patch, instance, timeout=timeout)


# ── Canonical backend (swebench's own eval_script + parser) ──────────────────

def _canonical_grade(repo: str, log: str, f2p, p2p):
    """Canonical grading: parse the test log with swebench's OWN parser, then
    RESOLVED iff every FAIL_TO_PASS and PASS_TO_PASS target is PASSED."""
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    parser = _MAP_REPO_TO_PARSER[repo]
    try:
        sm = parser(log)
    except TypeError:
        sm = parser(log, None)  # some parsers take (log, test_spec)
    passed = _TestStatus.PASSED.value
    if not f2p:
        return None
    ok = lambda t: sm.get(t) == passed
    return bool(all(ok(t) for t in f2p) and all(ok(t) for t in p2p))


def _canonical_docker_eval(patch: str, instance: dict, timeout: int = 180) -> dict:
    """Score via swebench's own eval_script on the prebuilt x86_64 image.

    Mirrors run_evaluation's flow: image at base -> git apply the MODEL patch ->
    run swebench's eval_script (which resets test files, applies the test_patch,
    and runs the canonical per-repo test command) -> grade with the repo's
    canonical parser. This is the grader of record for this project's numbers.
    """
    iid = instance["instance_id"]
    if not (patch or "").strip():
        return {"resolved": False, "returncode": -1, "stdout": "", "note": "empty patch"}
    try:
        spec = _make_test_spec(instance)
    except Exception as exc:
        return {"resolved": None, "returncode": -1, "stdout": "",
                "note": f"make_test_spec failed (instance missing fields?): {type(exc).__name__}: {exc}"}
    eval_script = spec.eval_script

    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as pf:
        pf.write(patch)
        ppath = pf.name
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as sf:
        sf.write(eval_script)
        spath = sf.name
    try:
        # git apply -v prints "Checking patch ..." / "Applied patch ..." which is
        # our reliable "the container ran" signal (see the `ran` gate below).
        cmd = ("cd /testbed && "
               "(git apply -v /tmp/model.patch || git apply -v --3way /tmp/model.patch || echo APPLY_FAIL) ; "
               "/bin/bash /tmp/eval.sh")  # /tmp/eval.sh is mounted :ro; bash runs it regardless
        r = subprocess.run(
            ["docker", "run", "--rm", "--platform", "linux/amd64",
             "-v", f"{ppath}:/tmp/model.patch:ro", "-v", f"{spath}:/tmp/eval.sh:ro",
             _image_name(iid), "bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout)
        log = r.stdout + r.stderr
        rc = r.returncode
    except subprocess.TimeoutExpired:
        return {"resolved": None, "returncode": -1, "stdout": "", "note": f"docker timeout {timeout}s"}
    except Exception as exc:
        return {"resolved": None, "returncode": -1, "stdout": "", "note": str(exc)}
    finally:
        for p in (ppath, spath):
            try:
                os.unlink(p)
            except OSError:
                pass

    # Never grade a run that did not execute. The "did it run" signal is the
    # `git apply -v` output ("Checking patch"/"Applied patch"/"APPLY_FAIL"). We
    # gate on THAT, not on docker pull noise: `docker run` auto-pulls a missing
    # image and prints "Unable to find image '...' locally" BEFORE running
    # successfully, so matching that string would false-negative every
    # post-prune first run. Apply markers present => the eval ran => grade it.
    ran = ("Checking patch" in log or "Applied patch" in log or "APPLY_FAIL" in log)
    if not ran:
        return {"resolved": None, "returncode": rc, "stdout": log[-3000:],
                "note": "eval-did-not-run: image/daemon unavailable or no apply markers (retry)"}
    if "APPLY_FAIL" in log:
        return {"resolved": False, "returncode": rc, "stdout": log[-3000:], "note": "model patch did not apply"}
    try:
        resolved = _canonical_grade(instance["repo"], log,
                                    instance.get("FAIL_TO_PASS"), instance.get("PASS_TO_PASS"))
        if resolved is None:
            return {"resolved": None, "returncode": rc, "stdout": log[-3000:], "note": "no FAIL_TO_PASS"}
        return {"resolved": resolved, "returncode": rc, "stdout": log[-3000:], "note": "canonical"}
    except Exception as exc:
        return {"resolved": None, "returncode": rc, "stdout": log[-3000:],
                "note": f"grade-error: {type(exc).__name__}: {exc}"}


# ── Self-contained backend (stdlib only; used when swebench is absent) ────────

def _selfcontained_docker_eval(patch: str, instance: dict, timeout: int = 180) -> dict:
    """
    Apply patch inside the SWE-bench Docker container and run tests, with no
    dependency on swebench (stdlib only). Resolution semantics match the canonical
    backend; tests/test_docker_eval_parity.py gates the two against each other.
    Container is removed automatically via --rm.
    Returns {"resolved": bool|None, "returncode": int, "stdout": str, "note": str}.
    """
    if not patch.strip():
        return {"resolved": False, "returncode": -1, "stdout": "", "note": "empty patch"}

    fail_tests: list[str] = instance.get("FAIL_TO_PASS", [])
    pass_tests: list[str] = instance.get("PASS_TO_PASS", [])
    if isinstance(fail_tests, str):
        fail_tests = json.loads(fail_tests)
    if isinstance(pass_tests, str):
        pass_tests = json.loads(pass_tests)

    if not fail_tests:
        return {"resolved": None, "returncode": -1, "stdout": "", "note": "no FAIL_TO_PASS tests"}

    if _is_sympy(instance):
        timeout = max(timeout, SYMPY_TIMEOUT)

    img = _image_name(instance["instance_id"])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
        f.write(patch)
        patch_path = f.name

    # Apply the SWE-bench `test_patch` BEFORE the model's patch. This updates
    # the test files to the post-bugfix expectations the FAIL_TO_PASS list
    # was authored against. Skipping this step makes correct model fixes
    # appear to fail (old test code asserts old-buggy behavior).
    test_patch = instance.get("test_patch", "") or ""
    test_patch_path = None
    if test_patch.strip():
        with tempfile.NamedTemporaryFile(mode="w", suffix="_test.diff", delete=False) as tf:
            tf.write(test_patch)
            test_patch_path = tf.name

    try:
        # Use the explicit testbed Python path - source activate testbed does not
        # work in non-interactive bash -c shells (PATH not updated).
        _py = "/opt/miniconda3/envs/testbed/bin/python"

        def _is_unittest_style(ids: list[str]) -> bool:
            """Detect Django/unittest-style IDs.

            Django uses 'method (module.Class)'. PASS_TO_PASS often mixes
            this format with free-form docstring labels like 'Regression for
            #9362'. If ANY id is Django-format, treat the whole list as
            Django (use runtests.py). pytest-format uses '::' separators.
            """
            if not ids:
                return False
            if instance.get("repo", "").startswith("django/"):
                return True
            return any(" (" in t for t in ids)

        def _django_cmd(ids: list[str]) -> str:
            # Django/unittest: run runtests.py with the canonical dotted labels.
            # We run ALL labels (FAIL_TO_PASS + PASS_TO_PASS) together; with
            # correct labels runtests gives a reliable exit code, so resolution
            # is by exit code (unlike the pytest path).
            if not ids:
                return ""
            dotted_list = [_django_dotted(t) for t in ids]
            dotted = " ".join(d for d in dotted_list if d)
            return f"cd /testbed/tests && {_py} runtests.py {dotted} --verbosity=0 2>&1"

        sympy_eval = _is_sympy(instance)
        targets = list(fail_tests) + list(pass_tests)
        django_eval = (not sympy_eval) and _is_unittest_style(targets)
        if sympy_eval:
            # sympy: run `bin/test -C --verbose <test files>` and parse per-test
            # status from the verbose log (resolution is NOT by exit code - see
            # _parse_log_sympy). Test files come from the test_patch headers.
            test_files = _sympy_test_files(test_patch)
            if not test_files:
                return {"resolved": None, "returncode": -1, "stdout": "",
                        "note": "sympy: no test files in test_patch"}
            parts = ["cd /testbed"]
            if test_patch_path is not None:
                parts.append("git apply /tmp/test_patch.diff 2>&1")
            parts.append("git apply /tmp/patch.diff 2>&1")
            parts.append(
                "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' "
                f"{_py} bin/test -C --verbose {' '.join(test_files)} 2>&1"
            )
            full_cmd = " && ".join(parts)
        elif django_eval:
            parts = ["cd /testbed"]
            if test_patch_path is not None:
                # Apply SWE-bench's test_patch first, then the model's patch.
                parts.append("git apply /tmp/test_patch.diff 2>&1")
            parts.append("git apply /tmp/patch.diff 2>&1")
            parts.append(_django_cmd(targets))
            full_cmd = " && ".join(p for p in parts if p)
        else:
            # pytest repos: do NOT pass individual node IDs (pytest cannot
            # reliably match some exact parametrize node-id strings, which
            # produced "no tests ran"/non-zero exit -> false negatives). Run one
            # pytest over the unique test FILES with -rA and parse per-test
            # status from the summary; resolution is by parsed status, NOT exit
            # code. The two `git apply` steps stay `&&`-before so a failed
            # patch-apply yields empty pytest output -> unresolved.
            test_files = _pytest_test_files(fail_tests, pass_tests)
            parts = ["cd /testbed"]
            if test_patch_path is not None:
                parts.append("git apply /tmp/test_patch.diff 2>&1")
            parts.append("git apply /tmp/patch.diff 2>&1")
            quoted = " ".join(f'"{f}"' for f in test_files)
            parts.append(
                f"{_py} -m pytest {quoted} -rA --tb=no -p no:cacheprovider 2>&1"
            )
            full_cmd = " && ".join(p for p in parts if p)

        # Echo the run sentinel FIRST so its presence in the output proves the
        # container executed (a failed image pull never prints it). `;` keeps the
        # downstream chain's exit code intact (matters for the django path, which
        # resolves by exit code).
        full_cmd = f"echo {_RAN_SENTINEL} ; " + full_cmd

        docker_mounts = ["-v", f"{patch_path}:/tmp/patch.diff:ro"]
        if test_patch_path is not None:
            docker_mounts.extend(["-v", f"{test_patch_path}:/tmp/test_patch.diff:ro"])

        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--platform", "linux/amd64",
                *docker_mounts,
                img,
                "bash", "-c", full_cmd,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        out = result.stdout + result.stderr

        # Never grade a run that did not execute. If the sentinel is absent the
        # container never ran (image pull failed / daemon error) -> None
        # (retryable), NOT a false-negative False.
        if _RAN_SENTINEL not in out:
            return {"resolved": None, "returncode": result.returncode, "stdout": out[-3000:],
                    "note": "eval-did-not-run: image/daemon unavailable (retry)"}

        if sympy_eval:
            # Resolved iff every FAIL_TO_PASS and PASS_TO_PASS target is PASSED in
            # the parsed log. A target that never appears (e.g. patch failed to
            # apply, or an import error suppressed the suite) counts as not-passed.
            status = _parse_log_sympy(out)
            resolved = bool(targets) and all(status.get(t) == "PASSED" for t in targets)
            note = "docker(sympy)"
        elif django_eval:
            # runtests.py with correct labels gives a reliable exit code.
            resolved = result.returncode == 0
            note = "docker(django)"
        else:
            # pytest repos: resolve from the parsed per-test status map, NOT the
            # exit code. Resolved iff every FAIL_TO_PASS and PASS_TO_PASS target
            # is PASSED. A target missing from the map (collection error, failed
            # patch-apply, parametrize-id mismatch) counts as not-passed.
            status = _parse_log_pytest(out)
            resolved = bool(targets) and all(status.get(t) == "PASSED" for t in targets)
            note = "docker"
        return {
            "resolved": resolved,
            "returncode": result.returncode,
            "stdout": out[-3000:],
            "note": note,
        }
    except subprocess.TimeoutExpired:
        return {"resolved": None, "returncode": -1, "stdout": "", "note": f"docker timeout {timeout}s"}
    except Exception as exc:
        return {"resolved": None, "returncode": -1, "stdout": "", "note": str(exc)}
    finally:
        if test_patch_path is not None:
            try:
                os.unlink(test_patch_path)
            except OSError:
                pass
        try:
            os.unlink(patch_path)
        except OSError:
            pass


__all__ = [
    "docker_eval",
    "_canonical_docker_eval",
    "_selfcontained_docker_eval",
    "_canonical_grade",
    "_HAVE_SWEBENCH",
    "_RAN_SENTINEL",
    "_image_name",
    "_is_sympy",
    "SYMPY_TIMEOUT",
    "_parse_log_sympy",
    "_parse_log_pytest",
    "_pytest_test_files",
    "_sympy_test_files",
    "_django_dotted",
]

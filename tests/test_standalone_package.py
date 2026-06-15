"""The package must be importable and usable with tools/ and hooks/ ABSENT.

This is the standalone-package regression gate. It runs in a subprocess whose path
contains only `src` (the repo-root `tools/` and `hooks/` are NOT importable), so any
re-coupling of the framework to the private harness fails here. Mirrors the manual
acceptance check used during the standalone refactor.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_IMPORT_WALK = """
import importlib, pkgutil, sys
import code_capsules
fails = []
for m in pkgutil.walk_packages(code_capsules.__path__, "code_capsules."):
    try:
        importlib.import_module(m.name)
    except Exception as e:           # noqa: BLE001
        fails.append((m.name, repr(e)))
assert not fails, f"submodules failed to import standalone: {fails}"
leaked = [m for m in sys.modules
          if m == "tools" or m.startswith("tools.") or m == "hooks" or m.startswith("hooks.")]
assert not leaked, f"package imported tools/hooks: {leaked}"
print("IMPORT-WALK OK")
"""

_CALL_TIME = """
import sys
import code_capsules
from code_capsules.controller.runtime import CodeCapsulesRunner, RunnerPolicy

# default p_source -> None; injected p_source -> used directly; neither imports tools/
assert CodeCapsulesRunner(RunnerPolicy())._build_p_source() is None
class _StubP:
    def p_resolve(self, evidence): return 0.5
assert CodeCapsulesRunner(RunnerPolicy(), p_source=_StubP())._build_p_source().p_resolve({}) == 0.5

# make_docker_eval_gate with an injected evaluator -> no tools import
from code_capsules.evaluation.swe_bench_adapter import make_docker_eval_gate
gate = make_docker_eval_gate(docker_eval=lambda patch, instance, timeout: {"resolved": False})
assert gate is not None

# the stream parser the adapters use lives in the package
import code_capsules.runtime.stream_parser as sp
assert hasattr(sp, "parse_stream") and hasattr(sp, "ToolCallRecord")

leaked = [m for m in sys.modules
          if m == "tools" or m.startswith("tools.") or m == "hooks" or m.startswith("hooks.")]
assert not leaked, f"call-time path imported tools/hooks: {leaked}"
print("CALL-TIME OK")
"""


def _run_isolated(script: str, tmp_path: Path) -> subprocess.CompletedProcess:
    """Run `script` with only the package src on the path, cwd outside the repo,
    so tools/ and hooks/ at the repo root are not importable."""
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path),
                          env=env, capture_output=True, text=True)


def test_package_imports_without_tools_or_hooks(tmp_path):
    r = _run_isolated(_IMPORT_WALK, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "IMPORT-WALK OK" in r.stdout


def test_entry_points_work_without_tools_or_hooks(tmp_path):
    r = _run_isolated(_CALL_TIME, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CALL-TIME OK" in r.stdout


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for fn in (test_package_imports_without_tools_or_hooks, test_entry_points_work_without_tools_or_hooks):
            fn(Path(d)); print(f"{fn.__name__}: PASS")

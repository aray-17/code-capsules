"""Live end-to-end smoke test of the shipped ModelClient adapters.

Exercises the FULL real path for each vendor - registry lookup -> adapter
`invoke()` -> live API call -> multi-turn tool loop (Read/Write/Edit/Bash in a
sandboxed worktree) -> InvocationResult - by handing each provider a tiny but
genuine coding task (a one-line bug to fix) and grading the result the way a
deployment would: re-import the patched module in a fresh subprocess and check
behaviour. No mocks, no replay.

Providers (the three the framework ships adapters for):
    anthropic  -> claude_cli   (shells out to `claude -p`; cheap model: haiku)
    oai        -> openai_api    (Chat Completions tool loop; gpt-4o)
    google     -> gemini_api    (google-genai tool loop; gemini-2.5-flash)

A provider is SKIPPED (not failed) when its API key is absent or its vendor SDK
is not importable. Marked `integration` - costs real tokens; run explicitly:

    pytest tests/test_e2e_providers.py -m integration -v
    # or, for a standalone report with per-provider telemetry:
    python tests/test_e2e_providers.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ── The task each provider must solve ────────────────────────────────────────
# `add` subtracts instead of adding - a single-character bug a competent agent
# fixes in one edit. The accompanying test lets a diligent agent self-verify.
_BUGGY = "def add(a, b):\n    return a - b  # BUG: should add\n"
_TEST = (
    "from buggy import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
    "    assert add(10, 5) == 15\n"
)
_TASK = (
    "The function `add(a, b)` in buggy.py is supposed to return the sum of its "
    "two arguments, but it has a bug. Read buggy.py, fix the bug so `add` "
    "returns a + b, then run `python -m pytest test_buggy.py` to confirm the "
    "test passes. Edit only buggy.py."
)

# label -> (registry client name, model, max_turns)
PROVIDERS: dict[str, tuple[str, str, int]] = {
    "anthropic": ("claude_cli", "haiku", 12),
    "oai":       ("openai_api", "gpt-4o", 12),
    "google":    ("gemini_api", "gemini-2.5-flash", 12),
}


# ── Availability gating ──────────────────────────────────────────────────────
def _importable(module: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _availability(label: str) -> str | None:
    """Return a skip reason if the provider can't run, else None."""
    if label == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return "ANTHROPIC_API_KEY not set"
        from shutil import which
        if which("claude") is None:
            return "`claude` CLI not on PATH"
        return None
    if label == "oai":
        if not os.environ.get("OPENAI_API_KEY"):
            return "OPENAI_API_KEY not set"
        if not _importable("openai"):
            return "openai SDK not installed"
        return None
    if label == "google":
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            return "GOOGLE_API_KEY / GEMINI_API_KEY not set"
        if not _importable("google.genai"):
            return "google-genai SDK not installed"
        return None
    return f"unknown provider {label!r}"


def _make_repo(tmp: Path) -> Path:
    """Create a throwaway git repo holding the buggy module + its test."""
    (tmp / "buggy.py").write_text(_BUGGY)
    (tmp / "test_buggy.py").write_text(_TEST)
    subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp, check=True)
    subprocess.run(
        ["git", "-c", "user.email=e2e@test", "-c", "user.name=e2e",
         "commit", "-qm", "seed"],
        cwd=tmp, check=True,
    )
    return tmp


def _grade(repo: Path) -> tuple[bool, str]:
    """Independent ground-truth grader: re-import the patched module fresh and
    check behaviour. Does NOT trust whatever the agent reported."""
    check = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); from buggy import add; "
         "assert add(2, 3) == 5, add(2, 3); assert add(10, 5) == 15, add(10, 5); "
         "print('GRADE_OK')"],
        cwd=repo, capture_output=True, text=True,
    )
    ok = check.returncode == 0 and "GRADE_OK" in check.stdout
    return ok, (check.stdout + check.stderr).strip()


def run_one(label: str, tmp: Path) -> dict:
    """Run a single provider end-to-end. Returns a structured result dict."""
    from code_capsules.api.registry import get

    client_name, model, max_turns = PROVIDERS[label]
    repo = _make_repo(tmp)
    client = get("model_client", client_name)

    result = client.invoke(_TASK, cwd=repo, max_turns=max_turns,
                            timeout=240, model=model)

    graded_ok, grade_detail = _grade(repo)
    diff = subprocess.run(["git", "diff", "--stat"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()
    n_tool_calls = len(result.tool_calls or [])
    return {
        "label": label,
        "client": client_name,
        "model": result.model or model,
        "error": result.error,
        "num_turns": result.num_turns,
        "n_tool_calls": n_tool_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "duration_ms": result.duration_ms,
        "graded_ok": graded_ok,
        "grade_detail": grade_detail,
        "diffstat": diff,
        # PASS iff: no transport error, the model actually used tools, and the
        # independent grader confirms the bug is fixed.
        "passed": (result.error is None and n_tool_calls > 0 and graded_ok),
    }


# ── pytest entry point ───────────────────────────────────────────────────────
@pytest.mark.integration
@pytest.mark.parametrize("label", list(PROVIDERS))
def test_provider_e2e(label, tmp_path):
    reason = _availability(label)
    if reason:
        pytest.skip(reason)
    r = run_one(label, tmp_path)
    assert r["error"] is None, f"{label}: adapter returned error: {r['error']}"
    assert r["n_tool_calls"] > 0, f"{label}: model made no tool calls"
    assert r["graded_ok"], f"{label}: bug not fixed - {r['grade_detail']}"


# ── standalone report ────────────────────────────────────────────────────────
def _main() -> int:
    import tempfile

    # Make the venv's tools (pytest, python) resolvable from the agent's Bash.
    venv_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

    print("=" * 72)
    print("Code-Capsules live e2e - provider adapter smoke test")
    print("=" * 72)

    rows, ran = [], []
    for label in PROVIDERS:
        reason = _availability(label)
        if reason:
            print(f"\n[{label}] SKIP - {reason}")
            rows.append((label, "SKIP", reason))
            continue
        client_name, model, _ = PROVIDERS[label]
        print(f"\n[{label}] running {client_name} (model={model}) …", flush=True)
        with tempfile.TemporaryDirectory() as d:
            try:
                r = run_one(label, Path(d))
            except Exception as exc:  # noqa: BLE001
                print(f"  EXCEPTION: {exc!r}")
                rows.append((label, "ERROR", repr(exc)))
                continue
        ran.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {status}  turns={r['num_turns']} tools={r['n_tool_calls']} "
              f"in={r['input_tokens']} out={r['output_tokens']} "
              f"cost=${r['cost_usd']:.4f} {r['duration_ms']/1000:.1f}s")
        print(f"        diff: {r['diffstat'] or '(none)'}")
        if not r["passed"]:
            print(f"        error={r['error']}  grade={r['grade_detail']}")
        rows.append((label, status, f"{r['n_tool_calls']} tool calls"))

    print("\n" + "=" * 72)
    print("SUMMARY")
    for label, status, detail in rows:
        print(f"  {label:<10} {status:<6} {detail}")
    print("=" * 72)

    failed = [r for r in ran if not r["passed"]]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())

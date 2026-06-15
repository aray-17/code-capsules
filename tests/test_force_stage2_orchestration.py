"""force_stage2 invariant for the FRAMEWORK variant runner
(variants/_orchestration.py::run_escalating) — the SECOND escalation runner, which
was the unfixed leaky copy (`escalate = not bool(resolved1)`, gold-in-decision) while
the harness copy was fixed. This pins that force_stage2 runs stage 2 UNCONDITIONALLY
(no resolved1/gold read for the decision), so the two runners can't drift apart again.

I/O boundary mocked (model client, quality gate, patch capture, prompt build); the
deterministic control flow is under test. Runs under pytest or `python3`.
"""
import sys
import types
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import code_capsules.variants._orchestration as O  # noqa: E402
from code_capsules.api import TaskDescriptor, VariantConfig  # noqa: E402


def _inv(num_turns=10, cost=0.10):
    # run_escalating reads these attributes off the InvocationResult; a namespace suffices.
    return types.SimpleNamespace(
        error=None, num_turns=num_turns, input_tokens=1000, output_tokens=500,
        cost_usd=cost, tool_calls=[], model="claude-sonnet-4-6")


def _task():
    return TaskDescriptor(text="bug", context={"task_id": "x", "worktree_path": "/tmp/x"})


def _run_force_stage2(*, gold_resolved):
    """Drive run_escalating with force_stage2=True; gate.check returns gold_resolved."""
    fake_client = types.SimpleNamespace(
        invoke=mock.Mock(side_effect=[_inv(10, 0.10), _inv(12, 0.15)]))
    fake_gate = types.SimpleNamespace(check=mock.Mock(return_value=gold_resolved))
    cfg = VariantConfig(
        name="two_pass_critique", mode="escalating",
        escalating_start_budget=10, escalating_target_budget=25,
        always_escalate=True, extra={"force_stage2": True})
    with mock.patch.object(O, "_resolve_client", return_value=fake_client), \
         mock.patch.object(O, "_resolve_gate", return_value=fake_gate), \
         mock.patch.object(O, "capture_patch", return_value="diff --git a/x b/x\n+fix\n"), \
         mock.patch.object(O, "build_user_prompt", return_value="prompt"):
        r = O.run_escalating(_task(), cfg)
    return r, fake_client


def test_force_stage2_runs_stage2_even_when_gold_resolved():
    # The leaky copy would have stopped here (escalate = not resolved1, resolved1=True).
    # force_stage2 must run stage 2 regardless of the gold result.
    r, client = _run_force_stage2(gold_resolved=True)
    assert client.invoke.call_count == 2            # stage 2 ran despite gold=resolved
    assert r.num_turns == 10 + 12                    # both stages billed
    assert abs(r.cost_usd - 0.25) < 1e-9
    # stage records show both stages; final resolved comes from the stage-2 gate call
    assert r.stage_records and r.stage_records[-1].get("stage") == 2


def test_force_stage2_runs_stage2_when_gold_unresolved():
    r, client = _run_force_stage2(gold_resolved=False)
    assert client.invoke.call_count == 2
    assert r.num_turns == 22


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"force_stage2 orchestration: {passed}/{len(fns)} pass")

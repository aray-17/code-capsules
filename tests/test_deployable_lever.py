"""Full-composition test for the DEPLOYABLE LEVER (the front door).

Drives `CodeCapsulesRunner` (the package-root lever) from the shipped policy.yaml
controller block with a NON-GOLD, repro-shaped grade_fn -- the exact composition
the README/EXTENSIONS quickstart tells users to copy. Asserts the three components
compose: SOLVERS (diverse pair) -> REPRO-VERIFIER select -> GOVERNOR abandon.

The grade_fn here simulates a deployable verifier (a generated reproduction +
regression check) as a PURE FUNCTION OF THE PATCH -- it never reads a gold/held-out
verdict. That is the deployable contract; this test fails if the lever can't be
driven by a non-gold signal. Runs under pytest or `python3` directly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from code_capsules import CodeCapsulesRunner            # the package-root LEVER  # noqa: E402
from code_capsules.controller.runtime import RunnerPolicy  # noqa: E402
from code_capsules.controller.verifier import RESOLVED, FLAT, NOPATCH  # noqa: E402

_POLICY_YAML = str(Path(__file__).parent.parent / "policy.yaml")


def _sampler(patches):
    """Return a sampler(config, tier) -> patch from a {config: patch} map."""
    return lambda config, tier: patches.get(config, "")


def _deployable_grade_fn(repro_passes):
    """A NON-GOLD verifier: grade a patch from a simulated repro+regression outcome
    (a set of patches the generated reproduction passes on). No gold/held-out read."""
    def grade(patch):
        if not (patch or "").strip():
            return NOPATCH
        return RESOLVED if patch in repro_passes else FLAT
    return grade


def test_lever_loads_from_policy_yaml():
    # The shipped controller block must drive the lever (the documented front door).
    r = CodeCapsulesRunner.from_policy_file(_POLICY_YAML)
    assert r.policy.configs == ("floor", "siginject")     # SOLVERS: the diverse pair
    assert r.policy.escalate_on_agreement is False         # GOVERNOR: abandon default
    assert r.policy.min_samples == 2


def test_verifier_selects_the_resolver():
    # One solver's patch passes the (non-gold) repro -> verifier-SELECT ships it.
    r = CodeCapsulesRunner.from_policy_file(_POLICY_YAML)
    patches = {"floor": "patch-A", "siginject": "patch-B"}
    grade_fn = _deployable_grade_fn(repro_passes={"patch-B"})
    run = r.run(_sampler(patches), grade_fn)
    assert run.outcome == "RESOLVED"
    assert run.selection.grade == RESOLVED                 # kept the repro-passer


def test_governor_abandons_on_agreement_failure():
    # Both solvers' patches fail the (non-gold) repro -> GOVERNOR abandons (no escalate).
    r = CodeCapsulesRunner.from_policy_file(_POLICY_YAML)
    patches = {"floor": "patch-A", "siginject": "patch-B"}
    grade_fn = _deployable_grade_fn(repro_passes=set())     # neither passes
    run = r.run(_sampler(patches), grade_fn)
    assert run.outcome == "ABANDONED"


def test_lever_never_reads_gold():
    # The grade_fn is a pure function of the patch (no gold/docker). If the lever
    # required a gold verdict to decide, this would fail. Deployable contract.
    r = CodeCapsulesRunner.from_policy_file(_POLICY_YAML)
    seen = []
    def grade_fn(patch):
        seen.append(patch)                                 # only ever sees the patch
        return RESOLVED if patch == "winner" else FLAT
    run = r.run(_sampler({"floor": "x", "siginject": "winner"}), grade_fn)
    assert run.outcome == "RESOLVED"
    assert set(seen) <= {"x", "winner"}                    # nothing but patches


def test_escalate_tier_is_opt_in_not_default():
    # Default lever does NOT tier-escalate (validated economical default); opt-in only.
    r = CodeCapsulesRunner(RunnerPolicy(tiers=("sonnet", "opus"), escalate_on_agreement=True))
    grades = {"sonnet:floor": FLAT, "sonnet:siginject": FLAT,
              "opus:floor": RESOLVED, "opus:siginject": FLAT}
    run = r.run(lambda c, t: f"{t}:{c}", lambda p: grades[p])
    assert run.outcome == "RESOLVED" and run.final_tier == "opus"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"deployable lever: {passed}/{len(fns)} pass")

## Summary

<!-- One or two sentences describing what this PR changes and why. -->

## Test plan

- [ ] `pytest -m "not integration and not slow and not benchmark"` passes locally
- [ ] `python3 benchmarks/verify_criteria.py` still reports 12/12 claims reproduce
- [ ] New behavior covered by a test (or this is doc/typo only)
- [ ] No changes to the controller, repro verifier (quality gate), cost
      governor, or the calibrated policy defaults (or: linked issue
      `#<number>` discusses the change)

## Affected modules

<!-- e.g. src/code_capsules/controller/, benchmarks/, examples/ -->

## Notes for the reviewer

<!-- Anything the maintainer should know: design context, trade-offs you
considered, follow-ups deferred. Optional. -->

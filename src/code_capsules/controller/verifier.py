"""
Execution verifier: grade a candidate patch's *execution* outcome into a coarse
class the controller can act on.

This is the linchpin primitive behind the adaptive-compute controller. EXP-1
(n=50 django, 2026-06-01) established the empirical basis: the agent's
self-report does NOT separate winnable from doomed escalations, but the
execution reading separates the *confident tails* cleanly --

    RESOLVED -> 100% eventually resolve  (stop-for-success: don't escalate)
    FLAT     ->   0% eventually resolve  (confident doomed: drop)

while the middle (NOPATCH/PARTIAL/BROKEN) does not separate (NOPATCH at a low
floor budget is 43% winnable -- "no patch yet" means *needs more turns*, not
doomed). So the controller acts on the tails and defaults to escalate on the
middle (see VerifierGate in cascade_triggers).

Generic by construction (neutrality mandate): `grade_execution` takes an
`ExecutionReading` of counts, not any benchmark-specific log format. A benchmark
adapter maps its eval output onto those fields; `from_swebench_eval` is the
shipped SWE-bench shim. The grade classes themselves are domain-agnostic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── Verifier grades (the coarsest signal the controller acts on) ──────────────
RESOLVED = "RESOLVED"   # all targets pass            -> stop-for-success
PARTIAL = "PARTIAL"     # applied, ran, some targets pass / clean assertion fail
FLAT = "FLAT"           # applied, ran, nothing moved  -> confident doomed (drop)
BROKEN = "BROKEN"       # didn't apply / import-collection error -> couldn't run
NOPATCH = "NOPATCH"     # no patch produced (e.g. agent unfinished) -> needs turns
UNKNOWN = "UNKNOWN"     # no readable execution signal

GRADES = (RESOLVED, PARTIAL, FLAT, BROKEN, NOPATCH, UNKNOWN)

# The confident tails EXP-1 validated: RESOLVED is a stop-for-success; the
# ABANDON set is the confident-doomed tail. Default {FLAT} is the zero-resolve-
# loss point; {FLAT, BROKEN} saves more at ~1 lost resolve per EXP-1.
STOP_GRADE = RESOLVED
DEFAULT_ABANDON = frozenset({FLAT})


@dataclass(frozen=True)
class ExecutionReading:
    """Generic execution outcome a verifier produces for one candidate patch.

    has_patch : the candidate produced a non-empty patch at all
    applied   : the patch applied cleanly to the worktree
    n_targets : number of target tests checked (FAIL_TO_PASS [+ PASS_TO_PASS])
    n_pass    : target tests passing
    n_error   : collection/import errors (couldn't run, distinct from assert-fail)
    resolved  : all targets pass (the success criterion)
    """
    has_patch: bool
    applied: bool
    n_targets: int
    n_pass: int
    n_error: int
    resolved: bool


def grade_execution(r: ExecutionReading) -> str:
    """Map an execution reading to a verifier grade (domain-agnostic)."""
    if not r.has_patch:
        return NOPATCH
    if r.resolved:
        return RESOLVED
    if not r.applied or (r.n_error > 0 and r.n_pass == 0):
        return BROKEN
    if r.n_targets <= 0:
        return UNKNOWN
    # n_targets > 0 here (guarded above); >0 passing = partial movement.
    return PARTIAL if r.n_pass > 0 else FLAT


# ── Reference DEPLOYABLE verifier: non-gold repro grade + regression channel ──

class ReproVerifier:
    """The framework's reference DEPLOYABLE (non-gold) verifier, with the
    regression check as its OWN channel rather than folded into the grade.

    The reference deployable verifier is constructed WITH ``regression_ok``: the
    regok signal feeds the shipped ``hybrid_regok`` governor (rescue a would-be
    ABANDON when the selected patch keeps the existing suite green) and the
    optional ``repro_and_regok`` ship gate. Keeping it a separate channel lets
    the governor weigh it (rescue/demote) instead of the grade silently eating
    it as a BROKEN demotion.

    Generic by construction (neutrality mandate): repo-agnostic callables, never
    a benchmark or a gold/held-out verdict. A SWE-bench deployment supplies
    docker-backed callables (see tools/repro_verifier); another domain supplies
    its own (its CI, a local test runner, ...).

    PASS_TO_PASS partial-spec caveat: in the EXP-4 benchmark harness the
    regression channel ran the SWE-bench PASS_TO_PASS subset -- a PARTIAL spec
    of "the existing tests". A deployment runs the repo's OWN suite instead;
    that is a partial-spec dependency of the validated numbers, not gold
    leakage (PASS_TO_PASS names pre-existing tests, never the held-out verdict).

    Both channels are memoized: one repro sweep and one regression run per
    DISTINCT patch (they are the expensive ops -- e.g. docker).
    """

    def __init__(self, repros, repro_passes, regression_ok=None):
        self._repros = list(repros or [])
        self._repro_passes = repro_passes
        self._regression_ok = regression_ok
        self._grade_cache: dict = {}
        self._regression_cache: dict = {}

    def grade(self, patch: str) -> str:
        """Repro-only grade (regression NOT folded in -- see .regression):
        NOPATCH (empty patch) | UNKNOWN (no repros -> no signal; the lever must
        not stop/abandon on it) | RESOLVED (all repros pass) | PARTIAL (some
        pass) | FLAT (a real patch passes none -> confident-doomed)."""
        if not (patch or "").strip():
            return NOPATCH
        if patch not in self._grade_cache:
            self._grade_cache[patch] = self._grade(patch)
        return self._grade_cache[patch]

    def _grade(self, patch: str) -> str:
        if not self._repros:
            return UNKNOWN
        n_pass = sum(1 for r in self._repros if self._repro_passes(patch, r))
        if n_pass == len(self._repros):
            return RESOLVED
        return PARTIAL if n_pass > 0 else FLAT

    def regression(self, patch: str):
        """The regression channel: True/None = existing tests stay green (None =
        no regression callable / no verdict -- None never rescues a ship/abandon
        decision); False = the patch broke them. One run per distinct patch."""
        if self._regression_ok is None or not (patch or "").strip():
            return None
        if patch not in self._regression_cache:
            self._regression_cache[patch] = self._regression_ok(patch)
        return self._regression_cache[patch]


def make_repro_verifier(repros, repro_passes, regression_ok=None) -> ReproVerifier:
    """Build the reference DEPLOYABLE verifier (see ReproVerifier): memoized
    ``.grade(patch) -> grade`` + ``.regression(patch) -> bool|None`` as separate
    channels. Wire ``.grade`` as the lever's grade_fn and ``.regression`` as
    run_round's regression_fn so the governor sees both signals."""
    return ReproVerifier(repros, repro_passes, regression_ok)


def make_repro_grade_fn(repros, repro_passes, regression_ok=None):
    """Back-compat wrapper over make_repro_verifier: the single-channel
    ``grade_fn(patch) -> grade`` older callers consume, with the regression
    verdict FOLDED INTO the grade (all repros pass but regression broke ->
    BROKEN). New callers should prefer make_repro_verifier, which keeps
    regression as its own channel for the hybrid_regok governor.
    """
    v = make_repro_verifier(repros, repro_passes, regression_ok)

    def grade(patch: str) -> str:
        g = v.grade(patch)
        if g == RESOLVED and v.regression(patch) is False:
            return BROKEN
        return g

    return grade


# ── SWE-bench shim: docker_eval output -> ExecutionReading -> grade ───────────
_RAN = re.compile(r"ran (\d+) test", re.I)
_FAILURES = re.compile(r"failures=(\d+)", re.I)
_ERRORS = re.compile(r"errors=(\d+)", re.I)
_PYTEST = re.compile(r"(\d+) passed|(\d+) failed|(\d+) error", re.I)
_BROKEN_MARKERS = ("error: patch failed", "corrupt patch", "modulenotfounderror",
                   "importerror", "no module named", "syntaxerror")


def from_swebench_eval(eval_result: dict, n_targets: int) -> str:
    """Grade a `run_swebench_docker.docker_eval` result dict.

    eval_result: {"resolved": bool|None, "note": str, "stdout": str, ...}
    n_targets:   len(FAIL_TO_PASS) (+ PASS_TO_PASS if scored together)

    SCORING-ONLY. The RESOLVED grade here is derived from the GOLD SWE-bench
    FAIL_TO_PASS verdict (`eval_result["resolved"]`). That is a benchmark-ceiling
    SCORING path: do NOT use this grade to drive a deployment stop / abandon /
    escalate control decision -- the gold verdict is unavailable at deployment time
    (using it is the in-loop leakage we removed via force_stage2). A DEPLOYABLE
    grade_fn must read a non-gold signal (a generated fail-on-base reproduction +
    a regression check); see tools/repro_verifier.grade_candidate.
    """
    note = (eval_result.get("note") or "").lower()
    if "empty" in note or "no patch" in note or not eval_result.get("has_patch", True):
        # callers without has_patch fall through on note; empty-patch is NOPATCH
        if "empty" in note or "no patch" in note:
            return NOPATCH
    out = (eval_result.get("stdout") or "")
    o = out.lower()
    resolved = eval_result.get("resolved") is True
    if resolved:
        return RESOLVED
    if not o.strip():
        return NOPATCH if ("empty" in note) else UNKNOWN
    if any(m in o for m in _BROKEN_MARKERS):
        return BROKEN
    ran = _RAN.search(o)
    if ran:
        n = int(ran.group(1)) or 1
        f = int(_FAILURES.search(o).group(1)) if _FAILURES.search(o) else 0
        e = int(_ERRORS.search(o).group(1)) if _ERRORS.search(o) else 0
        if e > 0 and f == 0:
            return BROKEN
        return PARTIAL if (n - f - e) > 0 else FLAT
    p_pass = re.search(r"(\d+) passed", o)
    p_fail = re.search(r"(\d+) failed", o)
    p_err = re.search(r"(\d+) error", o)
    if p_pass or p_fail:
        np_ = int(p_pass.group(1)) if p_pass else 0
        ne_ = int(p_err.group(1)) if p_err else 0
        nf_ = int(p_fail.group(1)) if p_fail else 0
        if ne_ > 0 and nf_ == 0 and np_ == 0:
            return BROKEN
        return PARTIAL if np_ > 0 else FLAT
    return UNKNOWN


__all__ = [
    "RESOLVED", "PARTIAL", "FLAT", "BROKEN", "NOPATCH", "UNKNOWN", "GRADES",
    "STOP_GRADE", "DEFAULT_ABANDON", "ExecutionReading", "grade_execution",
    "ReproVerifier", "make_repro_verifier", "make_repro_grade_fn",
    "from_swebench_eval",
]

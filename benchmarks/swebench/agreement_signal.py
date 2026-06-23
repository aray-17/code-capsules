"""Cross-sample diverse-agreement as the DOOMED signal - the paper's Figure 4
(fig:agreement, §9.1) computed offline from the existing 10-config Sonnet sweeps
(zero new compute). This is the signal the generated repro never gave: the repro is
a "done" signal (good for verifier-SELECT, the quality lever) but has no reliable
"doomed" signal.

FIGURE UNIVERSE (§9.1): the n=126 IN-FAMILY Sonnet intersection - the instances
that EVERY shipped configuration attempts. Some configurations attempted fewer than
150 instances (phaseStaged 127, toolAlloc 143, floor/siginject/planFirst/twopass
149); intersecting their attempted sets yields n=126, all sympy-free. The per-config
attempted sets are read from the canonical overlays in evals/scopeC/menu/; the
resolved labels carried in the raw evals/p10_sonnet_*_n150_*.jsonl sweeps are already
canonical (they match the overlays' canonical_resolved exactly), and cost/turns come
from those same raw sweeps.

CANONICAL Figure 4 numbers this script reproduces (paper §9.1, fig:agreement):
  n=126 in-family Sonnet intersection
  doomed-at-tier (0 of 10 configs resolve)      20/126
  oracle union (any one config resolves)       106/126
  single-config (floor) precision               64.5%  (fires 31, false-abandon 11)
  median of all 45 diverse pairs                66.7%
  best diverse pair (floor + siginject)         90.9%  (fires 22, false-abandon 2)
  independent (out-of-family) precision         ~22%   (6/27 truly-doomed, COLLAPSE)
  single baseline select (floor)                95/126 @ $43.20
  two-config select (floor + siginject)        104/126 @ $85.02  ($0.82/resolve)
  third config (planFirst) adds                  0     (still 104/126)

The independent bar is computed on the floor+siginject pairwise both-fail universe
(their own 149-instance intersection -> 27 both-fail), the natural abandon-universe
for that pair, scored against an out-of-family reference (Opus floor + Agentless):
6 of 27 are doomed everywhere => ~22% precision (most flagged instances recover one
tier up; the signal measures tier-relative difficulty, hence its deployable use is an
escalation trigger, not an early-abandon rule).

The cross-model / Opus-tier / Haiku-ladder sections below are SEPARATE n=150 cross-tier
escalation analyses (§11/§12 economics); they deliberately use the full n=150 universe
and do NOT assert the n=126 figure numbers.

  python3 benchmarks/swebench/agreement_signal.py
"""
from __future__ import annotations

import glob
import json
import re
import statistics
from itertools import combinations
from pathlib import Path

# Anchor every eval glob on the repo root so the scorer reproduces from ANY
# working directory. A reader running the bare
# `python3 benchmarks/swebench/agreement_signal.py` from elsewhere would
# otherwise silently get empty results.
ROOT = Path(__file__).resolve().parents[2]

# Canonical Figure 4 (fig:agreement, §9.1) targets the reproduction is checked against.
FIGURE_TARGETS = dict(
    n=126,
    doomed=20,
    oracle_union=106,
    single_fires=31,
    single_false=11,
    single_prec=0.645,
    median_prec=0.667,
    best_pair=("floor", "siginject_b10x3"),
    best_fires=22,
    best_false=2,
    best_prec=0.909,
    floor_select=95,
    floor_cost=43.20,
    two_select=104,
    two_cost=85.02,
    third_adds=0,
    independent_doomed=6,
    independent_universe=27,
    independent_prec=0.22,
)


def _attempted_keysets():
    """The per-config ATTEMPTED instance sets that define the n=126 in-family
    universe, read from the canonical overlays in evals/scopeC/menu/. The figure's
    universe is the intersection of these sets (the instances every shipped
    configuration attempts)."""
    keysets = {}
    for f in sorted(glob.glob(str(ROOT / "evals/scopeC/menu/canonical_p10_sonnet_*_n150_*.jsonl"))):
        if re.search(r"_var\d+_n\d+", f):
            continue
        name = re.search(r"canonical_p10_sonnet_(.+?)_n\d+", f).group(1)
        keysets[name] = set(json.loads(line)["instance_id"] for line in open(f) if line.strip())
    return keysets


def load_configs(pattern="evals/p10_sonnet_*n150*.jsonl", restrict_to=None):
    """Load resolved/cost/turns from the raw Sonnet sweeps. The `resolved` field in
    these files is already canonical (verified identical to the canonical overlays).
    When `restrict_to` (a {config_name: attempted_set} map) is given, each config is
    restricted to the instances it actually attempted - this is what carves the
    n=126 in-family universe out of the 150-row raw files."""
    configs = {}
    for f in sorted(glob.glob(str(ROOT / pattern))):
        # EXCLUDE same-strategy variance reruns (implicit40_var2/var3, twopass_*_var2/var3):
        # a reroll resolving an instance is sampling noise, not a recoverable strategy, so
        # absorbing it into the reference set would spuriously depress abandon precision.
        if re.search(r"_var\d+_n\d+", f):
            continue
        m = re.search(r"p10_[a-z]+_(.+?)_n\d+", f)   # model-agnostic config name
        name = m.group(1) if m else f
        attempted = restrict_to.get(name) if restrict_to else None
        if restrict_to is not None and attempted is None:
            # config not present in the canonical universe map -> skip it
            continue
        d = {}
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                iid = r["instance_id"]
                if attempted is not None and iid not in attempted:
                    continue
                d[iid] = dict(
                    resolved=(r.get("resolved") is True),
                    cost=float(r.get("cost_usd") or 0.0),
                    turns=int(r.get("actual_turns") or 0),
                )
        if d:
            configs[name] = d
    return configs


def policy(configs, ids, members):
    """A diverse-sampling policy = run `members` configs, keep the union (verifier-
    SELECT picks the resolver). cost/turns = sum across members (you ran them all)."""
    resolved = sum(1 for i in ids if any(configs[m][i]["resolved"] for m in members))
    cost = sum(configs[m][i]["cost"] for m in members for i in ids)
    turns = sum(configs[m][i]["turns"] for m in members for i in ids)
    return resolved, cost, turns


def best_abandon_pair(configs, ids):
    """The cross-sample mechanism, model-agnostic: the best 2-config abandon pair
    (max precision = 1 - recoverable-by-others/fired) over all diverse pairs. The
    PAIR is model-specific (siginject collapses on weak models), but the MECHANISM
    -- 2 diverse configs agreeing on failure => doomed -- is what we test transfers."""
    names = list(configs)
    best = None
    for a, b in combinations(names, 2):
        budget = {a, b}
        others = [c for c in names if c not in budget]
        fired = [i for i in ids if not configs[a][i]["resolved"] and not configs[b][i]["resolved"]]
        if not fired:
            continue
        false_ab = sum(1 for i in fired if any(configs[c][i]["resolved"] for c in others))
        prec = 1 - false_ab / len(fired)
        if best is None or prec > best[1]:
            best = ((a, b), prec, len(fired), false_ab)
    return best


def _independent_precision(keysets):
    """The right bar of Figure 4: deployment-relevant precision measured fully
    out-of-family. The floor+siginject both-fail set over the pair's own 149-instance
    intersection (= 27 instances, the natural abandon-universe for that pair) is scored
    against an independent reference that varies BOTH the model tier (Opus floor) and
    the agent harness (Agentless). Instances recovered there were never truly doomed."""
    def _res(path_glob):
        files = [x for x in glob.glob(str(ROOT / path_glob)) if "bak" not in x and "invalid" not in x]
        if not files:
            return None
        d = {}
        for line in open(sorted(files)[0]):
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["instance_id"]] = (r.get("resolved") is True)
        return d

    floor = _res("evals/p10_sonnet_floor_n150_*.jsonl")
    sig = _res("evals/p10_sonnet_siginject_b10x3_n150_*.jsonl")
    if floor is None or sig is None:
        return None
    fs_ids = keysets["floor"] & keysets["siginject_b10x3"]
    abandon = sorted(i for i in fs_ids if not floor.get(i) and not sig.get(i))
    # Independent out-of-family reference: Opus floor + Agentless (canonical overlays).
    opus = _res("evals/p9_opus_floor_n150_combined_20260527.jsonl")
    ag = _res("evals/h2h_agentless_sonnet_n150_run1_20260528.jsonl")
    if opus is None or ag is None:
        return None
    truly = [i for i in abandon if not opus.get(i) and not ag.get(i)]
    prec = len(truly) / max(1, len(abandon))
    return len(truly), len(abandon), prec


def cross_model_replication():
    """SEPARATE n=150 analysis (NOT the §9.1 figure). Does the abandon MECHANISM
    replicate across model tiers? Reports the best diverse-pair precision per model
    from existing n=150 sweeps (zero new compute). Strengthens generalizability: the
    mechanism transfers; the partner config is model-specific (Sonnet=floor+siginject,
    weaker models=a budget lever)."""
    print("\n=== CROSS-MODEL within-tier replication, n=150 universe (NOT the §9.1 figure) ===")
    for model, pat in (("Sonnet", "evals/p10_sonnet_*n150*.jsonl"),
                       ("Haiku", "evals/p10_haiku_*n150*.jsonl")):
        cfgs = {k: v for k, v in load_configs(pat).items() if "ranker" not in k}
        if len(cfgs) < 3:
            print(f"  {model:<7}: insufficient configs ({len(cfgs)})"); continue
        ids = sorted(set.intersection(*[set(d) for d in cfgs.values()]))
        oracle = {i: any(cfgs[c][i]["resolved"] for c in cfgs) for i in ids}
        doomed = sum(1 for i in ids if not oracle[i])
        bp = best_abandon_pair(cfgs, ids)
        (a, b), prec, fired, fa = bp
        print(f"  {model:<7} n={len(ids):>3} doomed={doomed:>3}: best pair [{a}+{b}] "
              f"precision {prec:.0%} (fires {fired}, false-abandon {fa})")
    print("  -> mechanism replicates across tiers; partner config is model-specific.")


def opus_tier_boundary():
    """SEPARATE n=150 cross-tier escalation analysis (§11/§12 economics, NOT the §9.1
    figure). The per-tier abandon-vs-escalate boundary (Opus siginject sweep). Does the
    agreement mechanism fire at the STRONG tier, and what does escalating Sonnet-doomed
    to Opus actually buy? Uses resolved values. The doomed SET is independently validated
    (Agentless recovers none of them)."""
    import glob as _g, json as _j
    def L(p):
        f = sorted(x for x in _g.glob(str(ROOT / p)) if "bak" not in x and "invalid" not in x)[0]
        d = {}
        for line in open(f):
            line = line.strip()
            if line:
                r = _j.loads(line)
                d[r["instance_id"]] = dict(res=(r.get("resolved") is True),
                                           cost=float(r.get("cost_usd") or 0))
        return d
    sf = L("evals/p10_sonnet_floor_n150_*.jsonl"); ss = L("evals/p10_sonnet_siginject_b10x3_n150_*.jsonl")
    # PIN the canonical Opus floor file (05-27): the 05-26 combined had 19 no-verdict
    # zero-cost rows (instances that never ran), counted as doomed and depressing Opus
    # recovery; the 05-27 re-run graded 18 of those 19 (11 resolved), 0 regressions.
    of = L("evals/p9_opus_floor_n150_combined_20260527.jsonl")
    osig_files = _g.glob(str(ROOT / "evals/opus_siginject_agreementfail65.jsonl"))
    if not osig_files:
        print("\n=== OPUS-TIER boundary: opus siginject sweep not found, skipping ==="); return
    osig = L("evals/opus_siginject_agreementfail65.jsonl")
    ag = L("evals/h2h_agentless_sonnet_n150_run1_20260528.jsonl")
    the_doomed = [i for i in (set(sf) & set(ss)) if not sf[i]["res"] and not ss[i]["res"]]
    ofR = {i for i in the_doomed if of.get(i, {}).get("res")}
    osR = {i for i in the_doomed if osig.get(i, {}).get("res")}
    union = ofR | osR
    bof = [i for i in the_doomed if i not in union]
    ag_rec = [i for i in bof if ag.get(i, {}).get("res")]
    oc = sum(of.get(i, {}).get("cost", 0) for i in the_doomed) + sum(osig.get(i, {}).get("cost", 0) for i in the_doomed)
    print(f"\n=== OPUS-TIER abandon-vs-escalate boundary, n=150 (n={len(the_doomed)} Sonnet-doomed) ===")
    print(f"  Opus floor {len(ofR)} | Opus siginject {len(osR)} | union {len(union)} "
          f"(diverse: +{len(osR-ofR)}/{len(ofR-osR)} each unique)")
    print(f"  escalate-tier reach {len(union)}/{len(the_doomed)} ({len(union)/len(the_doomed):.0%}) at modeled "
          f"${oc:.2f} = ${oc/max(1,len(union)):.2f}/resolve (vs $1.40 base => marginal; confirms the cascade is uneconomical)")
    print(f"  both-Opus-fail {len(bof)}; Agentless recovers {len(ag_rec)} => ABANDON precision "
          f"{1-len(ag_rec)/max(1,len(bof)):.0%} (doomed across 5 configs / 2 tiers / 2 harnesses)")
    print(f"  -> ABANDON is the economical default; the agreement mechanism FIRES at the Opus tier.")


def haiku_to_high_escalation():
    """SEPARATE n=150 low->high cost-ladder analysis (§11/§12, NOT the §9.1 figure).
    Does the agreement signal at the cheap (Haiku) tier identify doom, and what does
    escalating Haiku-doomed up to Sonnet/Opus recover? Uses the Haiku-appropriate
    diverse pair (floor+implicit40; siginject collapses on Haiku)."""
    import glob as _g, json as _j
    def L(p):
        fs=[x for x in _g.glob(str(ROOT / p)) if 'bak' not in x and 'invalid' not in x]
        if not fs: return None
        d={}
        for line in open(sorted(fs)[0]):
            line=line.strip()
            if line:
                r=_j.loads(line); d[r["instance_id"]]=dict(res=(r.get("resolved") is True), cost=float(r.get("cost_usd") or 0))
        return d
    hf=L("evals/p10_haiku_floor_n150_*.jsonl"); hi=L("evals/p10_haiku_implicit40_n150_*.jsonl")
    sf=L("evals/p10_sonnet_floor_n150_*.jsonl"); of=L("evals/p9_opus_floor_n150_combined_20260527.jsonl")  # canonical (see opus_tier_boundary note)
    if not all([hf,hi,sf,of]):
        print("\n=== Haiku->high ladder: missing data, skipping ==="); return
    ids=sorted(set(hf)&set(hi)&set(sf)&set(of)); n=len(ids)
    doomed=[i for i in ids if not hf[i]['res'] and not hi[i]['res']]
    s_rec=[i for i in doomed if sf[i]['res']]; o_rec=[i for i in doomed if of[i]['res']]
    either=[i for i in doomed if sf[i]['res'] or of[i]['res']]
    truly=[i for i in doomed if not sf[i]['res'] and not of[i]['res']]
    sc=sum(sf[i]['cost'] for i in doomed)
    print(f"\n=== LOW->HIGH ladder, n=150: Haiku agreement-fail (floor+implicit40) -> Sonnet/Opus (n={n}) ===")
    print(f"  Haiku agreement-fail (doomed): {len(doomed)}/{n}")
    print(f"  escalate -> Sonnet recovers {len(s_rec)} ({len(s_rec)/max(1,len(doomed)):.0%}); "
          f"Opus recovers {len(o_rec)} ({len(o_rec)/max(1,len(doomed)):.0%}); either {len(either)}")
    print(f"  truly-doomed up to Opus: {len(truly)}/{n} (abandon-safe across 3 tiers)")
    print(f"  Sonnet-escalation cost ${sc:.2f} -> ${sc/max(1,len(s_rec)):.2f}/resolve (marginal, > $1.40 base).")
    print(f"  -> low->high recovers MORE than mid->high (weaker start = more headroom) but stays a costly tail.")


def figure_reproduction():
    """Reproduce Figure 4 (fig:agreement, §9.1) on the n=126 in-family Sonnet
    intersection and assert every printed number matches the paper. Returns the
    computed dict + a list of (label, computed, target, ok) checks."""
    keysets = _attempted_keysets()
    if not keysets:
        print("ERROR: canonical menu overlays not found under evals/scopeC/menu/ - cannot define n=126 universe.")
        return None, None, keysets
    configs = load_configs(restrict_to=keysets)
    ids = sorted(set.intersection(*[set(d) for d in configs.values()]))
    n = len(ids)
    oracle = {i: any(configs[c][i]["resolved"] for c in configs) for i in ids}
    doomed = [i for i in ids if not oracle[i]]

    print("=== FIGURE 4 (fig:agreement, §9.1): n=126 in-family Sonnet intersection ===")
    print(f"configs: {list(configs)}")
    print(f"n={n}; oracle-union {sum(oracle.values())}/{n}; DOOMED(0/{len(configs)}) {len(doomed)}/{n}\n")

    # --- single-config (floor) precision: the blunt one-config signal ---
    others = [c for c in configs if c != "floor"]
    s_fired = [i for i in ids if not configs["floor"][i]["resolved"]]
    s_false = [i for i in s_fired if any(configs[c][i]["resolved"] for c in others)]
    s_prec = 1 - len(s_false) / max(1, len(s_fired))

    # --- all 45 diverse pairs: median + best ---
    names = list(configs)
    pair_prec = {}
    for a, b in combinations(names, 2):
        budget = {a, b}
        oth = [c for c in names if c not in budget]
        fired = [i for i in ids if not configs[a][i]["resolved"] and not configs[b][i]["resolved"]]
        if not fired:
            continue
        false_ab = sum(1 for i in fired if any(configs[c][i]["resolved"] for c in oth))
        pair_prec[(a, b)] = (1 - false_ab / len(fired), len(fired), false_ab)
    median_prec = statistics.median(sorted(p[0] for p in pair_prec.values()))
    best_pair, (best_prec, best_fires, best_false) = max(pair_prec.items(), key=lambda kv: kv[1][0])

    print("ABANDON-precision bars (Figure 4, in-family):")
    print(f"  single config (floor)        precision {s_prec:>5.1%}  (fires {len(s_fired)}, false-abandon {len(s_false)})")
    print(f"  median of {len(pair_prec)} diverse pairs   precision {median_prec:>5.1%}")
    print(f"  best diverse pair {best_pair[0]}+{best_pair[1]}  precision {best_prec:>5.1%}  "
          f"(fires {best_fires}, false-abandon {best_false})")

    indep = _independent_precision(keysets)
    if indep:
        i_truly, i_univ, i_prec = indep
        print(f"  independent (out-of-family)  precision {i_prec:>5.1%}  ({i_truly}/{i_univ} doomed everywhere -- COLLAPSE)")

    # --- coverage levers: select / escalate ---
    fr, fc, _ = policy(configs, ids, ["floor"])
    r2, c2, _ = policy(configs, ids, ["floor", "siginject_b10x3"])
    r3, c3, _ = policy(configs, ids, ["floor", "siginject_b10x3", "planFirst_b20"])
    print("\nCOVERAGE levers (gold-graded replay):")
    print(f"  single baseline select (floor)            {fr}/{n} @ ${fc:.2f}")
    print(f"  two-config select (floor+siginject)       {r2}/{n} @ ${c2:.2f}  (${c2/max(1,r2):.2f}/resolve)")
    print(f"  + third config (planFirst)                {r3}/{n}  (adds {r3 - r2})")

    # --- assertions against the canonical figure ---
    t = FIGURE_TARGETS
    checks = [
        ("n", n, t["n"]),
        ("doomed", len(doomed), t["doomed"]),
        ("oracle_union", sum(oracle.values()), t["oracle_union"]),
        ("single_fires", len(s_fired), t["single_fires"]),
        ("single_false", len(s_false), t["single_false"]),
        ("single_prec(%.1f%%)", round(s_prec * 100, 1), round(t["single_prec"] * 100, 1)),
        ("median_prec(%.1f%%)", round(median_prec * 100, 1), round(t["median_prec"] * 100, 1)),
        ("best_pair", tuple(sorted(best_pair)), tuple(sorted(t["best_pair"]))),
        ("best_fires", best_fires, t["best_fires"]),
        ("best_false", best_false, t["best_false"]),
        ("best_prec(%.1f%%)", round(best_prec * 100, 1), round(t["best_prec"] * 100, 1)),
        ("floor_select", fr, t["floor_select"]),
        ("floor_cost($)", round(fc, 2), round(t["floor_cost"], 2)),
        ("two_select", r2, t["two_select"]),
        ("two_cost($)", round(c2, 2), round(t["two_cost"], 2)),
        ("third_adds", r3 - r2, t["third_adds"]),
    ]
    if indep:
        # The paper states the independent bar as "roughly 22%" (6/27 = 22.2%); compare
        # at the integer-percent precision the paper itself uses for this approximate bar.
        checks += [
            ("independent_doomed", i_truly, t["independent_doomed"]),
            ("independent_universe", i_univ, t["independent_universe"]),
            ("independent_prec(~%d%%)", round(i_prec * 100), round(t["independent_prec"] * 100)),
        ]
    results = [(label, comp, targ, comp == targ) for (label, comp, targ) in checks]
    return dict(n=n, configs=configs, ids=ids), results, keysets


def main():
    _, results, _ = figure_reproduction()
    if results is None:
        print("\n*** BLOCKED: could not load canonical data to reproduce Figure 4. ***")
        return
    failed = [r for r in results if not r[3]]
    print("\n=== FIGURE 4 reproduction check vs paper §9.1 ===")
    for label, comp, targ, ok in results:
        mark = "OK " if ok else "XX "
        print(f"  {mark}{label:<26} computed={comp!s:<20} paper={targ!s}")
    if failed:
        print(f"\n*** MISMATCH: {len(failed)} of {len(results)} figure numbers do NOT match the paper. ***")
    else:
        print(f"\nMATCH: all {len(results)} Figure 4 (fig:agreement, §9.1) numbers reproduce the paper EXACTLY "
              f"on the n=126 in-family Sonnet intersection.")

    # Separate n=150 cross-tier escalation analyses (§11/§12 economics) - NOT the figure.
    cross_model_replication()
    opus_tier_boundary()
    haiku_to_high_escalation()


if __name__ == "__main__":
    main()

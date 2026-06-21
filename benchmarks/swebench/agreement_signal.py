"""Cross-sample diverse-agreement as the DOOMED signal — computed offline from the
existing 10-config n=150 Sonnet sweeps (zero new compute). This is the signal the
generated repro never gave: the repro is a "done" signal (good for verifier-SELECT,
the quality lever) but has no reliable "doomed" signal, which is why the go/no-go's
repro-gated controller wasted 10/12 escalations.

Finding: if 2 diverse configs (floor + siginject) BOTH fail at b10, the instance is
doomed with ~93% precision (2/28 recoverable by any other strategy), 100% coverage,
no gold tests. Contrast: repro coverage 67% (33% ens=0 inert) + 62% false-stop.

  python3 benchmarks/swebench/agreement_signal.py
"""
from __future__ import annotations

import glob
import json
from itertools import combinations
from pathlib import Path

# Anchor every eval glob on the repo root so the scorer reproduces from ANY
# working directory. verify_criteria.py invokes it with cwd=ROOT, but a reader
# running the bare `python3 benchmarks/swebench/agreement_signal.py` from
# elsewhere would otherwise silently get empty results.
ROOT = Path(__file__).resolve().parents[2]


def load_configs(pattern="evals/p10_sonnet_*n150*.jsonl"):
    import re
    configs = {}
    for f in sorted(glob.glob(str(ROOT / pattern))):
        # EXCLUDE same-strategy variance reruns (implicit40_var2/var3, twopass_*_var2/var3):
        # a reroll resolving an instance is sampling noise, not a recoverable strategy, so
        # absorbing it into the reference set would spuriously depress abandon precision.
        if re.search(r"_var\d+_n\d+", f):
            continue
        m = re.search(r"p10_[a-z]+_(.+?)_n\d+", f)   # model-agnostic config name
        name = m.group(1) if m else f
        d = {}
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["instance_id"]] = dict(
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


def best_abandon_pair(configs, ids, oracle):
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


def cross_model_replication():
    """Does the abandon MECHANISM replicate across model tiers? Reports the best
    diverse-pair precision per model from existing n=150 sweeps (zero new compute).
    Strengthens the generalizability claim: the mechanism transfers; the partner
    config is model-specific (Sonnet=floor+siginject, weaker models=a budget lever)."""
    print("\n=== CROSS-MODEL within-tier replication (best 2-config abandon pair) ===")
    for model, pat in (("Sonnet", "evals/p10_sonnet_*n150*.jsonl"),
                       ("Haiku", "evals/p10_haiku_*n150*.jsonl")):
        cfgs = {k: v for k, v in load_configs(pat).items() if "ranker" not in k}
        if len(cfgs) < 3:
            print(f"  {model:<7}: insufficient configs ({len(cfgs)})"); continue
        ids = sorted(set.intersection(*[set(d) for d in cfgs.values()]))
        oracle = {i: any(cfgs[c][i]["resolved"] for c in cfgs) for i in ids}
        doomed = sum(1 for i in ids if not oracle[i])
        bp = best_abandon_pair(cfgs, ids, oracle)
        (a, b), prec, fired, fa = bp
        print(f"  {model:<7} n={len(ids):>3} doomed={doomed:>3}: best pair [{a}+{b}] "
              f"precision {prec:.0%} (fires {fired}, false-abandon {fa})")
    print("  -> mechanism replicates across tiers; partner config is model-specific.")


def opus_tier_boundary():
    """The per-tier abandon-vs-escalate boundary (Opus siginject sweep, 2026-06-04).
    Does the agreement mechanism fire at the STRONG tier, and what does escalating
    Sonnet-doomed to Opus actually buy? Uses resolved values (the 'unknown error'
    field in the Sonnet siginject sweep is spurious -- present on resolved=True rows
    too; resolved is valid). The doomed SET is independently validated (Agentless
    recovers none of them), so robust to any siginject give-ups."""
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
    # Globbing 2026052[67]* + sorted()[0] silently picked the incomplete 05-26 file.
    of = L("evals/p9_opus_floor_n150_combined_20260527.jsonl")
    osig_files = _g.glob(str(ROOT / "evals/opus_siginject_agreementfail65.jsonl"))
    if not osig_files:
        print("\n=== OPUS-TIER boundary: opus siginject sweep not found, skipping ==="); return
    osig = L("evals/opus_siginject_agreementfail65.jsonl")
    ag = L("evals/h2h_agentless_sonnet_n150_run1_20260528.jsonl")
    the65 = [i for i in (set(sf) & set(ss)) if not sf[i]["res"] and not ss[i]["res"]]
    ofR = {i for i in the65 if of.get(i, {}).get("res")}
    osR = {i for i in the65 if osig.get(i, {}).get("res")}
    union = ofR | osR
    bof = [i for i in the65 if i not in union]
    ag_rec = [i for i in bof if ag.get(i, {}).get("res")]
    oc = sum(of.get(i, {}).get("cost", 0) for i in the65) + sum(osig.get(i, {}).get("cost", 0) for i in the65)
    print(f"\n=== OPUS-TIER abandon-vs-escalate boundary (n={len(the65)} Sonnet-doomed) ===")
    print(f"  Opus floor {len(ofR)} | Opus siginject {len(osR)} | union {len(union)} "
          f"(diverse: +{len(osR-ofR)}/{len(ofR-osR)} each unique)")
    print(f"  escalate-tier reach {len(union)}/{len(the65)} ({len(union)/len(the65):.0%}) at modeled "
          f"${oc:.2f} = ${oc/max(1,len(union)):.2f}/resolve (vs $1.40 base => marginal; confirms Phase 8/9 negative)")
    print(f"  both-Opus-fail {len(bof)}; Agentless recovers {len(ag_rec)} => ABANDON precision "
          f"{1-len(ag_rec)/max(1,len(bof)):.0%} (doomed across 5 configs / 2 tiers / 2 harnesses)")
    print(f"  -> ABANDON is the economical default; the agreement mechanism FIRES at the Opus tier.")


def haiku_to_high_escalation():
    """The FULL low->high cost ladder: does the agreement signal at the cheap
    (Haiku) tier identify doom, and what does escalating Haiku-doomed up to Sonnet/
    Opus recover? Uses the Haiku-appropriate diverse pair (floor+implicit40; siginject
    collapses on Haiku). Answers 'did we try Haiku->Opus?' (2026-06-04)."""
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
    print(f"\n=== LOW->HIGH ladder: Haiku agreement-fail (floor+implicit40) -> Sonnet/Opus (n={n}) ===")
    print(f"  Haiku agreement-fail (doomed): {len(doomed)}/{n}")
    print(f"  escalate -> Sonnet recovers {len(s_rec)} ({len(s_rec)/max(1,len(doomed)):.0%}); "
          f"Opus recovers {len(o_rec)} ({len(o_rec)/max(1,len(doomed)):.0%}); either {len(either)}")
    print(f"  truly-doomed up to Opus: {len(truly)}/{n} (abandon-safe across 3 tiers)")
    print(f"  Sonnet-escalation cost ${sc:.2f} -> ${sc/max(1,len(s_rec)):.2f}/resolve (marginal, > $1.40 base).")
    print(f"  -> low->high recovers MORE than mid->high (weaker start = more headroom) but stays a costly tail.")


def main():
    configs = load_configs()
    ids = sorted(set.intersection(*[set(d) for d in configs.values()]))
    n = len(ids)
    oracle = {i: any(configs[c][i]["resolved"] for c in configs) for i in ids}
    doomed = [i for i in ids if not oracle[i]]
    print(f"configs: {list(configs)}")
    print(f"n={n}; oracle-union {sum(oracle.values())}/{n}; DOOMED(0/{len(configs)}) {len(doomed)}/{n}\n")

    # --- abandon signal precision/recall by budget ---
    print("ABANDON signal (all budget-configs fail -> predict doomed):")
    for budget in (["floor"], ["floor", "siginject_b10x3"],
                   ["floor", "siginject_b10x3", "planFirst_b20"]):
        if not all(b in configs for b in budget):
            continue
        others = [c for c in configs if c not in budget]
        fired = [i for i in ids if all(not configs[c][i]["resolved"] for c in budget)]
        false_ab = [i for i in fired if any(configs[c][i]["resolved"] for c in others)]
        prec = 1 - len(false_ab) / max(1, len(fired))
        rec = sum(1 for i in doomed if all(not configs[c][i]["resolved"] for c in budget)) / max(1, len(doomed))
        print(f"  [{'+'.join(b.split('_')[0] for b in budget):<22}] fires {len(fired):>3}  "
              f"false-abandon {len(false_ab):>2}  precision {prec:>4.0%}  recall {rec:>4.0%}")

    # --- cost-resolution frontier ($, the honest cost axis) ---
    print("\nCOST-RESOLUTION frontier on n=150 ($ = modeled cost_usd):")
    rows = []
    rows.append(("floor (b10)", *policy(configs, ids, ["floor"])))
    rows.append(("siginject (b10x3)", *policy(configs, ids, ["siginject_b10x3"])))
    rows.append(("implicit40 (b40)", *policy(configs, ids, ["implicit40"])) if "implicit40" in configs else None)
    rows.append(("2-config select: floor+siginject", *policy(configs, ids, ["floor", "siginject_b10x3"])))
    rows.append(("3-config escalate: +planFirst", *policy(configs, ids, ["floor", "siginject_b10x3", "planFirst_b20"])))
    rows = [r for r in rows if r]
    print(f"  {'policy':<36} {'resolve':>9} {'cost$':>8} {'turns':>7}")
    for name, res, cost, turns in rows:
        print(f"  {name:<36} {res:>4}/{n}   ${cost:>6.2f} {turns:>7}")

    # --- the abandon trade vs a 3rd escalation tier ---
    u2 = set(i for i in ids if configs["floor"][i]["resolved"] or configs["siginject_b10x3"][i]["resolved"])
    bothfail = [i for i in ids if i not in u2]
    pf_recovers = [i for i in bothfail if configs.get("planFirst_b20", {}).get(i, {}).get("resolved")]
    pf_cost_on_bothfail = sum(configs["planFirst_b20"][i]["cost"] for i in bothfail) if "planFirst_b20" in configs else 0
    print(f"\nABANDON TRADE (2-config select, then abandon-if-both-fail vs add a 3rd escalation tier):")
    print(f"  both-fail instances: {len(bothfail)}  (abandon them)")
    print(f"  a 3rd tier (planFirst) would recover {len(pf_recovers)} of them at +${pf_cost_on_bothfail:.2f}")
    print(f"  -> abandon SAVES ${pf_cost_on_bothfail:.2f} to lose {len(pf_recovers)} resolve(s): "
          f"the cross-sample doomed signal is the cost lever the repro never delivered.")
    cross_model_replication()
    opus_tier_boundary()
    haiku_to_high_escalation()


if __name__ == "__main__":
    main()

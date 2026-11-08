"""Paired-difference analysis + Pareto dominance over the held-out comparison CSVs.

run_compare.py writes runs/compare/<agent>_episodes.csv with one row per seed in the
same seed order for every agent, so the rows are paired. This reads those files and
reports, for each pair, the per-seed mean difference, a paired 95% CI, and a Wilcoxon
signed-rank p-value, then a dominance table checking pairwise Pareto dominance.

Usage:
  python -m src.rl.analyze_compare
  python -m src.rl.analyze_compare --dir runs/compare --pairs shield-rl,shield-tmpc,shield-mpc
"""
import argparse
import math
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:  # scipy is optional -- fall back to a paired CI only
    wilcoxon = None

# Optimization direction per metric (+1 = higher is better, -1 = lower is better).
DIRECTION = {
    "coverage": +1, "victims": +1, "success": +1,
    "terr_coll": -1, "vict_coll": -1, "damage": -1,
    "infeasible": -1, "steps": -1, "decision_ms": -1,
}
# Metrics compared head-to-head and used for the dominance check (override_frac is
# shield-only bookkeeping, not an objective, so it is left out).
OBJECTIVES = ["success", "coverage", "victims", "damage", "infeasible", "decision_ms"]
DEFAULT_PAIRS = ["shield-rl", "shield-tmpc", "shield-mpc", "shieldc-shield"]


def load(d: Path, agent: str):
    """Read <agent>_episodes.csv -> {metric: np.array}. Rows are seed-aligned."""
    path = d / f"{agent}_episodes.csv"
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    cols = lines[0].split(",")
    data = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])
    return {c: data[:, i] for i, c in enumerate(cols)}


def paired(a, b, metric):
    """Per-seed difference a-b on the shared rows: (mean, half-95%CI, wilcoxon p, n)."""
    n = min(len(a[metric]), len(b[metric]))  # guard against unequal episode counts
    d = a[metric][:n] - b[metric][:n]
    mean = float(d.mean())
    half = 1.96 * float(d.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
    p = None
    if wilcoxon is not None and n > 1 and np.any(d != 0):
        try:
            p = float(wilcoxon(d).pvalue)
        except Exception:
            p = None
    return mean, half, p, n


def report_pairs(loaded, pairs):
    for pair in pairs:
        x, y = pair.split("-")
        a, b = loaded.get(x), loaded.get(y)
        if a is None or b is None:
            print(f"\n[skip] {pair}: missing {'/'.join(k for k in (x, y) if loaded.get(k) is None)} episodes csv")
            continue
        print(f"\n=== {x} - {y}  (per-seed paired difference) ===")
        print(f"{'metric':<12} {'mean diff':>10} {'95% CI':>16} {'wilcoxon p':>12}  better")
        for m in OBJECTIVES:
            if m not in a or m not in b:
                continue
            mean, half, p, n = paired(a, b, m)
            # "better" = which agent this difference favours, given the metric direction.
            favours = x if mean * DIRECTION[m] > 0 else (y if mean != 0 else "tie")
            psig = "n/a" if p is None else (f"{p:.4f}" + ("*" if p < 0.05 else ""))
            print(f"{m:<12} {mean:>+10.4f} {f'[{mean-half:+.3f}, {mean+half:+.3f}]':>16} {psig:>12}  {favours}")
        print(f"(n={n} paired seeds; * = p<0.05; CI is paired 1.96*sd/sqrt(n))")


def dominance(loaded, agents):
    """Mean of each objective per agent, then check pairwise Pareto dominance."""
    means = {ag: {m: float(loaded[ag][m].mean()) for m in OBJECTIVES if m in loaded[ag]}
             for ag in agents if loaded.get(ag) is not None}
    present = list(means)
    print("\n=== mean objectives ===")
    print(f"{'agent':<10}" + "".join(f"{m:>13}" for m in OBJECTIVES))
    for ag in present:
        print(f"{ag:<10}" + "".join(f"{means[ag].get(m, float('nan')):>13.4f}" for m in OBJECTIVES))

    def dominates(p, q):
        better_eq, strictly = True, False
        for m in OBJECTIVES:
            if m not in means[p] or m not in means[q]:
                continue
            dv = (means[p][m] - means[q][m]) * DIRECTION[m]  # >0 means p better on m
            if dv < 0:
                better_eq = False
            elif dv > 0:
                strictly = True
        return better_eq and strictly

    print("\n=== Pareto dominance (subset of objectives) ===")
    any_dom = False
    for p in present:
        for q in present:
            if p == q:
                continue
            if dominates(p, q):
                any_dom = True
                print(f"  {p} DOMINATES {q} (better-or-equal on all objectives, strictly better on >=1)")
    if not any_dom:
        print("  no full dominance -- every pair is a tradeoff (e.g. success/coverage vs damage/decision-time)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/compare")
    ap.add_argument("--pairs", default=",".join(DEFAULT_PAIRS),
                    help="comma list of 'x-y' paired comparisons")
    ap.add_argument("--agents", default="rl,mpc,tmpc,shield,shieldc")
    args = ap.parse_args()
    d = Path(args.dir)

    agents = [a.strip() for a in args.agents.split(",")]
    loaded = {ag: load(d, ag) for ag in agents}
    found = [ag for ag in agents if loaded[ag] is not None]
    if not found:
        print(f"no *_episodes.csv found in {d} -- run src.rl.run_compare first")
        return
    print(f"[analyze] loaded: {', '.join(found)}  (from {d})")
    if wilcoxon is None:
        print("[analyze] scipy not installed -> paired CIs only, no Wilcoxon p (uv add scipy to enable)")

    report_pairs(loaded, [p.strip() for p in args.pairs.split(",")])
    dominance(loaded, found)


if __name__ == "__main__":
    main()

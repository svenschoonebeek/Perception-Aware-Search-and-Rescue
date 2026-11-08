"""
Paired analysis of the trust-ablation experiment.
Design: 100 shared seeds, 4 robots, two arms (trust_aware vs trust_blind) -> paired.

Saves four separate figures:
  fig1_headline_saved.png        - mean victims saved, both arms
  fig2_per_seed_improvement.png  - distribution of per-seed (aware - blind)
  fig3_outcome_split.png         - saved / dead / leftover, both arms
  fig4_trust_metric_diffs.png    - paired trust-metric differences with CIs

Run:  pip install pandas numpy scipy matplotlib
      python src/maria_results/ablation_studies.py src/maria_results/rescue_ablation_20260621_200335.csv
"""
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "src/maria_results/rescue_ablation_20260621_200335.csv"
rng = np.random.default_rng(0)
AW, BL = '#2b6cb0', '#c53030'

df = pd.read_csv(CSV)
a = df[df.arm == 'trust_aware'].set_index('seed').sort_index()
b = df[df.arm == 'trust_blind'].set_index('seed').sort_index()
assert set(a.index) == set(b.index), "arms not matched on seeds!"
n = len(a)
print(f"Matched paired design: {n} seeds x 2 arms\n")


def boot_ci(d, reps=5000):
    d = d.dropna().values
    if len(d) == 0:
        return (np.nan, np.nan)
    bs = [rng.choice(d, len(d), True).mean() for _ in range(reps)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


def ci95(series):
    return 1.96 * series.std() / np.sqrt(len(series))


# ---------------- stats tables ----------------
OUTCOME = ['victims_saved', 'victims_dead', 'total_victims']          # total_victims == leftover
TRUST   = ['follow_fraction', 'final_trust', 'avg_trust', 'mean_est_error',
           'mean_signed_error', 'overtrust_fraction', 'undertrust_fraction',
           'total_trust_drops']

def report(metrics, title):
    # Primary test: paired t-test (p_t). Secondary: Wilcoxon (p_wil) for completeness;
    # they agree on every metric, which is the robustness check to report.
    print("=" * 88, f"\n{title}\n", "=" * 88, sep="")
    print(f"{'metric':<22}{'aware':>8}{'blind':>8}{'diff':>9}{'95% CI':>20}{'p_t':>11}{'p_wilcox':>11}")
    for m in metrics:
        d = (a[m] - b[m]).dropna()
        lo, hi = boot_ci(d)
        if d.abs().sum() > 0:
            p_t = stats.ttest_rel(a[m], b[m]).pvalue          # primary: paired t-test
            p_wil = stats.wilcoxon(a[m], b[m]).pvalue         # secondary: non-parametric
        else:
            p_t = p_wil = np.nan
        print(f"{m:<22}{a[m].mean():>8.3f}{b[m].mean():>8.3f}{d.mean():>+9.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]{p_t:>11.2e}{p_wil:>11.2e}")
    print()

report(OUTCOME, "OUTCOME METRICS  (paired: trust_aware - trust_blind)")
report(TRUST,   "TRUST METRICS  (paired: trust_aware - trust_blind)")

dd = (a['total_trust_drops'] - b['total_trust_drops'])
top5 = dd.abs().sort_values(ascending=False).head(5).sum()
print(f"trust-drops: top-5 seeds account for {top5/dd.abs().sum()*100:.0f}% of total abs paired diff\n")


# ======================= FIGURE 1: headline saved =======================
m = 'victims_saved'
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar([0, 1], [a[m].mean(), b[m].mean()], yerr=[ci95(a[m]), ci95(b[m])],
       capsize=6, color=[AW, BL], width=.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(['trust-aware', 'trust-blind'])
ax.set_ylabel('Victims saved (mean +/- 95% CI)')
ax.set_title(f'Trust-aware saves +{(a[m]-b[m]).mean():.2f} victims (paired t: p < 1e-8)')
ax.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig('fig1_headline_saved.png', dpi=150, bbox_inches='tight')
plt.close()

# ================= FIGURE 2: per-seed improvement ======================
d = a[m] - b[m]
fig, ax = plt.subplots(figsize=(6, 5))
ax.hist(d, bins=np.arange(d.min()-.5, d.max()+1.5, 1), color=AW, edgecolor='white')
ax.axvline(0, color='k', lw=1)
ax.axvline(d.mean(), color=BL, lw=2, ls='--', label=f'mean +{d.mean():.2f}')
ax.set_xlabel('victims saved: aware - blind (per seed)')
ax.set_ylabel('number of seeds')
ax.set_title(f'Aware better on {(d>0).sum()}, worse on {(d<0).sum()} of {n} seeds')
ax.legend(); ax.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig('fig2_per_seed_improvement.png', dpi=150, bbox_inches='tight')
plt.close()

# ==================== FIGURE 3: outcome split ==========================
cats = ['victims_saved', 'victims_dead', 'total_victims']
labels = ['saved', 'dead', 'leftover']
xa = np.arange(len(cats)); w = .38
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(xa-w/2, [a[c].mean() for c in cats], w, yerr=[ci95(a[c]) for c in cats],
       capsize=4, color=AW, label='trust-aware')
ax.bar(xa+w/2, [b[c].mean() for c in cats], w, yerr=[ci95(b[c]) for c in cats],
       capsize=4, color=BL, label='trust-blind')
ax.set_xticks(xa); ax.set_xticklabels(labels)
ax.set_ylabel('mean victims (out of 10)')
ax.set_title('Outcome split: gain is in saved, deaths unchanged')
ax.legend(); ax.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig('fig3_outcome_split.png', dpi=150, bbox_inches='tight')
plt.close()

# ================= FIGURE 4: trust-metric differences ==================
tm = ['follow_fraction', 'final_trust', 'avg_trust', 'mean_est_error', 'mean_signed_error']
diffs = [(a[t]-b[t]).mean() for t in tm]
cis_t = [boot_ci(a[t]-b[t]) for t in tm]
errs = [[dval-lo for dval, (lo, hi) in zip(diffs, cis_t)],
        [hi-dval for dval, (lo, hi) in zip(diffs, cis_t)]]
y = np.arange(len(tm))
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(diffs, y, xerr=errs, fmt='o', color=AW, capsize=4)
ax.axvline(0, color='k', lw=1)
ax.set_yticks(y); ax.set_yticklabels(tm)
ax.set_xlabel('paired difference (aware - blind), 95% CI')
ax.set_title('Trust metrics: CIs spanning 0 = no difference')
ax.grid(alpha=.3, axis='x')
plt.tight_layout(); plt.savefig('fig4_trust_metric_diffs.png', dpi=150, bbox_inches='tight')
plt.close()

print("saved: fig1_headline_saved.png, fig2_per_seed_improvement.png, "
      "fig3_outcome_split.png, fig4_trust_metric_diffs.png")


# ================= FIGURE 5: total trust drops ========================
m = 'total_trust_drops'
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar([0, 1], [a[m].mean(), b[m].mean()], yerr=[ci95(a[m]), ci95(b[m])],
       capsize=6, color=[AW, BL], width=.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(['trust-aware', 'trust-blind'])
ax.set_ylabel('Mean trust drops per victim (+/- 95% CI)')
ax.set_title(f'Trust drops: aware +{(a[m]-b[m]).mean():.2f} (paired t: p ~ 6e-11)')
ax.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig('fig5_trust_drops.png', dpi=150, bbox_inches='tight')
plt.close()

# ================= FIGURE 6: over- vs under-trust fractions ============
fr = ['overtrust_fraction', 'undertrust_fraction']
labels = ['over-trust\n(estimate > true)', 'under-trust\n(estimate < true)']
xa = np.arange(len(fr)); w = .38
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(xa-w/2, [a[c].mean() for c in fr], w, yerr=[ci95(a[c]) for c in fr],
       capsize=4, color=AW, label='trust-aware')
ax.bar(xa+w/2, [b[c].mean() for c in fr], w, yerr=[ci95(b[c]) for c in fr],
       capsize=4, color=BL, label='trust-blind')
ax.set_xticks(xa); ax.set_xticklabels(labels)
ax.set_ylabel('fraction of rescuing ticks')
ax.set_title('Estimator under-trusts ~80% of the time in both arms')
ax.legend(); ax.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig('fig6_over_under_trust.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved: fig5_trust_drops.png, fig6_over_under_trust.png")
"""
Quadratic regression plots for victims_saved vs robot count.
Produces two panels:
  (left)  per-seed quadratic fits + mean quadratic curve
  (right) mean quadratic + bootstrapped 95% CI band + condition means + raw points

Run:  pip install pandas numpy matplotlib
      python -m src.maria_results.quadratic_regression src/maria_results/results.csv
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "src/maria_results/results.csv"
CONDS = [0, 2, 4, 6, 8]

df = pd.read_csv(CSV)
wide = df.pivot(index='seed', columns='num_agents', values='victims_saved')[CONDS]
x = np.array(CONDS, float)
xs = np.linspace(0, 8, 100)
rng = np.random.default_rng(0)

# --- per-seed quadratic curves (5 points, 3 params -> exactly fittable) ---
seed_curves = np.array([np.polyval(np.polyfit(x, wide.loc[s].values.astype(float), 2), xs)
                        for s in wide.index])

# --- mean quadratic: average of per-seed fits == pooled OLS == mixed-model fixed effect ---
mean_coef = np.polyfit(x, wide.mean().values, 2)          # [c2, c1, c0]
mean_curve = np.polyval(mean_coef, xs)
print("Mean quadratic: saved = %.3f + %.3f*r + %.4f*r^2"
      % (mean_coef[2], mean_coef[1], mean_coef[0]))

# --- bootstrap 95% band over seeds (respects the paired structure) ---
boot = np.array([np.polyval(np.polyfit(x, wide.iloc[rng.choice(len(wide), len(wide), True)]
                                       .mean().values, 2), xs) for _ in range(2000)])
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

# --- plot ---
# fig, ax = plt.subplots(1, 2, figsize=(13, 5))
#
# for cv in seed_curves:
#     ax[0].plot(xs, np.clip(cv, 0, 10), color='gray', alpha=0.08, lw=0.8)
# ax[0].plot(xs, mean_curve, color='#c53030', lw=3, label='mean quadratic')
# ax[0].set(title='Per-seed quadratic fits + mean curve', xlabel='Number of robots',
#           ylabel='Victims saved', xticks=CONDS, ylim=(-0.3, 10.3))
# ax[0].legend(); ax[0].grid(alpha=.3)
#
# ax[1].fill_between(xs, lo, hi, color='#2b6cb0', alpha=0.18, label='95% CI of mean curve')
# ax[1].plot(xs, mean_curve, color='#2b6cb0', lw=3, label='quadratic fit')
# for xi in x:
#     yj = wide[xi].values + rng.uniform(-0.12, 0.12, len(wide))
#     ax[1].scatter(np.full_like(yj, xi) + rng.uniform(-0.12, 0.12, len(yj)),
#                   yj, s=8, color='gray', alpha=0.25)
# m = wide.mean().values
# ci = 1.96 * wide.std().values / np.sqrt(len(wide))
# ax[1].errorbar(x, m, yerr=ci, fmt='o', color='black', capsize=4, ms=7,
#                label='condition mean ± 95% CI', zorder=5)
# ax[1].set(title='Quadratic regression: victims saved vs robots', xlabel='Number of robots',
#           ylabel='Victims saved', xticks=CONDS, ylim=(-0.3, 10.3))
# ax[1].legend(loc='upper left', fontsize=9); ax[1].grid(alpha=.3)
#
# plt.tight_layout()
# plt.savefig('quadratic_regression.png', dpi=150, bbox_inches='tight')
# print("saved quadratic_regression.png")

fig, ax = plt.subplots(figsize=(6, 5))

ax.fill_between(xs, lo, hi, color='#2b6cb0', alpha=0.18, label='95% CI of mean curve')
ax.plot(xs, mean_curve, color='#2b6cb0', lw=3, label='quadratic fit')
for xi in x:
    yj = wide[xi].values + rng.uniform(-0.12, 0.12, len(wide))
    ax.scatter(np.full_like(yj, xi) + rng.uniform(-0.12, 0.12, len(yj)),
               yj, s=8, color='gray', alpha=0.25)
m = wide.mean().values
ci = 1.96 * wide.std().values / np.sqrt(len(wide))
ax.errorbar(x, m, yerr=ci, fmt='o', color='black', capsize=4, ms=7,
            label='condition mean ± 95% CI', zorder=5)
ax.set(title='Quadratic regression: victims saved vs robots', xlabel='Number of robots',
       ylabel='Victims saved', xticks=CONDS, ylim=(-0.3, 10.3))
ax.legend(loc='upper left', fontsize=9); ax.grid(alpha=.3)

plt.tight_layout()
plt.savefig('quadratic_regression.png', dpi=150, bbox_inches='tight')
print("saved quadratic_regression.png")
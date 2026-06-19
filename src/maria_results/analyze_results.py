"""
Statistical analysis for matched search-and-rescue simulation results.
Design: 5 robot conditions (0,2,4,6,8) x 100 shared seeds = paired/repeated measures.

Run:  pip install pandas numpy matplotlib
      python -m src.maria_results.analyze_results src/maria_results/results.csv
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "src/maria_results/results.csv"
CONDS = [0, 2, 4, 6, 8]
PRIMARY = "victims_saved"   # primary outcome

# ---- load & clean -----------------------------------------------------------
df = pd.read_csv(CSV)
df = df[['seed', 'num_agents', 'outcome', 'victims_saved',
         'victims_dead', 'victims_left']]
# NOTE: 'total_victims' is actually LEFTOVER (saved+dead+leftover=10). Renamed.
df = df.rename(columns={'total_victims': 'leftover'})

# sanity: matched design?
seedsets = {n: set(g['seed']) for n, g in df.groupby('num_agents')}
assert all(s == seedsets[0] for s in seedsets.values()), "Seeds not matched!"
print(f"Matched design confirmed: {len(seedsets[0])} shared seeds x {len(CONDS)} conditions\n")


# ---- 1. descriptives --------------------------------------------------------
def describe(col):
    g = df.groupby('num_agents')[col]
    n = g.size()
    return pd.DataFrame({
        'mean':   g.mean().round(2),
        'SD':     g.std().round(2),
        'median': g.median(),
        'IQR':    (g.quantile(.75) - g.quantile(.25)).round(2),
        'CI95_lo': (g.mean() - 1.96 * g.std() / np.sqrt(n)).round(2),
        'CI95_hi': (g.mean() + 1.96 * g.std() / np.sqrt(n)).round(2),
    })

print("=" * 60, "\nDESCRIPTIVES\n", "=" * 60, sep="")
for col in ['victims_saved', 'victims_dead', 'victims_left']:
    print(f"\n{col}:\n{describe(col).to_string()}")

# wide (paired) layout for the primary outcome
wide = df.pivot(index='seed', columns='num_agents', values=PRIMARY)[CONDS]
n, k = wide.shape

# ---- plots ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))

# mean saved with 95% CI
g = df.groupby('num_agents')['victims_saved']
means, sds, ns = g.mean(), g.std(), g.size()
ci = 1.96 * sds / np.sqrt(ns)
ax.errorbar(CONDS, means, yerr=ci, marker='o', capsize=5, color='#2b6cb0')
ax.set(xlabel='Number of robots', ylabel='Victims saved (mean ± 95% CI)',
       title='Victims saved vs robot count')
ax.set_xticks(CONDS); ax.grid(alpha=.3)

plt.tight_layout()
plt.savefig('analysis_plot.png', dpi=150, bbox_inches='tight')
print("\nSaved figure -> analysis_plot.png")
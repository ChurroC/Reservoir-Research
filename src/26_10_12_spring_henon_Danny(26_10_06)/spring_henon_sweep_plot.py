from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent
d = np.load(here / "results" / "sweep.npz")
sigma, scores, baseline = d["sigma"], d["scores"], d["baseline"]
names = list(d["metric_names"])
sigmas = np.unique(sigma)


def mean_sem(col):
    # mean and standard error over seeds (NaN = failed trial) at each sigma
    rows = [scores[sigma == s, col] for s in sigmas]
    mean = np.array([np.nanmean(r) for r in rows])
    sem = np.array([np.nanstd(r) / np.sqrt(np.sum(~np.isnan(r))) for r in rows])
    return mean, sem


fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=144, sharex=True)

# left: next-step R² with every trial
col = names.index("x(t+1)")
axes[0].scatter(sigma, scores[:, col], s=5, c="k", alpha=0.2, label="single draw")
mean, sem = mean_sem(col)
axes[0].errorbar(sigmas, mean, yerr=sem, fmt="o-", c="SteelBlue", capsize=2, label="mean ± SE")
axes[0].axhline(baseline[col], ls="--", c="firebrick", label="no reservoir")
axes[0].set(xlabel=r"$\sigma$", ylabel="R² for x(t+1)")
axes[0].legend(fontsize=7)

# right: the other metrics
for i, name in enumerate(names):
    mean, sem = mean_sem(i)
    axes[1].errorbar(sigmas, mean, yerr=sem, fmt="o-", ms=3, capsize=2, label=name)
axes[1].set(xlabel=r"$\sigma$", ylabel="R² (mean ± SE)")
axes[1].legend(fontsize=7)
for ax in axes:
    ax.set_box_aspect(1)

fig.tight_layout()
(here / "plots").mkdir(exist_ok=True)
fig.savefig(here / "plots" / "sweep.pdf", bbox_inches="tight")
plt.show()

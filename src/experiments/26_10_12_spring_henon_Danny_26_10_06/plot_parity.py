"""
Figure for the parity test (results/parity.csv, written by design_map.py --parity).
Rows: structures; columns: functions of a symmetric random input that the readout must reconstruct,
odd on the left, even on the right. Colour and number: test R^2 (negative values shown as 0).
Run:  python plot_parity.py   ->  plots/parity.pdf
"""
import csv
import sys
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
here = Path(__file__).parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results" / "parity.csv"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "plots" / "parity.pdf"

# rows grouped by element type (bracket on the left); each row labelled by its disorder
GROUPS = [("1 well\n(monostable)", [("A", "uniform"), ("B", "stiffness"), ("C", "rest length")]),
          ("2 wells\n(bistable)",   [("D", "uniform"), ("E", "stiffness"), ("F", "rest length")])]
ROWS = [row for _, rows_ in GROUPS for row in rows_]
LABELS = {"u(t-1)": r"$u_{t-1}$", "u(t-2)": r"$u_{t-2}$", "u(t)^3": r"$u_t^3$",
          "u(t)u(t-1)u(t-2)": r"$u_t u_{t-1} u_{t-2}$",
          "u(t)^2": r"$u_t^2$", "u(t)u(t-1)": r"$u_t u_{t-1}$", "u(t-1)^2": r"$u_{t-1}^2$",
          "u(t)u(t-2)": r"$u_t u_{t-2}$"}

rows = list(csv.DictReader(open(src)))
cols = []
for r in rows:                              # keep the order written by design_map.py: odd, then even
    if (r["parity"], r["target"]) not in cols:
        cols.append((r["parity"], r["target"]))
val = {(r["struct"][0], r["parity"], r["target"]): float(r["r2"]) for r in rows}
M = np.array([[max(val.get((L, par, t), np.nan), 0.0) for par, t in cols] for L, _ in ROWS])
n_odd = sum(par == "odd" for par, _ in cols)

fig, ax = plt.subplots(figsize=(6.5, 2.6), constrained_layout=True)
BRACKET_X = -0.14   # axes fraction; move left if the row labels get longer
cmap = cmr.get_sub_cmap(cmr.eclipse, 0, 0.9)
im = ax.pcolormesh(np.arange(len(cols) + 1), np.arange(len(ROWS) + 1), M, cmap=cmap, vmin=0, vmax=1,
                   shading="flat", edgecolors="face", linewidth=0.4)
for i in range(len(ROWS)):
    for j in range(len(cols)):
        v = M[i, j]
        if np.isfinite(v):
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="k" if v > 0.6 else "w")
ax.axvline(n_odd, color="w", lw=2.5)                        # odd | even divider
ax.axhline(len(GROUPS[0][1]), color="w", lw=2.5)           # 1 well | 2 wells divider
ax.set_xticks(np.arange(len(cols)) + 0.5)
ax.set_xticklabels([LABELS.get(t, t) for _, t in cols], fontsize=8)
ax.set_yticks(np.arange(len(ROWS)) + 0.5)
ax.set_yticklabels([lab for _, lab in ROWS], fontsize=8)
ax.invert_yaxis()
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
# bracket and label for each group, left of the row labels (x in axes fraction, y in data units)
trans = ax.get_yaxis_transform()
start = 0
for glabel, grows in GROUPS:
    y0, y1 = start + 0.12, start + len(grows) - 0.12
    ax.plot([BRACKET_X + 0.012, BRACKET_X, BRACKET_X, BRACKET_X + 0.012], [y0, y0, y1, y1],
            transform=trans, clip_on=False, color="0.3", lw=0.8)
    ax.text(BRACKET_X - 0.012, (y0 + y1) / 2, glabel, transform=trans, ha="right", va="center",
            fontsize=8, linespacing=1.1)
    start += len(grows)
ax.text(n_odd / 2, -0.35, "odd functions of the input", ha="center", va="bottom", fontsize=8)
ax.text(n_odd + (len(cols) - n_odd) / 2, -0.35, "even functions of the input", ha="center", va="bottom", fontsize=8)
cb = fig.colorbar(im, ax=ax, ticks=[0, 0.5, 1], fraction=0.04, pad=0.02)
cb.ax.set_title(r"$R^2$", fontsize=8)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8, length=0)

dst.parent.mkdir(exist_ok=True)
fig.savefig(dst, bbox_inches="tight")
fig.savefig(dst.with_suffix(".png"), bbox_inches="tight", dpi=200)
print(f"Saved -> {dst}")

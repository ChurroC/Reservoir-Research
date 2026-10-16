"""
Figure for the white paper from results/design_map.csv (written by design_map.py --grid).
  (a) fraction of designs with slow enough loading (Omega >= OMEGA_MIN) reaching at least a given R^2
  (b-d) R^2 over coupling kappa and loading interval Omega, averaged over load amplitude f and random draws
Run:  python plot_design_map.py   ->  plots/design_map.pdf
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"]=42
import numpy as np
import cmasher as cmr

here = Path(__file__).parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results" / "design_map.csv"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "plots" / "design_map.pdf"

# structure key -> (short label, color, linestyle). Symmetric structures dashed (secondary encoding).
STYLE = {
    "A": ("monostable, uniform", "C0", "-", 0.5),
    "D": ("bistable, uniform", "C0", "--", 1.0),
    "B": ("monostable, stiffness disorder", "C1", "-", 0.5),
    "E": ("bistable, stiffness disorder", "C1", "--", 1.0),
    "C": ("monostable, rest length disorder", "C2", "-", 0.5),
    "F": ("bistable, rest length disorder", "C2", "--", 1.0),
}
HEAT = [("A", "monostable,\nuniform"), ("E", "bistable,\nstiffness disorder")]
INK, INK2, GRID = "0.1", "0.3", "0.9"
OMEGA_MIN = 0   # panel (a): loads no faster than this many natural time scales apart

rows = list(csv.DictReader(open(src)))
pts = {}
for r in rows:
    k = (r["struct"][0], float(r["kappa"]), float(r["f"]), float(r["Omega"]))
    pts.setdefault(k, []).append(float(r["r2"]))
# one value per design point (mean over draws); unstable runs count as R^2 = 0
val = {k: (0.0 if np.isnan(v).any() else max(np.mean(v), 0.0)) for k, v in pts.items()}
kappas = sorted({k[1] for k in val}); fs = sorted({k[2] for k in val}); omegas = sorted({k[3] for k in val})

# plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2,
#                      "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig = plt.figure(figsize=(6.5, 2.5), constrained_layout=True)
gs = fig.add_gridspec(1, 1 + len(HEAT), wspace=0.1)

# ---------- (a) fraction of design space with R^2 >= threshold ----------
ax = fig.add_subplot(gs[0])
thr = np.linspace(0, 1, 401)
ends = []
for s, (lab, col, ls, alpha) in STYLE.items():
    v = np.array([x for k, x in val.items() if k[0] == s and k[3] >= OMEGA_MIN])
    if v.size == 0:
        continue
    frac = [(v >= t).mean() for t in thr]
    ax.plot(thr, frac, color=col, lw=1.6, ls=ls, label=lab, alpha=alpha)
    ends.append((s, np.median(v)))
ax.set(
    xlabel=r"prediction accuracy $R^2$",
    ylabel=rf"fraction $\geq R^2$ ($\Omega \geq {OMEGA_MIN:g}$)",
    xlim=(-0.05, 1.05),
    ylim=(-0.05, 1.05)
)
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.grid(color=GRID, lw=0.5);
ax.set_aspect(1)
fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=7, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.2), handlelength=2.6, labelcolor=INK, columnspacing=1.6)

# ---------- (b-d) heatmaps over (kappa, Omega), mean over f ----------
cmap = cmr.get_sub_cmap(cmr.eclipse, 0, 0.9)
CELL_FONTSIZE = 6   # numbers written in the heatmap cells
for i, (s, title) in enumerate(HEAT):
    ax = fig.add_subplot(gs[1 + i])
    M = np.array([[np.mean([val[(s, kp, f, om)] for f in fs]) for kp in kappas] for om in omegas])
    # vector cells (one rectangle per design point) instead of an embedded image: imshow writes a
    # low-bit indexed image when few colours are used, which Illustrator can misread as one colour
    im = ax.pcolormesh(np.arange(len(kappas) + 1) - 0.5, np.arange(len(omegas) + 1) - 0.5, M,
                       cmap=cmap, vmin=0, vmax=1, shading="flat", edgecolors="face", linewidth=0.4)  # face-coloured edges hide seams
    ax.grid(False)
    for r, om in enumerate(omegas):                           # value in each cell; dark text on light cells
        for c, kp in enumerate(kappas):
            v = M[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=CELL_FONTSIZE,
                    color="k" if v > 0.6 else "w")
    ax.set_xticks(range(len(kappas)));
    ax.set_xticklabels([f"{k:g}" for k in kappas], fontsize=8)
    ax.set_yticks(range(len(omegas)))
    ax.set_yticklabels([f"{om:g}" for om in omegas] if i == 0 else [], fontsize=8)
    ax.set_xlabel(r"coupling $\kappa$")
    ax.set_box_aspect(1)
    if i == 0:
        ax.set_ylabel(r"loading interval $\Omega$", labelpad=2)
    ax.set_title(title, loc="center", fontsize=7.5, color=INK)
    ax.tick_params(length=0)

cax = ax.inset_axes([1.05, 0, 0.05, 1])
cb = fig.colorbar(im, cax=cax, ax=ax, ticks=[0, 0.5, 1])
cb.ax.set_title(r"$\langle R^2 \rangle_f$", fontsize=8)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8, length=0)

dst.parent.mkdir(exist_ok=True)
fig.savefig(dst, bbox_inches="tight")
fig.savefig(dst.with_suffix(".png"), bbox_inches="tight", dpi=144)
print(f"medians (Omega >= {OMEGA_MIN:g}):", {s: round(float(m), 3) for s, m in ends})
print(f"Saved -> {dst}")

"""
Ablation: does the benefit of disorder come from prestress imbalance?

In spring_henon.py the between-cell springs have rest length 1.1 but span dx = 2.5,
so every one is under tension ~1.4 k. With disordered k, neighbouring tensions no
longer cancel and each middle node feels a static sideways force
    f_i = 1.4 (k_right - k_left),
which breaks the x -> -x symmetry of each cell. A symmetric element driven about its
centre only makes odd harmonics (x, x^3, ...); a static bias adds an x^2 response,
and x(t)^2 is exactly what next-step Henon prediction needs.

Here we set the between-cell rest length to dx, so those springs start unstretched:
no tension, no static bias. Their stiffness along the chain (k) is unchanged, so the
chain's linear x-stiffness is the same; only the prestress and its imbalance go away.

Same seeds as spring_henon_sweep.py, so each disorder pattern is compared with itself.

Run:  python spring_henon_noprestress.py            (default sigmas below)
      python spring_henon_noprestress.py 0 0.75 1   (custom sigmas)
"""
import os
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import spring_henon as sh
from spring_henon_sweep import metrics, metric_names

SIGMAS = [float(s) for s in sys.argv[1:]] or [0, 0.5, 0.75, 1.0, 1.5]
N_SEEDS = 30
REST_BETWEEN = sh.dx  # unstretched between-cell springs
here = Path(__file__).parent
out_file = here / "results" / "sweep_noprestress.npz"


def remove_prestress():
    # runs in every worker (macOS spawns fresh processes): edit the module's rest array in place
    sh.rest[sh.n_in:] = REST_BETWEEN


def trial(task):
    sigma, seed = task
    assert sh.rest[-1] == REST_BETWEEN and sh.rest[0] == sh.rest_len  # ablation is active
    X = sh.run_reservoir(sh.spring_constants(sigma, seed))
    if not np.isfinite(X).all():
        return [np.nan] * len(metric_names)
    return metrics(X)


if __name__ == "__main__":
    tasks = [(0.0, 0)] + [(s, seed) for s in SIGMAS if s > 0 for seed in range(N_SEEDS)]
    print(f"{len(tasks)} runs, between-cell rest length {sh.rest_len} -> {REST_BETWEEN}")
    with Pool(initializer=remove_prestress) as pool:
        scores = []
        for i, sc in enumerate(pool.imap(trial, tasks)):
            scores.append(sc)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)} done", flush=True)
    scores = np.array(scores)
    sig = np.array([t[0] for t in tasks]); seed = np.array([t[1] for t in tasks])
    out_file.parent.mkdir(exist_ok=True)
    np.savez(out_file, sigma=sig, seed=seed, scores=scores, metric_names=metric_names)

    # ---------- comparison with the original (prestressed) sweep ----------
    ref = np.load(here / "results" / "sweep.npz")
    rs, rseed, rsc = ref["sigma"], ref["seed"], ref["scores"]
    col = list(metric_names).index("x(t+1)")
    x2 = list(metric_names).index("x(t)^2")

    def pick(S, SE, SC, s, k, c):
        m = (S == s) & (SE == k)
        return SC[m, c][0] if m.any() else np.nan

    base_p = pick(rs, rseed, rsc, 0, 0, col)
    base_n = pick(sig, seed, scores, 0, 0, col)
    print(f"\nNext-step R^2 for x(t+1). Uniform chain: prestressed {base_p:.3f}, no prestress {base_n:.3f}")
    print(f"{'sigma':>6} | {'prestressed':>17} | {'no prestress':>17} | {'paired diff (no-pre minus pre)':>32} | {'x^2 recall pre / no-pre':>24}")
    for s in SIGMAS:
        if s == 0:
            continue
        p = np.array([pick(rs, rseed, rsc, s, k, col) for k in range(N_SEEDS)])
        n = np.array([pick(sig, seed, scores, s, k, col) for k in range(N_SEEDS)])
        p2 = np.array([pick(rs, rseed, rsc, s, k, x2) for k in range(N_SEEDS)])
        n2 = np.array([pick(sig, seed, scores, s, k, x2) for k in range(N_SEEDS)])
        ok = np.isfinite(p) & np.isfinite(n)
        d = n[ok] - p[ok]
        sem = lambda v: np.nanstd(v) / np.sqrt(np.sum(np.isfinite(v)))
        print(f"{s:>6} | {np.nanmean(p):.3f} ± {sem(p):.3f} (max {np.nanmax(p):.2f}) | "
              f"{np.nanmean(n):.3f} ± {sem(n):.3f} (max {np.nanmax(n):.2f}) | "
              f"{d.mean():+.3f} ± {d.std() / np.sqrt(len(d)):.3f}, worse in {np.sum(d < 0)}/{len(d)} | "
              f"{np.nanmean(p2):.3f} / {np.nanmean(n2):.3f}")
    print("\nDisorder gain = mean R^2 at sigma minus the same condition's uniform chain.")
    print("If the gain mostly vanishes without prestress, prestress imbalance is the main mechanism.")
    print(f"Saved -> {out_file}")

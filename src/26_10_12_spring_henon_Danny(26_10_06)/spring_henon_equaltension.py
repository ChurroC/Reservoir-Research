"""
Control: keep the prestress, remove only its imbalance.

Original model: every between-cell spring spans dx = 2.5 with rest length 1.1, so its
tension is T_i = 1.4 k_i. With disordered k_i, neighbouring tensions differ and each
middle node feels a static sideways force 1.4 (k_right - k_left).

Here each between-cell spring keeps its disordered stiffness k_i, but its rest length
is set so that every spring carries the same tension T = 1.4 * k_between_val (the
uniform chain's tension, 5.6):
    r_i = dx - T / k_i.
So: same mean tension and same stiffness disorder as the original, but zero net static
force on interior nodes. At sigma = 0 this is identical to the original chain.

Side effects worth knowing:
  * the springs' sideways (y) stiffness T/L is now uniform instead of disordered;
  * soft springs (k_i < T/dx ~ 2.24) get negative rest lengths. In this model that is
    just a linear spring with extra built-in tension, force = k (L - r_i), and is fine
    as long as L stays far from 0 (it stays near 2.5).

Compare three conditions on the same seeds:
  prestressed (results/sweep.npz) | no prestress (results/sweep_noprestress.npz) | equal tension (this run)

Run:  python spring_henon_equaltension.py            (default sigmas below)
      python spring_henon_equaltension.py 0 0.75 1   (custom sigmas)
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
T_EQUAL = (sh.dx - sh.rest_len) * sh.k_between_val  # = 1.4 * 4.0 = 5.6
here = Path(__file__).parent
out_file = here / "results" / "sweep_equaltension.npz"


def trial(task):
    sigma, seed = task
    k = sh.spring_constants(sigma, seed)
    k_between = k[sh.n_in:]
    # each worker runs one trial at a time, so editing the module's rest array here is safe
    sh.rest[: sh.n_in] = sh.rest_len
    sh.rest[sh.n_in:] = sh.dx - T_EQUAL / k_between
    tension = k_between * (sh.dx - sh.rest[sh.n_in:])
    assert np.allclose(tension, T_EQUAL)  # every between-cell spring carries the same tension
    X = sh.run_reservoir(k)
    if not np.isfinite(X).all():
        return [np.nan] * len(metric_names)
    return metrics(X)


def load(name):
    f = here / "results" / name
    return np.load(f) if f.exists() else None


if __name__ == "__main__":
    tasks = [(0.0, 0)] + [(s, seed) for s in SIGMAS if s > 0 for seed in range(N_SEEDS)]
    print(f"{len(tasks)} runs, equal tension T = {T_EQUAL:.2f} in every between-cell spring")
    with Pool() as pool:
        scores = []
        for i, sc in enumerate(pool.imap(trial, tasks)):
            scores.append(sc)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)} done", flush=True)
    scores = np.array(scores)
    sig = np.array([t[0] for t in tasks]); seed = np.array([t[1] for t in tasks])
    out_file.parent.mkdir(exist_ok=True)
    np.savez(out_file, sigma=sig, seed=seed, scores=scores, metric_names=metric_names,
             tension=T_EQUAL)

    # ---------- three-way comparison on x(t+1) ----------
    col = list(metric_names).index("x(t+1)")
    conds = {"prestressed": load("sweep.npz"), "no prestress": load("sweep_noprestress.npz"),
             "equal tension": dict(sigma=sig, seed=seed, scores=scores)}
    conds = {k: v for k, v in conds.items() if v is not None}

    def column(d, s):
        out = np.full(N_SEEDS if s > 0 else 1, np.nan)
        for j in range(len(out)):
            m = (d["sigma"] == s) & (d["seed"] == j)
            if m.any():
                out[j] = d["scores"][m, col][0]
        return out

    base = {k: column(v, 0)[0] for k, v in conds.items()}
    print("\nNext-step R^2 for x(t+1).  Uniform chain: "
          + ", ".join(f"{k} {b:.3f}" for k, b in base.items()))
    print("Per sigma: mean ± SE | std across seeds | max | gain over own uniform chain")
    for s in SIGMAS:
        if s == 0:
            continue
        print(f"sigma = {s}")
        cols = {k: column(v, s) for k, v in conds.items()}
        for k, c in cols.items():
            n = np.sum(np.isfinite(c))
            print(f"  {k:>14}: {np.nanmean(c):.3f} ± {np.nanstd(c) / np.sqrt(n):.3f} | "
                  f"std {np.nanstd(c):.3f} | max {np.nanmax(c):.2f} | gain {np.nanmean(c) - base[k]:+.3f}")
        if "prestressed" in cols:
            d = cols["equal tension"] - cols["prestressed"]
            d = d[np.isfinite(d)]
            print(f"  paired equal-tension minus prestressed: {d.mean():+.3f} ± "
                  f"{d.std() / np.sqrt(len(d)):.3f}, worse in {np.sum(d < 0)}/{len(d)}")
    print("\nReading it: if equal tension keeps the prestressed tail (large std and max), tension itself")
    print("matters and the imbalance does not. If it collapses toward no prestress, the imbalance is what")
    print("creates the rare high-performing chains.")
    print(f"Saved -> {out_file}")

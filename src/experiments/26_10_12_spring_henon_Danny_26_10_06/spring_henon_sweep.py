import os

# one BLAS thread per worker, the parallelism is across trials
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from spring_henon import henon_scaled, n_steps, readout, run_reservoir, spring_constants, u, washout

# ---------- parameters ----------
sigmas = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]
n_seeds = 30  # spring-constant draws per sigma (sigma = 0 is deterministic, run once)
horizons = [1, 2, 3, 4, 5]
metric_names = [f"x(t+{h})" for h in horizons] + ["x(t)^2"]
out_file = Path(__file__).parent / "results" / "sweep.npz"


# ---------- metrics ----------
def metrics(X):
    # R² of a ridge readout of X for: x at each horizon ahead, and the square of the current x
    scores = []
    for h in horizons:
        Y = henon_scaled[washout + h : n_steps + 1, :1]
        scores.append(r2_score(*readout(X[: len(Y)], Y)))
    scores.append(r2_score(*readout(X, u[washout:, :1] ** 2)))
    return scores


def trial(sigma, seed):
    X = run_reservoir(spring_constants(sigma, seed))
    if not np.isfinite(X).all():  # simulation blew up (too-stiff springs)
        return [np.nan] * len(metric_names)
    return metrics(X)


if __name__ == "__main__":
    tasks = [(0, 0)] + [(s, seed) for s in sigmas if s > 0 for seed in range(n_seeds)]
    with Pool() as pool:
        scores = np.array(pool.starmap(trial, tasks))

    # baseline: no reservoir, regress directly on the current Henon state [x(t), y(t)]
    baseline = metrics(u[washout:])

    out_file.parent.mkdir(exist_ok=True)
    np.savez(
        out_file,
        sigma=np.array([t[0] for t in tasks]),
        seed=np.array([t[1] for t in tasks]),
        scores=scores,
        metric_names=metric_names,
        baseline=baseline,
    )

    # summary: mean over seeds for each sigma
    sigma_col = np.array([t[0] for t in tasks])
    print(f"{'sigma':>6} {'failed':>7} " + " ".join(f"{m:>9}" for m in metric_names))
    for s in sigmas:
        rows = scores[sigma_col == s]
        failed = np.isnan(rows[:, 0]).sum()
        print(f"{s:>6} {failed:>3}/{len(rows):<3} " + " ".join(f"{v:>9.4f}" for v in np.nanmean(rows, axis=0)))
    print(f"{'none':>6} {'':>7} " + " ".join(f"{v:>9.4f}" for v in baseline))
    print(f"Saved → {out_file}")

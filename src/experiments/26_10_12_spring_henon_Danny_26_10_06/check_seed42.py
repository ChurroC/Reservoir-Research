# Is seed 42's R² = 0.78 at sigma = 1 a real outlier, or an artifact of the readout?
# spring_henon.py fits x and y jointly (one ridge alpha); the sweep fits x alone.
# Score seed 42 and seeds 0-29 both ways and compare.
import os
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"
from multiprocessing import Pool
import numpy as np
from sklearn.metrics import r2_score
from spring_henon import run_reservoir, spring_constants, readout, y_next

SIGMA = 1.0

def score(seed):
    X = run_reservoir(spring_constants(SIGMA, seed))
    if not np.isfinite(X).all():
        return np.nan, np.nan
    true, pred = readout(X, y_next)               # joint [x, y] fit (as in spring_henon.py)
    joint = r2_score(true[:, 0], pred[:, 0])
    alone = r2_score(*readout(X, y_next[:, :1]))   # x-only fit (as in the sweep)
    return joint, alone

if __name__ == "__main__":
    seeds = [42] + list(range(30))
    with Pool() as pool:
        R = np.array(pool.map(score, seeds))
    s42, rest = R[0], R[1:]
    for i, name in enumerate(["joint [x,y] readout", "x-only readout"]):
        print(f"{name:>20}: seed 42 = {s42[i]:.4f} | seeds 0-29 mean = {np.nanmean(rest[:, i]):.4f}, "
              f"max = {np.nanmax(rest[:, i]):.4f}, # above seed 42 = {(rest[:, i] > s42[i]).sum()}/30")

"""
Re-check the Bayesian-optimization winner from
  src/0_summer/week7_8/1_test_if_working/7_nonline_henon_bay.ipynb  (trial #2568, R^2 = 0.99999999)

Questions:
  1. Is R^2 ~ 1 a numerical artifact?  The winner sits near the integrator's stability limit
     (omega*dt ~ 1.5 with k_wall = 65, m = 0.006).  We rerun with smaller dt, keeping the same
     physical time per Henon step (0.03) and the same input impulse (force held for 0.01).
  2. What is R^2 for x(t+1) alone?  The notebook's r2_score averages x(t+1) and y(t+1), and
     y(t+1) = 0.3 x(t) is trivially linear.
  3. Does it hold across random draws of the structure?  (BO optimized the disorder statistics;
     each trial was a fresh random draw.)  Seed 100 is the notebook's own re-check.
  Optional:  --x-only-input  feeds only x(t) to the input nodes (a single measured load history),
             so the structure must remember x(t-1) to get y(t) = 0.3 x(t-1).

Run:  python check_bo_winner.py
      python check_bo_winner.py --x-only-input
      python check_bo_winner.py --ablate       (which disorder matters; dt = 0.002, x(t) input, 3 seeds)
      python check_bo_winner.py --symmetry     (does disorder help by breaking x -> -x symmetry?)
      python check_bo_winner.py --only-one     (switch ON one kind of disorder at a time, from the ordered chain)
      add  --wall-rest 1.1  to any of these for compressed (bistable) cells; default 1.0 is what was optimized
"""
import os
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import sys
from multiprocessing import Pool

import numpy as np
from dysts.maps import Henon
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from numba import njit

import spring_henon as sh  # geometry (100 cells, dx = 2.5) and the numba spring forces

spring_forces = sh.spring_forces  # module-level name so numba can call it


@njit(cache=True)
def run_sampled(n_h, free, hold, dt, m_inv, c, F_in, in_dofs, pos, conn, k, rest, fixed, rec_dofs):
    # Same velocity-Verlet scheme as sh.run_simulation, but stores only the recorded dofs at the
    # last substep of each Henon step (sh.run_simulation keeps every substep: ~6 GB at dt = 0.001).
    # F_in[h, j] is the force on dof in_dofs[j] during the first `hold` substeps of Henon step h.
    n, d = pos.shape
    ndof = n * d
    mask = np.ones(ndof)
    for node in fixed:
        mask[node * d : node * d + d] = 0
    x = np.zeros(ndof); v = np.zeros(ndof)
    Xd = np.zeros((n_h, len(rec_dofs))); Xv = np.zeros((n_h, len(rec_dofs)))
    U_prev = np.zeros(ndof); U_now = np.zeros(ndof)
    for j in range(len(in_dofs)):
        U_prev[in_dofs[j]] = F_in[0, j]          # step 0, substep 0
    F = spring_forces(conn, x, pos, rest, k, n, d)
    for i in range(1, n_h * free):
        h, sub = i // free, i % free
        U_now[:] = 0.0
        if sub < hold:
            for j in range(len(in_dofs)):
                U_now[in_dofs[j]] = F_in[h, j]
        acc = m_inv * (F - c * v + U_prev) * mask
        x = x + v * dt + 0.5 * acc * dt**2
        F = spring_forces(conn, x, pos, rest, k, n, d)
        acc2 = m_inv * (F - c * (v + 0.5 * acc * dt) + U_now) * mask
        v = v + 0.5 * (acc + acc2) * dt
        U_prev[:] = U_now
        if sub == free - 1:
            for j in range(len(rec_dofs)):
                Xd[h, j] = x[rec_dofs[j]]; Xv[h, j] = v[rec_dofs[j]]
    return Xd, Xv

ABLATE = "--ablate" in sys.argv
SYMMETRY = "--symmetry" in sys.argv
ONLY_ONE = "--only-one" in sys.argv
X_ONLY_INPUT = "--x-only-input" in sys.argv or ABLATE or SYMMETRY or ONLY_ONE
# base wall-spring rest length: 1.0 = geometric (what was optimized), 1.1 = compressed, bistable cells
WALL_REST = float(sys.argv[sys.argv.index("--wall-rest") + 1]) if "--wall-rest" in sys.argv else 1.0

# ---------- BO winner (trial #2568) ----------
P = dict(input_force=14.007215766574747, m_val=0.0062590214410398, m_spread=0.10003117774772569,
         c_val=0.8604972776791918, c_spread=0.056458005379281584, k_wall_val=65.2663180731547,
         k_wall_spread=0.3526340108179594, k_between_val=13.54246039410458,
         k_between_spread=0.21268494142399075, wall_rest_spread=0.31662458351660316,
         between_rest_spread=0.1538020767464257)
ALPHAS = {"BO alpha 1.46": 1.4573173680720782, "recheck alpha 0.01": 0.01}

# dt variants: same physical time per Henon step (3 x 0.01 = 0.03), force held for 0.01
DTS = [(0.01, 3), (0.005, 6), (0.002, 15), (0.001, 30)]
SEEDS = [100, 0, 1, 2, 3]

# ---------- data, as in the notebook ----------
steps, tau = 20_000, 1
tr_chaos = tr_res = int(steps * 0.1)
test_steps = int(steps * 0.2)
total = steps + tr_chaos + tr_res + tau
henon = Henon().make_trajectory(total)[tr_chaos:]
scaler = StandardScaler().fit(henon[:-test_steps])
hs = scaler.transform(henon)

N, dims = sh.N, sh.dims
num_nodes = len(sh.nodes_pos)
N_step = int(N / 11)
target_cells = np.arange(1, 11) * N_step          # notebook: cells 9, 18, ..., 90
target_mid = sh.mid[target_cells]


def lognormal(rng, mean, spread, size):
    return rng.lognormal(np.log(mean) - spread**2 / 2, spread, size)


def structure(seed, P=P):
    # same draw order as the notebook: mass, damping, k_wall, k_between, wall rest, between rest
    rng = np.random.default_rng(seed)
    m = lognormal(rng, P["m_val"], P["m_spread"], num_nodes)
    c = lognormal(rng, P["c_val"], P["c_spread"], num_nodes)
    k_wall = lognormal(rng, P["k_wall_val"], P["k_wall_spread"], N)        # shared by a cell's 2 springs
    k_btw = lognormal(rng, P["k_between_val"], P["k_between_spread"], N - 1)
    r_wall = lognormal(rng, 1.0, P["wall_rest_spread"], N)                  # x geometric length 1
    r_btw = lognormal(rng, 1.0, P["between_rest_spread"], N - 1)            # x geometric length 2.5
    # sh.conn order: [bottom->mid for all cells, mid->top for all cells, between]
    k = np.concatenate([k_wall, k_wall, k_btw])
    rest = np.concatenate([WALL_REST * r_wall, WALL_REST * r_wall, sh.dx * r_btw])
    return np.repeat(1 / m, dims), np.repeat(c, dims), k, rest


def trial(task):
    dt, free = task[0]
    seed = task[1]
    zeroed = task[2] if len(task) > 2 else ()
    params = {**P, **{key: 0.0 for key in zeroed}}
    m_inv, c, k, rest = structure(seed, params)
    hold = max(1, round(0.01 / dt))                   # substeps the force is applied for
    comps = [0 if X_ONLY_INPUT else i % dims for i in range(len(target_mid))]
    F_in = np.ascontiguousarray(hs[:, comps] * P["input_force"])
    in_dofs = (target_mid * dims).astype(np.int64)
    rec = (sh.mid * dims).astype(np.int64)
    Xd, Xv = run_sampled(len(hs), free, hold, dt, m_inv, c, F_in, in_dofs,
                         sh.nodes_pos, sh.conn, k, rest, sh.fixed, rec)
    X = np.column_stack([Xd, Xv])
    pos = sh.nodes_pos[sh.mid, 0] + Xd
    if not np.isfinite(X).all() or (np.diff(pos, axis=1) <= 0.01).any():
        return task, None
    Xs = X[:-tau][tr_res:]
    Y = hs[tr_res + tau :]
    Xtr, Xte, Ytr, Yte = Xs[:-test_steps], Xs[-test_steps:], Y[:-test_steps], Y[-test_steps:]
    out = {}
    for name, a in ALPHAS.items():                    # centred closed-form ridge, as in the notebook
        xm, ym = Xtr.mean(0), Ytr.mean(0)
        A = (Xtr - xm).T @ (Xtr - xm)
        A.flat[:: A.shape[0] + 1] += a
        w = np.linalg.solve(A, (Xtr - xm).T @ (Ytr - ym))
        pred = (Xte - xm) @ w + ym
        out[name] = (r2_score(Yte, pred), r2_score(Yte[:, 0], pred[:, 0]), r2_score(Yte[:, 1], pred[:, 1]))
    return task, out


SPREADS = {"mass": "m_spread", "damping": "c_spread", "wall stiffness": "k_wall_spread",
           "between stiffness": "k_between_spread", "wall rest length": "wall_rest_spread",
           "between rest length": "between_rest_spread"}


def run_ablation(symmetry=False, only_one=False):
    # Means fixed at the optimized values; switch disorder off all at once, or one kind at a time.
    configs = {"full optimized disorder": ()}
    configs["NO disorder (all spreads 0)"] = tuple(SPREADS.values())
    if only_one:
        # start from the ordered chain and switch ON one kind of disorder at a time
        for name, key in SPREADS.items():
            configs[f"ONLY {name} disorder"] = tuple(k for k in SPREADS.values() if k != key)
    elif symmetry:
        # With every spring at its geometric length, each mass's dynamics is odd in x, so the
        # response has only odd powers of the input and cannot make the x(t)^2 that x(t+1) needs.
        # Only rest-length disorder can break that symmetry.
        rest = ("wall_rest_spread", "between_rest_spread")
        others = tuple(k for k in SPREADS.values() if k not in rest)
        configs["no rest-length disorder (others on)"] = rest
        configs["ONLY wall rest-length disorder"] = others + ("between_rest_spread",)
        configs["ONLY between rest-length disorder"] = others + ("wall_rest_spread",)
    else:
        for name, key in SPREADS.items():
            configs[f"without {name} disorder"] = (key,)
    dt_free, seeds = (0.002, 15), [100, 0, 1]
    tasks = [(dt_free, s, z) for z in configs.values() for s in seeds]
    print(f"Base wall rest length {WALL_REST} (1.0 = optimized, geometric; >1 = compressed, bistable)")
    print(f"Ablation: {len(tasks)} runs, dt = {dt_free[0]}, input = x(t) only, seeds {seeds}")
    with Pool() as pool:
        res = dict(pool.imap(trial, tasks))
    for alpha in ALPHAS:
        print(f"\nRidge {alpha}.  R^2 for x(t+1), per seed and mean")
        print(f"  {'configuration':38s} " + " ".join(f"{'seed ' + str(s):>9}" for s in seeds) + f" {'mean':>9}")
        for name, z in configs.items():
            vals = [res[(dt_free, s, z)] for s in seeds]
            nums = [np.nan if v is None else v[alpha][1] for v in vals]
            cells = ["   blowup" if v is None else f"{x:9.4f}" for v, x in zip(vals, nums)]
            print(f"  {name:38s} " + " ".join(cells) + f" {np.nanmean(nums):9.4f}")
    if only_one:
        print("\nReading it: compare each ONLY row with the NO-disorder row; the biggest jump is the kind of")
        print("disorder that, on its own, does the most. Symmetry-breaking kinds (rest lengths) should lead.")
    elif symmetry:
        print("\nPrediction: 'no rest-length disorder' collapses to the ordered value (~0.27), while either")
        print("rest-length disorder alone stays near 1. That would mean disorder helps by breaking mirror symmetry.")
    else:
        print("\nReading it: a big drop when one spread is removed means that disorder does real work.")
        print("If no single removal hurts but removing all does, the kinds of disorder can substitute for each other.")


if __name__ == "__main__" and (ABLATE or SYMMETRY or ONLY_ONE):
    run_ablation(symmetry=SYMMETRY, only_one=ONLY_ONE)
    sys.exit()

if __name__ == "__main__":
    tasks = [(d, s) for d in DTS for s in SEEDS]
    print(f"Base wall rest length {WALL_REST}")
    print(f"{len(tasks)} runs; input = {'x(t) only' if X_ONLY_INPUT else 'x(t) and y(t) alternating'}")
    print(f"omega*dt at mean params: {np.sqrt(2 * P['k_wall_val'] / P['m_val']) * 0.01:.2f} (dt=0.01); unstable above 2")
    with Pool() as pool:
        res = dict(pool.imap(trial, tasks))
    for name in ALPHAS:
        print(f"\nRidge {name}.  R^2: pooled (as in notebook) | x(t+1) only | y(t+1) only")
        for d in DTS:
            print(f"  dt = {d[0]:<6} ({d[1]:>2} substeps/Henon step)")
            for s in SEEDS:
                r = res[(d, s)]
                txt = "BLEW UP / masses overlapped" if r is None else \
                      f"{r[name][0]:.6f} | {r[name][1]:.6f} | {r[name][2]:.6f}"
                print(f"     seed {s:>3}{' (notebook recheck)' if s == 100 else '':19}: {txt}")
    print("\nReading it: if x(t+1) R^2 stays near 1 as dt shrinks and across seeds, the result is real.")
    print("If it drops or changes a lot with dt, the dt = 0.01 number was partly a numerical artifact.")

"""
Dimensionless design map: where does symmetry breaking make the chain compute?

Units: L0 = mass-to-wall distance (1), m = mass (1), k_w = mean wall-spring stiffness (1),
so the natural time is tau = sqrt(m / k_w) = 1. Cells are spaced dx = 2.5 L0 (as before).

Dimensionless groups
  eps    wall-spring compression, r_w = (1 + eps) L0      (eps > 0: bistable cells)
  kappa  between-spring stiffness / wall stiffness
  zeta   damping ratio, c = 2 zeta sqrt(m k_w)
  f      load amplitude / (k_w L0)
  Omega  time between loads / tau
  s_k    lognormal spread of all stiffnesses             (keeps mirror symmetry)
  s_rw   lognormal spread of wall rest lengths            (breaks it)
  s_rb   lognormal spread of between rest lengths, i.e. random prestress (breaks it)
Between springs are unstressed on average (rest length = dx), as intended.
Task: x(t) of the Henon map applied as a force (held for the first third of each interval) on
10 masses; ridge readout (cross-validated alpha, standardized features) of all mass positions
and velocities predicts x(t+1). Score: test R^2 for x(t+1).

Run:
  python design_map.py --validate   # reproduce the old results from their dimensionless values (~2 min)
  python design_map.py --quick      # time 4 runs before launching the grid
  python design_map.py --grid       # the full map, ~960 runs; resumable (rerun to continue)
  python design_map.py --summary    # tables from results/design_map.csv
  python design_map.py --example    # true vs predicted attractor for structures A and E at one design
       options: --example A C E --design 1,0.5,2   (structures; kappa,f,Omega)
  python design_map.py --signflip   # is the response odd? compare states for load u and -u
  python design_map.py --parity     # symmetric random input: reconstruct odd vs even functions of it
       both take --design kappa,f,Omega (default 1,0.5,2)
"""
import os
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import csv
import itertools
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from dysts.maps import Henon
from numba import njit
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import spring_henon as sh  # geometry only: 100 cells, wall nodes at y = 0, 2, masses at y = 1, dx = 2.5

spring_forces = sh.spring_forces
N, DIMS, DX = sh.N, sh.dims, sh.dx
NUM_NODES = len(sh.nodes_pos)
DT_MAX = 0.1          # largest time step in units of tau (old runs converged by ~0.2)
here = Path(__file__).parent
OUT = here / "results" / "design_map.csv"

# ---------------- grid ----------------
ZETA = 0.5
KAPPAS = [0.1, 0.2, 0.5, 1.0, 2.0]
FS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
OMEGAS = [0.5, 1.0, 2.0, 4.0]
INPUT_CELLS = np.arange(1, 11) * 9   # cells 9, 18, ..., 90
STRUCTS = {
    "A uniform, uncompressed (symmetric)":             dict(eps=0.0, s_k=0.0, s_rw=0.0, s_rb=0.0,  seeds=[0]),
    "B uncompressed + stiffness disorder (symmetric)": dict(eps=0.0, s_k=0.5, s_rw=0.0, s_rb=0.0,  seeds=[0, 1]),
    "C uncompressed + prestress disorder (breaks)":    dict(eps=0.0, s_k=0.0, s_rw=0.0, s_rb=0.15, seeds=[0, 1]),
    "D uniform, bistable (spontaneous breaking)":      dict(eps=0.1, s_k=0.0, s_rw=0.0, s_rb=0.0,  seeds=[0]),
    "E bistable + stiffness disorder":                 dict(eps=0.1, s_k=0.5, s_rw=0.0, s_rb=0.0,  seeds=[0, 1]),
    "F bistable + prestress disorder":                 dict(eps=0.1, s_k=0.0, s_rw=0.0, s_rb=0.15, seeds=[0, 1]),
}

# ---------------- data ----------------
STEPS, TAU_STEPS, TR, TEST = 20_000, 1, 2_000, 4_000
henon = Henon().make_trajectory(STEPS + 2 * TR + TAU_STEPS)[TR:]
HS_SCALER = StandardScaler().fit(henon[:-TEST])
hs = HS_SCALER.transform(henon)


@njit(cache=True)
def run_sampled(n_h, free, hold, dt, m_inv, c, F_in, in_dofs, pos, conn, k, rest, fixed, rec_dofs):
    # velocity Verlet (same scheme as spring_henon.run_simulation); stores only the recorded dofs
    # at the last substep of each input interval. F_in[h, j] acts on in_dofs[j] for the first
    # `hold` substeps of interval h.
    n, d = pos.shape
    ndof = n * d
    mask = np.ones(ndof)
    for node in fixed:
        mask[node * d : node * d + d] = 0
    x = np.zeros(ndof); v = np.zeros(ndof)
    Xd = np.zeros((n_h, len(rec_dofs))); Xv = np.zeros((n_h, len(rec_dofs)))
    U_prev = np.zeros(ndof); U_now = np.zeros(ndof)
    for j in range(len(in_dofs)):
        U_prev[in_dofs[j]] = F_in[0, j]
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


def lognormal(rng, mean, s, n):
    return np.full(n, mean) if s == 0 else rng.lognormal(np.log(mean) - s**2 / 2, s, n)


def structure(eps, kappa, zeta, s_k, s_rw, s_rb, seed):
    rng = np.random.default_rng(seed)
    k_w = lognormal(rng, 1.0, s_k, N)                 # shared by a cell's two wall springs
    k_b = lognormal(rng, kappa, s_k, N - 1)
    r_w = (1.0 + eps) * lognormal(rng, 1.0, s_rw, N)
    r_b = DX * lognormal(rng, 1.0, s_rb, N - 1)       # unstressed on average
    k = np.concatenate([k_w, k_w, k_b])               # sh.conn order: bottom-mid, mid-top, between
    rest = np.concatenate([r_w, r_w, r_b])
    m_inv = np.ones(NUM_NODES * DIMS)
    c = np.full(NUM_NODES * DIMS, 2.0 * zeta)
    return m_inv, c, k, rest


def simulate(p, return_pred=False):
    m_inv, c, k, rest = structure(p["eps"], p["kappa"], p["zeta"], p["s_k"], p["s_rw"], p["s_rb"], p["seed"])
    n_sub = p.get("n_sub") or max(3, int(np.ceil(p["Omega"] / DT_MAX)))
    hold = p.get("hold") or max(1, round(n_sub / 3))
    dt = p["Omega"] / n_sub
    cells = np.asarray(p.get("cells", INPUT_CELLS))
    comps = [0 if p.get("x_only", True) else i % 2 for i in range(len(cells))]
    F_in = np.ascontiguousarray(hs[:, comps] * p["f"])
    in_dofs = (sh.mid[cells] * DIMS).astype(np.int64)
    rec = (sh.mid * DIMS).astype(np.int64)
    Xd, Xv = run_sampled(len(hs), n_sub, hold, dt, m_inv, c, F_in, in_dofs,
                         sh.nodes_pos, sh.conn, k, rest, sh.fixed, rec)
    X = np.column_stack([Xd, Xv])
    pos = sh.nodes_pos[sh.mid, 0] + Xd
    if not np.isfinite(X).all() or (np.diff(pos, axis=1) <= 0.01).any():
        return np.nan                                  # blew up or masses passed through each other
    Xs = X[:-TAU_STEPS][TR:]
    Y = hs[TR + TAU_STEPS :, 0]
    Xtr, Xte, Ytr, Yte = Xs[:-TEST], Xs[-TEST:], Y[:-TEST], Y[-TEST:]
    sc = StandardScaler().fit(Xtr)
    model = RidgeCV(alphas=np.logspace(-6, 4, 21)).fit(sc.transform(Xtr), Ytr)
    pred = model.predict(sc.transform(Xte))
    if return_pred:
        return Yte, pred
    return r2_score(Yte, pred)


def _job(p):
    t = time.time()
    return p, simulate(p), time.time() - t


# ---------------- validation: old results from their dimensionless values ----------------
def validate():
    # original sweep (spring_henon.py, intended unstressed between springs): k_w=5, k_b=4, m=0.01, c=0.2,
    # F0=10, 0.03 per Henon step, 3 substeps, force on one substep, x and y inputs at cells 0,10,...,90
    tau_s = np.sqrt(0.01 / 5)
    sweep = dict(eps=0.1, kappa=4 / 5, zeta=0.2 / (2 * np.sqrt(0.01 * 5)), f=10 / 5, Omega=0.03 / tau_s,
                 n_sub=3, hold=1, cells=np.arange(10) * 10, x_only=False, s_k=0, s_rw=0, s_rb=0, seed=0)
    # Bayesian-optimization averages (check_bo_winner.py, dt = 0.002, 15 substeps, force held 5)
    m, kw = 0.0062590214410398, 65.2663180731547
    tau_o = np.sqrt(m / kw)
    opt = dict(kappa=13.54246039410458 / kw, zeta=0.8604972776791918 / (2 * np.sqrt(m * kw)),
               f=14.007215766574747 / kw, Omega=0.03 / tau_o, n_sub=15, hold=5, cells=INPUT_CELLS,
               x_only=True, s_k=0, s_rw=0, s_rb=0, seed=0)
    cases = [("original sweep, uniform bistable (old R^2 0.29)", sweep, 0.293),
             ("optimized averages, uniform bistable (old 0.964)", {**opt, "eps": 0.1}, 0.964),
             ("optimized averages, uniform uncompressed (old 0.275)", {**opt, "eps": 0.0}, 0.275)]
    for name, p, _ in cases:
        print(f"{name}: kappa={p['kappa']:.3f} zeta={p['zeta']:.3f} f={p['f']:.3f} Omega={p['Omega']:.3f} eps={p['eps']}")
    with Pool() as pool:
        res = pool.map(_job, [p for _, p, _ in cases])
    print()
    for (name, _, old), (_, r2, sec) in zip(cases, res):
        print(f"  {name:55s} new R^2 = {r2:.4f}   ({sec:.0f} s)")
    print("\nNew values within a few hundredths of the old ones mean the dimensionless model is right")
    print("(small differences come from the cross-validated readout and Henon trajectory).")


# ---------------- grid ----------------
FIELDS = ["struct", "seed", "kappa", "f", "Omega", "r2", "seconds"]


def grid_tasks(quick=False):
    pts = list(itertools.product(KAPPAS, FS, OMEGAS))
    if quick:
        pts = [(0.2, 0.2, 1.0), (1.0, 1.0, 4.0)]
    tasks = []
    for name, s in STRUCTS.items():
        if quick and name[0] not in "AD":
            continue
        for seed in s["seeds"]:
            for kappa, f, Om in pts:
                tasks.append(dict(struct=name, eps=s["eps"], s_k=s["s_k"], s_rw=s["s_rw"], s_rb=s["s_rb"],
                                  seed=seed, kappa=kappa, f=f, Omega=Om, zeta=ZETA))
    return tasks


def key(d):
    return (d["struct"], int(d["seed"]), float(d["kappa"]), float(d["f"]), float(d["Omega"]))


def run_grid(quick=False):
    out = here / "results" / ("design_map_quick.csv" if quick else "design_map.csv")
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {key(r) for r in csv.DictReader(fh)}
    tasks = [t for t in grid_tasks(quick) if key(t) not in done]
    print(f"{len(tasks)} runs to do ({len(done)} already in {out.name})")
    new = not out.exists()
    t0 = time.time()
    with open(out, "a", newline="") as fh, Pool() as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, (p, r2, sec) in enumerate(pool.imap_unordered(_job, tasks)):
            w.writerow({k: p[k] for k in FIELDS[:5]} | {"r2": r2, "seconds": round(sec, 1)})
            fh.flush()
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                el = time.time() - t0
                print(f"  {i + 1}/{len(tasks)} done, {el / 60:.1f} min elapsed, "
                      f"~{el / (i + 1) * (len(tasks) - i - 1) / 60:.0f} min left", flush=True)
    summary(out)


def summary(path=OUT):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    print(f"\nSummary of {path.name} ({len(rows)} runs). R^2 for x(t+1), seeds averaged per design point.")
    print(f"{'structure':50s} {'points':>6} {'failed':>6} {'median':>7} {'R2>0.9':>7} {'R2>0.5':>7}  best (kappa, f, Omega)")
    for name in STRUCTS:
        pts = {}
        for r in rows:
            if r["struct"] == name:
                pts.setdefault((float(r["kappa"]), float(r["f"]), float(r["Omega"])), []).append(float(r["r2"]))
        if not pts:
            continue
        vals = {p: (np.nan if np.isnan(v).any() else np.mean(v)) for p, v in pts.items()}
        arr = np.array(list(vals.values()))
        ok = arr[np.isfinite(arr)]
        failed = np.sum(~np.isfinite(arr))
        best = max((p for p in vals if np.isfinite(vals[p])), key=lambda p: vals[p], default=None)
        # failed points count as not reaching the threshold
        print(f"{name:50s} {len(arr):6d} {failed:6d} {np.median(ok) if len(ok) else np.nan:7.3f} "
              f"{np.sum(ok > 0.9) / len(arr):7.0%} {np.sum(ok > 0.5) / len(arr):7.0%}  "
              f"{vals[best]:.4f} at {best}" if best else "")
    print("\nmedian = typical design;  R2>0.9 = fraction of the design space that works well;  best = upper bound")


# ---------------- example: true vs predicted attractor ----------------
def _example_job(args):
    name, p = args
    return name, simulate(p, return_pred=True)


def run_example():
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    from scipy.stats import binned_statistic

    i = sys.argv.index("--example")
    letters = [a for a in sys.argv[i + 1:] if len(a) == 1 and a in "ABCDEF"] or ["A", "E"]
    design = (1.0, 0.5, 2.0)
    if "--design" in sys.argv:
        design = tuple(float(x) for x in sys.argv[sys.argv.index("--design") + 1].split(","))
    kappa, f, Om = design
    by_letter = {name[0]: (name, s) for name, s in STRUCTS.items()}
    jobs = [(by_letter[L][0], dict(eps=by_letter[L][1]["eps"], s_k=by_letter[L][1]["s_k"],
                                   s_rw=by_letter[L][1]["s_rw"], s_rb=by_letter[L][1]["s_rb"],
                                   seed=0, kappa=kappa, f=f, Omega=Om, zeta=ZETA)) for L in letters]
    with Pool() as pool:
        res = dict(pool.map(_example_job, jobs))

    unscale = lambda z: z * HS_SCALER.scale_[0] + HS_SCALER.mean_[0]   # back to Henon units
    (here / "results").mkdir(exist_ok=True); (here / "plots").mkdir(exist_ok=True)
    fig, axes = plt.subplots(
        len(letters), 2, figsize=(6, 3 * len(letters)), dpi=144,
        squeeze=False, sharex="col"
    )
    for row, (name, _) in zip(axes, jobs):
        true, pred = (unscale(v) for v in res[name])
        r2 = r2_score(true, pred)
        np.savez(here / "results" / f"example_{name[0]}.npz", true=true, pred=pred, r2=r2,
                 kappa=kappa, f=f, Omega=Om, zeta=ZETA)
        # delay map: true attractor (black) and prediction (blue), each rasterized as its own image
        row[0].scatter(true[:-1], true[1:], s=2, c="k", rasterized=True, zorder=1)
        row[0].scatter(pred[:-1], pred[1:], s=2, c="SteelBlue", alpha=0.5, rasterized=True, zorder=3)
        row[0].set(xlabel="x(t)", ylabel="x(t+1)")
        row[0].set_title(f"{name}", fontsize=7, loc="left")
        # predicted vs true, with binned mean and spread of the truth
        lo, hi = -1.5, 1.5
        row[1].scatter(pred, true, s=5, alpha=0.1, c="k", rasterized=True)
        row[1].axline([0, 0], slope=1, ls="--", c="k", zorder=-1)
        stat = lambda st: binned_statistic(pred, true, statistic=st, bins=20, range=(lo, hi))
        mean, edges, _ = stat("mean"); std = stat("std")[0]
        row[1].errorbar(0.5 * (edges[:-1] + edges[1:]), mean, yerr=std, fmt="o", ms=4, mec="w",
                        c="FireBrick", elinewidth=1.5, capsize=2, zorder=4)
        row[1].set(xlabel="predicted x(t+1)", ylabel="true x(t+1)")
        row[1].text(0.05, 0.95, f"R$^2$ = {r2:.3f}", transform=row[1].transAxes, ha="left", va="top")
        for ax in row:
            ax.set(xlim=[lo, hi], ylim=[lo, hi], xticks=[-1, 0, 1], yticks=[-1, 0, 1]); ax.set_aspect(1)
        print(f"{name:50s} R^2 = {r2:.4f}")
    fig.suptitle(rf"$\kappa$ = {kappa:g}, $f$ = {f:g}, $\Omega$ = {Om:g}, $\zeta$ = {ZETA:g}", fontsize=8)
    fig.tight_layout()
    out = here / "plots" / f"example_k{kappa:g}_f{f:g}_O{Om:g}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved -> {out}  (arrays in results/example_<letter>.npz)")


# ---------------- symmetry tests ----------------
def _design():
    if "--design" in sys.argv:
        return tuple(float(x) for x in sys.argv[sys.argv.index("--design") + 1].split(","))
    return (1.0, 0.5, 2.0)


def _params(letter, seed=0):
    name, st = next((n, v) for n, v in STRUCTS.items() if n[0] == letter)
    kappa, f, Om = _design()
    return name, dict(eps=st["eps"], s_k=st["s_k"], s_rw=st["s_rw"], s_rb=st["s_rb"], seed=seed,
                      kappa=kappa, f=f, Omega=Om, zeta=ZETA)


def run_states(p, U):
    """States (positions and velocities of all masses, one row per input interval) for the input
    U[h, j] (dimensionless load on input mass j during interval h, before scaling by f).
    Returns None if the structure blew up."""
    m_inv, c, k, rest = structure(p["eps"], p["kappa"], p["zeta"], p["s_k"], p["s_rw"], p["s_rb"], p["seed"])
    n_sub = max(3, int(np.ceil(p["Omega"] / DT_MAX)))
    hold = max(1, round(n_sub / 3))
    F_in = np.ascontiguousarray(U * p["f"])
    in_dofs = (sh.mid[INPUT_CELLS] * DIMS).astype(np.int64)
    rec = (sh.mid * DIMS).astype(np.int64)
    Xd, Xv = run_sampled(len(U), n_sub, hold, p["Omega"] / n_sub, m_inv, c, F_in, in_dofs,
                         sh.nodes_pos, sh.conn, k, rest, sh.fixed, rec)
    X = np.column_stack([Xd, Xv])
    return X if np.isfinite(X).all() else None


def _signflip_job(letter):
    name, p = _params(letter)
    U = np.repeat(hs[:, :1], len(INPUT_CELLS), axis=1)        # Henon x(t) at every input mass
    Xp, Xm = run_states(p, U), run_states(p, -U)
    if Xp is None or Xm is None:
        return name, np.nan
    return name, np.abs(Xp + Xm).max() / np.abs(Xp).max()


def run_signflip():
    print(f"Sign-flip test at (kappa, f, Omega) = {_design()}: drive with u and with -u.")
    print("If the dynamics are odd, the states are exact negatives: X[u] + X[-u] = 0.\n")
    with Pool() as pool:
        res = pool.map(_signflip_job, "ABCDEF")
    for name, r in res:
        print(f"  {name:50s} max|X[u] + X[-u]| / max|X[u]| = {r:.2e}")
    print("\nExpected: ~1e-12 (rounding) for A, B (symmetric) and also for D, E: their equations are odd too;")
    print("bistable chains break the symmetry only through which wells the early loads select.")
    print("C and F (rest-length disorder = constant forces) should be O(1).")


PARITY_TARGETS = {   # functions of the symmetric input u; t = the interval whose load was just applied
    "odd":  {"u(t-1)": lambda u, t: u[t - 1], "u(t-2)": lambda u, t: u[t - 2],
             "u(t)^3": lambda u, t: u[t] ** 3, "u(t)u(t-1)u(t-2)": lambda u, t: u[t] * u[t - 1] * u[t - 2]},
    "even": {"u(t)^2": lambda u, t: u[t] ** 2, "u(t)u(t-1)": lambda u, t: u[t] * u[t - 1],
             "u(t-1)^2": lambda u, t: u[t - 1] ** 2, "u(t)u(t-2)": lambda u, t: u[t] * u[t - 2]},
}


def _parity_job(letter):
    name, p = _params(letter)
    rng = np.random.default_rng(12345)
    u = rng.uniform(-np.sqrt(3), np.sqrt(3), len(hs))         # symmetric, unit variance, bounded
    X = run_states(p, np.repeat(u[:, None], len(INPUT_CELLS), axis=1))
    if X is None:
        return name, None
    t = np.arange(TR, len(u))                                 # skip the washout
    Xs = X[t]
    out = {}
    for parity, targets in PARITY_TARGETS.items():
        for tname, fn in targets.items():
            Y = fn(u, t)
            Xtr, Xte, Ytr, Yte = Xs[:-TEST], Xs[-TEST:], Y[:-TEST], Y[-TEST:]
            sc = StandardScaler().fit(Xtr)
            model = RidgeCV(alphas=np.logspace(-6, 4, 21)).fit(sc.transform(Xtr), Ytr)
            out[(parity, tname)] = r2_score(Yte, model.predict(sc.transform(Xte)))
    return name, out


def run_parity():
    print(f"Parity test at (kappa, f, Omega) = {_design()}: symmetric random input u (uniform, unit variance).")
    print("Readout reconstructs odd vs even functions of the recent input; test R^2 shown.\n")
    with Pool() as pool:
        res = pool.map(_parity_job, "ABCDEF")
    cols = [(par, t) for par in PARITY_TARGETS for t in PARITY_TARGETS[par]]
    print(f"  {'structure':44s}" + "".join(f"{('[' + par[0] + '] ' + t):>20s}" for par, t in cols))
    for name, out in res:
        row = "".join(f"{out[c]:20.3f}" for c in cols) if out else "   blew up"
        print(f"  {name:44s}{row}")
    out_csv = here / "results" / "parity.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["struct", "parity", "target", "r2", "kappa", "f", "Omega"])
        for name, out in res:
            for (par, t) in cols:
                w.writerow([name, par, t, out[(par, t)] if out else np.nan, *_design()])
    print(f"\nSaved -> {out_csv}")
    print("\n[o] odd targets, [e] even targets. Prediction: symmetric structures (A, B) reach ~0 on every even")
    print("target (exactly 0 for infinite data); structures with broken symmetry (C-F) reconstruct both.")
    print("D and E test whether well selection by early loads breaks the symmetry on the readout's time scale.")


if __name__ == "__main__":
    if "--signflip" in sys.argv:
        run_signflip()
    elif "--parity" in sys.argv:
        run_parity()
    elif "--example" in sys.argv:
        run_example()
    elif "--validate" in sys.argv:
        validate()
    elif "--quick" in sys.argv:
        run_grid(quick=True)
    elif "--grid" in sys.argv:
        run_grid()
    elif "--summary" in sys.argv:
        summary()
    else:
        print(__doc__)

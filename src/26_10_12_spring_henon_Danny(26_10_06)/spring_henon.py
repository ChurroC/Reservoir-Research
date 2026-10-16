import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from dysts.maps import Henon
from sklearn.linear_model import RidgeCV
from scipy.stats import binned_statistic
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ---------- parameters ----------
N = 100  # bistable cells in the chain
dx = 2.5  # spacing between cells
dims = 2
k_val, k_between_val = 5.0, 4.0  # mean spring constants (in-cell, between cells)
rest_len = 1.1
m_val, c_val = 0.01, 0.2
dt, free_steps, input_force = 0.01, 3, 10.0
target_cells = np.arange(10) * (N // 10)  # cells that receive the Henon forcing

n_steps, washout, test_frac = 20000, 2000, 0.2
sigmas = [0, 0.5, 1]  # LogNormal std of the log spring constants (0 = uniform)


# ---------- spring simulation ----------
@njit(cache=True)
def spring_forces(conn, disp, pos, rest, k, n, d):
    f = np.zeros((n, d))
    disp = disp.reshape(n, d)
    for s in range(len(conn)):
        a, b = conn[s, 0], conn[s, 1]
        delta = (pos[b] + disp[b]) - (pos[a] + disp[a])
        dist = np.sqrt(np.sum(delta**2))
        fs = k[s] * (dist - rest[s]) * delta / dist
        f[a] += fs
        f[b] -= fs
    return f.reshape(-1)


@njit(cache=True)
def run_simulation(steps, dt, m_inv, c, U, pos, conn, k, rest, fixed):
    n, d = pos.shape
    disp = np.zeros((steps, n * d))
    v = np.zeros((steps, n * d))
    mask = np.ones(n * d)
    for node in fixed:
        mask[node * d : node * d + d] = 0

    F = spring_forces(conn, disp[0], pos, rest, k, n, d)
    for i in range(1, steps):
        acc = m_inv * (F - c * v[i - 1] + U[i - 1]) * mask
        disp[i] = disp[i - 1] + v[i - 1] * dt + 0.5 * acc * dt**2
        F = spring_forces(conn, disp[i], pos, rest, k, n, d)
        acc_next = m_inv * (F - c * (v[i - 1] + 0.5 * acc * dt) + U[i]) * mask
        v[i] = v[i - 1] + 0.5 * (acc + acc_next) * dt
    return disp, v


# ---------- chain geometry ----------
# each cell = bottom wall node, free middle node, top wall node
cell = 3 * np.arange(N)
nodes_pos = np.column_stack([np.repeat(dx * np.arange(N), 3), np.tile([0.0, 1.0, 2.0], N)])
mid = cell + 1
fixed = np.sort(np.concatenate([cell, cell + 2]))

in_cell = np.column_stack([np.concatenate([cell, mid]), np.concatenate([mid, cell + 2])])
between = np.column_stack([mid[:-1], mid[1:]])
conn = np.vstack([in_cell, between])
n_in = len(in_cell)

num_nodes = len(nodes_pos)
m_inv = np.full(num_nodes * dims, 1 / m_val)
c_diag = np.full(num_nodes * dims, c_val)
rest = np.full(len(conn), rest_len)


def lognormal(mean, sigma, size, rng):
    # LogNormal with the given mean and log-variance sigma^2 (sigma = 0 gives the mean)
    return rng.lognormal(np.log(mean) - sigma**2 / 2, sigma, size)


def spring_constants(sigma, seed=42):
    rng = np.random.default_rng(seed)
    # one value per cell, shared by its two springs to keep the y-symmetry
    k_cell = lognormal(k_val, sigma, N, rng)
    k_between = lognormal(k_between_val, sigma, len(between), rng)
    return np.concatenate([k_cell, k_cell, k_between])


# ---------- Henon data ----------
henon = Henon(a=1.4, b=0.3).make_trajectory(n_steps + 1 + 1000)[1000:]
n_test = int(n_steps * test_frac)
scaler = StandardScaler().fit(henon[: -n_test - 1])
henon_scaled = scaler.transform(henon)
u = henon_scaled[:-1]  # input at time t
y = henon_scaled[1:]  # target: Henon state at t+1


# ---------- reservoir run ----------
def run_reservoir(k):
    # force the x-direction of the target middle nodes with the Henon signal
    U = np.zeros((n_steps * free_steps, num_nodes * dims))
    for i, c in enumerate(target_cells):
        U[::free_steps, mid[c] * dims] = u[:, i % dims] * input_force

    disp, vel = run_simulation(len(U), dt, m_inv, c_diag, U, nodes_pos, conn, k, rest, fixed)

    # features: x position and velocity of every middle node, sampled once per Henon step
    idx = mid * dims
    X = np.column_stack([disp[:, idx], vel[:, idx]])[free_steps - 1 :: free_steps]
    return X[washout:]


def readout(X, Y):
    # ridge regression fitted on the train part, returns (true, pred) on the test part
    model = RidgeCV(alphas=np.logspace(-5, 5, 30), cv=3).fit(X[:-n_test], Y[:-n_test])
    return Y[-n_test:], model.predict(X[-n_test:])


# ---------- plots ----------
def plot_results(true, pred, title):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=144)
    fig.suptitle(title)

    # Henon delay map: x(t) -> x(t+1)
    axes[0].scatter(
        true[:-1, 0], true[1:, 0],
        s=2, c="k",
        label="true",
        rasterized=True
    )
    axes[0].scatter(
        pred[:-1, 0], pred[1:, 0],
        s=2, c="SteelBlue", alpha=0.5,
        label="pred",
        rasterized=True,
        zorder=3,  # draw after the other artists so it is rasterized as its own image, separate from the black dots
    )
    axes[0].set(
        xlabel="x(t)", ylabel="x(t+1)",
    )
    axes[0].text(
        0.49, 1.01, "true", color="k",
        transform=axes[0].transAxes,
        ha="right", va="bottom"
    )
    axes[0].text(
        0.51, 1.01, "pred", color="SteelBlue",
        transform=axes[0].transAxes,
        ha="left", va="bottom"
    )
    # axes[0].legend(loc="outer upper middle", ncols=2)

    # predicted vs actual x(t+1)
    lo, hi = min(true[:, 0].min(), pred[:, 0].min()), max(true[:, 0].max(), pred[:, 0].max())
    axes[1].scatter(
        pred[:, 0], true[:, 0],
        s=5, alpha=0.1, c="k",
        rasterized=True
    )
    axes[1].axline([0, 0], slope=1, ls="--", c="k", zorder=-1)

    # mean and std of the true values within bins of the prediction
    stat = lambda name: binned_statistic(
        pred[:, 0], true[:, 0], statistic=name, bins=20, range=(lo, hi)
    )
    mean, edges, _ = stat("mean")
    std = stat("std")[0]
    axes[1].errorbar(
        0.5 * (edges[:-1] + edges[1:]), mean, yerr=std,
        fmt="o", ms=5, mec="w", c="FireBrick",
        elinewidth=2, capsize=2
    )
    axes[1].set(
        xlabel="predicted x(t+1)",
        ylabel="true x(t+1)"
    )
    axes[1].text(
        0.05, 0.95,
        f"R²= {r2_score(true[:, 0], pred[:, 0]):.4f}",
        transform=axes[1].transAxes, ha="left", va="top"
    )

    # axes[0].set()
    for ax in axes.ravel():
        ax.set(
            xlim=[-1.5, 1.5],
            ylim=[-1.5, 1.5],
            xticks=[-1, 0, 1],
            yticks=[-1, 0, 1]
        )
        ax.set_aspect(1)

    fig.tight_layout()
    return fig


# ---------- run baseline and each sigma ----------
y_next = y[washout:]  # target: Henon state at t+1
lag0 = u[washout:, :1]  # x of the last input the reservoir saw
lag1 = u[washout - 1 : -1, :1]  # x of the input before that


def r2(true, pred):
    return r2_score(true, pred, multioutput="raw_values")


if __name__ == "__main__":
    # baseline: no reservoir, regress directly on the current Henon state [x(t), y(t)]
    true, pred = readout(u[washout:], y_next)
    print(f"No reservoir: next-step R² x = {r2(true, pred)[0]:.4f}, y = {r2(true, pred)[1]:.4f}")

    for sigma in sigmas:
        X = run_reservoir(spring_constants(sigma))
        true, pred = readout(X, y_next)
        label = f"σ = {sigma}" + (" (uniform)" if sigma == 0 else "")
        print(
            f"{label}: next-step R² x = {r2(true, pred)[0]:.4f}, y = {r2(true, pred)[1]:.4f}"
            f" | recall R² last input = {r2(*readout(X, lag0))[0]:.4f},"
            f" one before = {r2(*readout(X, lag1))[0]:.4f}"
        )
        fig = plot_results(scaler.inverse_transform(true), scaler.inverse_transform(pred), label)
        fig.savefig(f"plots/henon_prediction_sigma{sigma}.pdf", bbox_inches="tight")

    plt.show()

import numpy as np
from numba import njit
from sklearn.linear_model import Ridge
import optuna
import time
import os
import pandas as pd

slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
trial_result_path = os.path.join(slurm_job_id, "trials_results.csv")
best_results_path = os.path.join(slurm_job_id, "best_results.csv")
config_path = os.path.join(slurm_job_id, "config.json")

@njit(cache=True)
def get_spring_forces(connections_list, disp, initial_pos, rest_lens, k_vals, num_nodes, dims):
    forces = np.zeros((num_nodes, dims))
    disp_reshaped = disp.reshape(num_nodes, dims)

    for i in range(len(connections_list)):
        idx_a = connections_list[i, 0]
        idx_b = connections_list[i, 1]

        delta = np.zeros(dims)
        dist_sq = 0.0
        for j in range(dims):
            pos_a = initial_pos[idx_a, j] + disp_reshaped[idx_a, j]
            pos_b = initial_pos[idx_b, j] + disp_reshaped[idx_b, j]
            delta[j] = pos_b - pos_a
            dist_sq += delta[j] ** 2

        dist = np.sqrt(dist_sq)

        mag = k_vals[i] * (dist - rest_lens[i])

        for j in range(dims):
            f_component = mag * (delta[j] / dist)
            forces[idx_a, j] += f_component
            forces[idx_b, j] -= f_component

    return forces.reshape(-1)

@njit(cache=True)
def run_simulation(
    steps, dt, m_inv_diag, c_diag, U, initial_pos, connections_list, k_vals, rest_lens, wall_nodes=[-1]
):
    num_nodes = initial_pos.shape[0]
    dims = initial_pos.shape[1]
    matrix_size = num_nodes * dims

    disp = np.zeros((steps, matrix_size))
    v = np.zeros((steps, matrix_size))
    acc = np.zeros(matrix_size)

    mask = np.ones(matrix_size)
    if wall_nodes[0] != -1:
        for wall in wall_nodes:
            idx = wall * dims
            mask[idx : idx + dims] = 0

    F_spring = get_spring_forces(
        connections_list, disp[0], initial_pos, rest_lens, k_vals, num_nodes, dims
    )

    for i in range(1, steps):
        acc = m_inv_diag * (F_spring - c_diag * v[i - 1] + U[i - 1])
        acc *= mask

        disp[i] = disp[i - 1] + v[i - 1] * dt + acc * 0.5 * dt**2
        
        F_spring = get_spring_forces(
            connections_list, disp[i], initial_pos, rest_lens, k_vals, num_nodes, dims
        )

        acc_next = m_inv_diag * (F_spring - c_diag * (v[i - 1] + .5 * acc * dt) + U[i])
        acc_next *= mask

        v[i] = v[i - 1] + 0.5 * (acc + acc_next) * dt

    return disp, v



N = 30
dist_between = 2.5
dist_y = 1.0
x = np.zeros(N * 3)
for i in range(0, N):
    x[i * 3 : (i + 1) * 3] = np.array([0, 0, 0]) + dist_between * i
y = np.tile(np.array([0, 1 * dist_y, 2 * dist_y]), N)
nodes_pos = np.column_stack((x, y))

dt = 0.01
target_node_count = 10
mu = lambda x, sigma: np.log(x) - (sigma**2 / 2)

k_max = int(2 * 100)
transient = 1000
train_steps = 5000
test_steps = 3000
start_idx = max(transient, k_max)
total_steps = start_idx + train_steps + test_steps

N_step = int(N / target_node_count)
target_nodes = 1 + (np.arange(target_node_count) * N_step) * 3
starts = 3 * np.arange(N)
wall_nodes = np.column_stack([starts, starts + 2]).flatten()
num_nodes = nodes_pos.shape[0]
dims = nodes_pos.shape[1]
matrix_size = num_nodes * dims

node_ids = np.arange(x.size)
starts = 3 * np.arange(N)
connection_src = np.column_stack([starts, starts + 1]).flatten()
connection_dst = np.column_stack([starts + 1, starts + 2]).flatten()
betweens = np.arange(1, node_ids[-1], 3)
between_src = betweens[:-1]
between_dst = betweens[1:]
src_nodes = np.concatenate([connection_src, between_src])
dst_nodes = np.concatenate([connection_dst, between_dst])
connections_list = np.column_stack((src_nodes, dst_nodes))

def evaluate_capacity(X_train, X_test, Y_train, Y_test, num_lags, ridge_alpha):
    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    c_k = np.zeros(num_lags)
    for k in range(num_lags):
        y_true = Y_test[:, k]
        y_hat = Y_pred[:, k]

        cov_matrix = np.cov(y_true, y_hat)
        cov = cov_matrix[0, 1]
        var_true = cov_matrix[0, 0]
        var_pred = cov_matrix[1, 1]

        denom = var_true * var_pred
        c_k[k] = (cov**2) / denom if denom > 1e-12 else 0.0

    return c_k

def spring_trial(
    rng_seed,
    input_force,
    m_val,
    m_spread,
    c_val,
    c_spread,
    k_wall_val,
    k_wall_spread,
    k_between_val,
    k_between_spread,
    wall_rest_val,
    wall_rest_spread,
    between_rest_val,
    between_rest_spread,
):
    rng = np.random.default_rng(rng_seed)

    u = rng.uniform(-1.0, 1.0, size=(total_steps, 1))
    U = np.zeros((total_steps, matrix_size))
    for j, node_index in enumerate(target_nodes):
        U[::, node_index * dims] = u.reshape(-1)
    U = U * input_force

    m_nodes = rng.lognormal(mean=mu(m_val, m_spread), sigma=m_spread, size=num_nodes)
    m_diag = np.repeat(m_nodes, dims)
    m_inv_diag = 1.0 / m_diag

    c_nodes = rng.lognormal(mean=mu(c_val, c_spread), sigma=c_spread, size=num_nodes)
    c_diag = np.repeat(c_nodes, dims)

    k_connection_vals = rng.lognormal(
        mean=mu(k_wall_val, k_wall_spread), sigma=k_wall_spread, size=connection_src.shape[0] // 2
    ).repeat(2)
    k_between_vals = rng.lognormal(
        mean=mu(k_between_val, k_between_spread), sigma=k_between_spread, size=between_src.shape[0]
    )
    k_vals = np.concatenate([k_connection_vals, k_between_vals])

    rest_connection_lens = rng.lognormal(
        mean=mu(wall_rest_val, wall_rest_spread),
        sigma=wall_rest_spread,
        size=connection_src.shape[0] // 2,
    ).repeat(2)
    rest_between_lens = rng.lognormal(
        mean=mu(between_rest_val, between_rest_spread), sigma=between_rest_spread, size=between_src.shape[0]
    )
    rest_lens = np.concatenate([rest_connection_lens, rest_between_lens])

    displacement, velocity = run_simulation(
        steps=total_steps,
        dt=dt,
        m_inv_diag=m_inv_diag,
        c_diag=c_diag,
        U=U,
        initial_pos=nodes_pos,
        connections_list=connections_list,
        k_vals=k_vals,
        rest_lens=rest_lens,
        wall_nodes=wall_nodes,
    )

    movement_nodes = 1 + np.arange(N) * 3
    movement_idx = movement_nodes * dims
    X = np.column_stack((displacement[:, movement_idx], velocity[:, movement_idx]))

    return X, u

def memory_trial(
    input, X_states, k_max, total_steps, start_idx, train_steps, test_steps, ridge_alpha
):
    u = input
    X = X_states

    X_clean = X[start_idx:]
    X_train = X_clean[:train_steps]
    X_test = X_clean[train_steps : train_steps + test_steps]

    Y_deg1 = np.zeros((total_steps, k_max))
    for k in range(1, k_max + 1):
        u_delayed = u[:-k, 0]
        Y_deg1[k:, k - 1] = np.sqrt(3.0) * u_delayed
    Y_deg2 = np.zeros((total_steps, k_max))
    for k in range(1, k_max + 1):
        u_delayed = u[:-k, 0]
        Y_deg2[k:, k - 1] = np.sqrt(5.0) * (3.0 * (u_delayed**2) - 1.0) / 2.0

    Y_deg3 = np.zeros((total_steps, k_max))
    for k in range(1, k_max + 1):
        u_delayed = u[:-k, 0]
        Y_deg3[k:, k - 1] = (
            np.sqrt(7.0) * (5.0 * (u_delayed**3) - 3.0 * u_delayed) / 2.0
        )

    # Degree 1
    Y1_clean = Y_deg1[start_idx:]
    Y1_train = Y1_clean[:train_steps]
    Y1_test = Y1_clean[train_steps : train_steps + test_steps]
    c_deg1 = evaluate_capacity(X_train, X_test, Y1_train, Y1_test, k_max, ridge_alpha)
    # Degree 2
    Y2_clean = Y_deg2[start_idx:]
    Y2_train = Y2_clean[:train_steps]
    Y2_test = Y2_clean[train_steps : train_steps + test_steps]
    c_deg2 = evaluate_capacity(X_train, X_test, Y2_train, Y2_test, k_max, ridge_alpha)
    # Degree 3
    Y3_clean = Y_deg3[start_idx:]
    Y3_train = Y3_clean[:train_steps]
    Y3_test = Y3_clean[train_steps : train_steps + test_steps]
    c_deg3 = evaluate_capacity(X_train, X_test, Y3_train, Y3_test, k_max, ridge_alpha)

    k_cross = 25
    cross_pairs = [
        (k1, k2) for k1 in range(1, k_cross + 1) for k2 in range(k1 + 1, k_cross + 1)
    ]
    num_cross_pairs = len(cross_pairs)
    Y_cross2 = np.zeros((total_steps, num_cross_pairs))
    for idx, (k1, k2) in enumerate(cross_pairs):
        max_k = k2
        u_k1 = u[max_k - k1 : len(u) - k1, 0]
        u_k2 = u[max_k - k2 : len(u) - k2, 0]
        Y_cross2[max_k:, idx] = 3.0 * (u_k1 * u_k2)

    # Cross Degree 2
    Y_cross2_clean = Y_cross2[start_idx:]
    c_cross2 = evaluate_capacity(
        X_train,
        X_test,
        Y_cross2_clean[:train_steps],
        Y_cross2_clean[train_steps : train_steps + test_steps],
        num_cross_pairs,
        ridge_alpha,
    )

    k_cross3 = 25
    # Cross Degree 3 (2+1 interaction)
    pairs_21 = [
        (k1, k2)
        for k1 in range(1, k_cross3 + 1)
        for k2 in range(1, k_cross3 + 1)
        if k1 != k2
    ]
    num_21 = len(pairs_21)
    Y_cross3_21 = np.zeros((total_steps, num_21))
    for idx, (k1, k2) in enumerate(pairs_21):
        max_k = max(k1, k2)
        u_k1 = u[max_k - k1 : len(u) - k1, 0]
        u_k2 = u[max_k - k2 : len(u) - k2, 0]

        # sqrt(5) * P2 * sqrt(3) * P1 = sqrt(15) * ((3*u^2 - 1)/2) * u
        p2 = np.sqrt(5.0) * (3.0 * (u_k1**2) - 1.0) / 2.0
        p1 = np.sqrt(3.0) * u_k2
        Y_cross3_21[max_k:, idx] = p2 * p1
    # Cross Degree 3 (1+1+1 interaction)
    triples_111 = [
        (k1, k2, k3)
        for k1 in range(1, k_cross3 + 1)
        for k2 in range(k1 + 1, k_cross3 + 1)
        for k3 in range(k2 + 1, k_cross3 + 1)
    ]
    num_111 = len(triples_111)
    Y_cross3_111 = np.zeros((total_steps, num_111))
    for idx, (k1, k2, k3) in enumerate(triples_111):
        max_k = k3
        u_k1 = u[max_k - k1 : len(u) - k1, 0]
        u_k2 = u[max_k - k2 : len(u) - k2, 0]
        u_k3 = u[max_k - k3 : len(u) - k3, 0]

        Y_cross3_111[max_k:, idx] = (3.0 * np.sqrt(3.0)) * (u_k1 * u_k2 * u_k3)
    # Cross Degree 3 (2+1 interaction)
    c_cross3_21 = evaluate_capacity(
        X_train,
        X_test,
        Y_cross3_21[start_idx : start_idx + train_steps],
        Y_cross3_21[start_idx + train_steps : start_idx + train_steps + test_steps],
        num_21,
        ridge_alpha,
    )
    # Cross Degree 3 (1+1+1 interaction)
    c_cross3_111 = evaluate_capacity(
        X_train,
        X_test,
        Y_cross3_111[start_idx : start_idx + train_steps],
        Y_cross3_111[start_idx + train_steps : start_idx + train_steps + test_steps],
        num_111,
        ridge_alpha,
    )

    return [
        c_deg1,
        c_deg2,
        c_deg3,
        c_cross2,
        c_cross3_21,
        c_cross3_111,
    ]

rng = np.random.default_rng(42)


def objective(trial):
    if trial.number % 10 == 0:
        print(f"\r[Optuna] Processing Trial #{trial.number}...", end="", flush=True)
    trial_seeds = rng.integers(0, 2**31 - 1, size=3)
    trial.set_user_attr("trial_seeds", trial_seeds)

    start_time = time.perf_counter()
    rng_trials_results_avg = np.zeros(6)
    for trial_seed in trial_seeds:
        X, u = spring_trial(
            rng_seed=trial_seed,
            input_force=trial.suggest_float("input_force", 1, 50),
            m_val=trial.suggest_float("m_val", 0.001, 0.1, log=True),
            m_spread=0,
            c_val=trial.suggest_float("c_val", 0.01, 1.0, log=True),
            c_spread=0,
            k_wall_val=trial.suggest_float("k_wall_val", 1, 100, log=True),
            k_wall_spread=0,
            k_between_val=trial.suggest_float("k_between_val", 1, 100, log=True),
            k_between_spread=0,
            wall_rest_val=dist_y * 1.1,
            wall_rest_spread=0,
            between_rest_val=dist_between,
            between_rest_spread=0,
        )
        if np.isnan(X).any():
            raise optuna.TrialPruned("NaN detected in spring_trial results.")
        values = memory_trial(
            u,
            X,
            k_max,
            total_steps,
            start_idx,
            train_steps,
            test_steps,
            ridge_alpha=trial.suggest_float("ridge_alpha", 1e-6, 1e2, log=True),
        )
        values_sum = np.array([np.sum(c) for c in values])
        rng_trials_results_avg += values_sum
    elapsed_time = time.perf_counter() - start_time
    trial.set_user_attr("duration_sec", elapsed_time)

    results = rng_trials_results_avg / len(trial_seeds)
    return results[0] + results[1] + results[2], results[3] + results[4] + results[5]


order_study = optuna.create_study(directions=["maximize"] * 2)
optuna.logging.set_verbosity(optuna.logging.WARNING)
order_study.optimize(objective, timeout=60, n_jobs=-1)  # timeout, n_trials


df = order_study.trials_dataframe()
df.to_csv(trial_result_path, index=False)
best_trials_df = pd.DataFrame(
    [
        {"trial": t.number, "values": t.values, "params": t.params}
        for t in order_study.best_trials
    ]
)
best_trials_df.to_csv(best_results_path, index=False)
print("Finished Trials")
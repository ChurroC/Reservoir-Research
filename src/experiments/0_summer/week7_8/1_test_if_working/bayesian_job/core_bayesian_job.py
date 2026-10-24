import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import optuna
import pandas as pd
from dysts.maps import Henon
from numba import njit
import threading

# --- 1. CONFIGURATION & HPC PATH SETUP ---
steps = 100_000
N = 300
target_node_count = 10
timeout = 60

slurm_job_id = os.environ.get("SLURM_JOB_ID", "local_run")
os.makedirs(slurm_job_id, exist_ok=True)

trial_result_path = os.path.join(slurm_job_id, "trials_results.csv")
best_results_path = os.path.join(slurm_job_id, "best_results.csv")
config_path = os.path.join(slurm_job_id, "config.json")

config_metadata = {
    "steps": steps,
    "N": N,
    "target_node_count": target_node_count,
    "timeout": timeout,
}
with open(config_path, "w") as f:
    json.dump(config_metadata, f, indent=4)

# --- 2. TIME & DATA PREPARATION ---
tau_steps = 1
transient_steps_chaos = int(steps * 0.1)
transient_steps_reservoir = int(steps * 0.1)
total_steps = steps + transient_steps_chaos + transient_steps_reservoir + tau_steps
test_size = 0.2
test_steps = int(steps * test_size)

henon_model = Henon()
henon_dataset = henon_model.make_trajectory(total_steps)
henon_dataset = henon_dataset[transient_steps_chaos:]
henon_scaler = StandardScaler()

henon_train_scaled = henon_scaler.fit_transform(henon_dataset[:-test_steps])
henon_test_scaled = henon_scaler.transform(henon_dataset[-test_steps:])
henon_scaled = np.concatenate((henon_train_scaled, henon_test_scaled), axis=0)

# --- 3. GEOMETRY & TOPOLOGY SETUP ---
dist_between = 2.5
x = np.zeros(N * 3)
for i in range(0, N):
    x[i * 3 : (i + 1) * 3] = np.array([0, 0, 0]) + dist_between * i
y = np.tile(np.array([0, 1, 2]), N)
nodes_pos = np.column_stack((x, y))

dt = 0.01
N_step = int(N / (target_node_count + 1))
target_nodes = 1 + (np.arange(1, (target_node_count + 1)) * N_step) * 3
starts = 3 * np.arange(N)
wall_nodes = np.column_stack([starts, starts + 2]).flatten()
num_nodes = nodes_pos.shape[0]
dims = nodes_pos.shape[1]
matrix_size = num_nodes * dims

node_ids = np.arange(x.size)
wall_src = np.column_stack([starts, starts + 1]).flatten()
wall_dst = np.column_stack([starts + 1, starts + 2]).flatten()
betweens = np.arange(1, node_ids[-1], 3)
between_src = betweens[:-1]
between_dst = betweens[1:]
src_nodes = np.concatenate([wall_src, between_src])
dst_nodes = np.concatenate([wall_dst, between_dst])
connections_list = np.column_stack((src_nodes, dst_nodes)).astype(np.int64)

init_vecs = nodes_pos[connections_list[:, 0]] - nodes_pos[connections_list[:, 1]]
rest_lens = np.sqrt(np.sum(init_vecs**2, axis=1))

total_steps_with_free = steps + transient_steps_reservoir + tau_steps
U_host = np.zeros((total_steps_with_free, matrix_size))
for i, node_index in enumerate(target_nodes):
    U_host[:, node_index * dims] = henon_scaled[:, i % dims]

movement_nodes = 1 + np.arange(N) * 3
movement_idx = movement_nodes * dims
movement_nodes_pos = nodes_pos[movement_nodes, 0]
base_wall_rest = rest_lens[: wall_src.shape[0]]
base_between_rest = rest_lens[wall_src.shape[0] :]


# --- 4. NUMBA SIMULATION KERNELS ---
@njit(cache=True)
def get_spring_forces_numba(
    connections_list, disp, initial_pos, rest_lens, k_vals, num_nodes, dims, forces
):
    forces.fill(0)
    disp_reshaped = disp.reshape(num_nodes, dims)

    for idx in range(connections_list.shape[0]):
        idx_a = connections_list[idx, 0]
        idx_b = connections_list[idx, 1]

        pos_a = initial_pos[idx_a] + disp_reshaped[idx_a]
        pos_b = initial_pos[idx_b] + disp_reshaped[idx_b]

        r_vecs = pos_b - pos_a
        current_len = np.sqrt(np.sum(r_vecs**2))

        if current_len == 0:
            continue

        force_magnitude = k_vals[idx] * (current_len - rest_lens[idx])
        unit_dir = r_vecs / current_len

        forces[idx_a] += force_magnitude * unit_dir
        forces[idx_b] -= force_magnitude * unit_dir

    return forces.reshape(-1)


@njit(cache=True)
def run_simulation_numba(
    steps,
    dt,
    m_inv_diag,
    c_diag,
    U,
    initial_pos,
    connections_list,
    k_vals,
    rest_lens,
    wall_nodes,
    disp,
    v,
    acc,
    acc_next,
    forces_buf,
):
    disp.fill(0)
    v.fill(0)
    acc.fill(0)
    acc_next.fill(0)

    mask = np.ones(matrix_size)
    if wall_nodes[0] != -1:
        for wall in wall_nodes:
            idx = wall * dims
            mask[idx : idx + dims] = 0

    F_spring = get_spring_forces_numba(
        connections_list,
        disp[0],
        initial_pos,
        rest_lens,
        k_vals,
        num_nodes,
        dims,
        forces_buf,
    )

    for i in range(1, steps):
        acc[:] = m_inv_diag * (F_spring - c_diag * v[i - 1] + U[i - 1])
        acc *= mask

        disp[i] = disp[i - 1] + v[i - 1] * dt + acc * 0.5 * dt**2

        F_spring = get_spring_forces_numba(
            connections_list,
            disp[i],
            initial_pos,
            rest_lens,
            k_vals,
            num_nodes,
            dims,
            forces_buf,
        )

        acc_next[:] = m_inv_diag * (
            F_spring - c_diag * (v[i - 1] + 0.5 * acc * dt) + U[i]
        )
        acc_next *= mask

        v[i] = v[i - 1] + 0.5 * (acc + acc_next) * dt

    return disp, v


# Thread-local storage to securely hold preallocated buffers per parallel core worker thread
thread_local = threading.local()


def get_worker_buffers():
    if not hasattr(thread_local, "disp"):
        thread_local.disp = np.zeros(
            (total_steps_with_free, matrix_size), dtype=np.float64
        )
        thread_local.v = np.zeros(
            (total_steps_with_free, matrix_size), dtype=np.float64
        )
        thread_local.acc = np.zeros(matrix_size, dtype=np.float64)
        thread_local.acc_next = np.zeros(matrix_size, dtype=np.float64)
        thread_local.forces_buf = np.zeros((num_nodes, dims), dtype=np.float64)
    return (
        thread_local.disp,
        thread_local.v,
        thread_local.acc,
        thread_local.acc_next,
        thread_local.forces_buf,
    )


# --- 5. TRIAL EVALUATION FUNCTION ---
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
    wall_rest_spread,
    between_rest_spread,
    ridge_alpha,
):
    np.random.seed(rng_seed)
    random_rest_lens = np.empty(rest_lens.shape, dtype=float)

    mu = np.log(m_val) - (m_spread**2 / 2)
    m_nodes = np.random.lognormal(mean=mu, sigma=m_spread, size=num_nodes)
    m_diag = np.repeat(m_nodes, dims)
    m_inv_diag = 1.0 / m_diag

    mu = np.log(c_val) - (c_spread**2 / 2)
    c_nodes = np.random.lognormal(mean=mu, sigma=c_spread, size=num_nodes)
    c_diag = np.repeat(c_nodes, dims)

    mu = np.log(k_wall_val) - (k_wall_spread**2 / 2)
    k_wall_vals = np.random.lognormal(
        mean=mu, sigma=k_wall_spread, size=wall_src.shape[0] // 2
    ).repeat(2)

    mu = np.log(k_between_val) - (k_between_spread**2 / 2)
    k_between_vals = np.random.lognormal(
        mean=mu, sigma=k_between_spread, size=between_src.shape[0]
    )
    k_vals = np.concatenate([k_wall_vals, k_between_vals])

    rest_val = 1.0
    mu = np.log(rest_val) - (wall_rest_spread**2 / 2)
    wall_rest_len_randomness = np.random.lognormal(
        mean=mu, sigma=wall_rest_spread, size=(wall_src.shape[0]) // 2
    ).repeat(2)

    mu = np.log(rest_val) - (between_rest_spread**2 / 2)
    between_rest_len_randomness = np.random.lognormal(
        mean=mu, sigma=between_rest_spread, size=between_src.shape[0]
    )

    random_rest_lens[: wall_src.shape[0]] = base_wall_rest * wall_rest_len_randomness
    random_rest_lens[wall_src.shape[0] :] = (
        base_between_rest * between_rest_len_randomness
    )

    # Fetch preallocated thread-local RAM buffers
    disp_buf, v_buf, acc_buf, acc_next_buf, forces_buf = get_worker_buffers()

    displacement, velocity = run_simulation_numba(
        steps=total_steps_with_free,
        dt=dt,
        m_inv_diag=m_inv_diag,
        c_diag=c_diag,
        U=U_host * input_force,
        initial_pos=nodes_pos,
        connections_list=connections_list,
        k_vals=k_vals,
        rest_lens=random_rest_lens,
        wall_nodes=wall_nodes,
        disp=disp_buf,
        v=v_buf,
        acc=acc_buf,
        acc_next=acc_next_buf,
        forces_buf=forces_buf,
    )

    movement_disp = displacement[:, movement_idx]
    movement_vel = velocity[:, movement_idx]
    X = np.column_stack((movement_disp, movement_vel))

    positions = movement_nodes_pos + movement_disp.reshape(-1, len(movement_nodes))
    node_spacings = np.diff(positions, axis=1)
    overlap_mask = node_spacings <= 0.01
    is_finite = np.isfinite(X).all(axis=1)
    invalid_steps_mask = ~is_finite
    has_overlap = np.any(overlap_mask)
    has_invalid = np.any(invalid_steps_mask)

    if has_overlap or has_invalid:
        first_overlap = int(np.where(overlap_mask)[0][0]) if has_overlap else len(X)
        first_invalid = (
            int(np.where(invalid_steps_mask)[0][0]) if has_invalid else len(X)
        )
        first_failure = min(len(X), first_overlap, first_invalid)
        survival_fraction = first_failure / len(X)
        return -1.0 * (2.0 - survival_fraction), 1e6 / (survival_fraction + 1e-3)

    X_delayed = X[:-tau_steps]
    X_data = X_delayed[transient_steps_reservoir:]
    Y_data = henon_scaled[transient_steps_reservoir + tau_steps :]
    X_train, X_test = X_data[:-test_steps], X_data[-test_steps:]
    Y_train, Y_test = Y_data[:-test_steps], Y_data[-test_steps:]

    try:
        X_mean = np.mean(X_train, axis=0)
        Y_mean = np.mean(Y_train, axis=0)
        X_train_centered = X_train - X_mean
        Y_train_centered = Y_train - Y_mean
        XtX = X_train_centered.T @ X_train_centered
        XtX.flat[:: X_train_centered.shape[1] + 1] += ridge_alpha
        w = np.linalg.solve(XtX, X_train_centered.T @ Y_train_centered)
        bias = Y_mean - (X_mean @ w)
        Y_pred = (X_test @ w) + bias
    except np.linalg.LinAlgError:
        return -1.0, 1e9

    ss_res = np.sum((Y_test - Y_pred) ** 2)
    ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
    r_2 = float(1.0 - (ss_res / ss_tot))
    mse = float(np.sqrt(np.mean((Y_test - Y_pred) ** 2)))
    return r_2, mse


rng = np.random.default_rng(42)


# --- 6. OPTUNA OBJECTIVE & MULTIPROCESSING SETUP ---
def objective(trial):
    print(f"\r[Optuna] Processing Trial #{trial.number}...", end="", flush=True)

    trial_seed = int(rng.integers(0, 2**31 - 1))
    trial.set_user_attr("trial_seed", trial_seed)

    if trial.number % 1000 == 0 and trial.number > 0:
        df = study.trials_dataframe()
        best_trials_df = pd.DataFrame(
            [
                {
                    "number": t.number,
                    "input_force": t.params["input_force"],
                    "r2": t.values[0],
                    "mse": t.values[1],
                }
                for t in study.best_trials
            ]
        )
        df.to_csv(trial_result_path, index=False)
        best_trials_df.to_csv(best_results_path, index=False)

    r_2, mse = spring_trial(
        rng_seed=trial_seed,
        input_force=trial.suggest_float("input_force", 1, 50),
        m_val=trial.suggest_float("m_val", 0.001, 0.1, log=True),
        m_spread=trial.suggest_float("m_spread", 0, 1.5, log=False),
        c_val=trial.suggest_float("c_val", 0.01, 1.0, log=True),
        c_spread=trial.suggest_float("c_spread", 0, 1.5, log=False),
        k_wall_val=trial.suggest_float("k_wall_val", 1, 100, log=True),
        k_wall_spread=trial.suggest_float("k_wall_spread", 0, 1.5, log=False),
        k_between_val=trial.suggest_float("k_between_val", 1, 100, log=True),
        k_between_spread=trial.suggest_float("k_between_spread", 0, 1.5, log=False),
        wall_rest_spread=trial.suggest_float("wall_rest_spread", 0, 1.5, log=False),
        between_rest_spread=trial.suggest_float(
            "between_rest_spread", 0, 1.5, log=False
        ),
        ridge_alpha=trial.suggest_float("ridge_alpha", 1e-6, 1e2, log=True),
    )

    return r_2, mse


if __name__ == "__main__":
    study = optuna.create_study(directions=["maximize", "minimize"])
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Utilizing all available CPU cores safely via thread-local memory buffers
    study.optimize(objective, timeout=timeout, n_jobs=32)

    df = study.trials_dataframe()
    best_trials_df = pd.DataFrame(
        [
            {
                "number": t.number,
                "input_force": t.params["input_force"],
                "r2": t.values[0],
                "mse": t.values[1],
            }
            for t in study.best_trials
        ]
    )
    df.to_csv(trial_result_path, index=False)
    best_trials_df.to_csv(best_results_path, index=False)
    print("\nFinished Trials")

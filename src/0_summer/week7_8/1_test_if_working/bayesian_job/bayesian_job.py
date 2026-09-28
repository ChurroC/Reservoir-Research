import numpy as np
import cupy as cp
import cupyx
from sklearn.preprocessing import StandardScaler
import optuna
from dysts.maps import Henon
import pandas as pd
import os
import json

# steps = 100_000
# N = 1_000
# timeout = 300
# target_node_count = 10

steps = 20_000
N = 12
target_node_count = 3
timeout = 60


slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
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


tau_steps = 1
transient_steps_chaos = int(steps * 0.1)
transient_steps_reservoir = int(steps * 0.1)
total_steps = steps + transient_steps_chaos + transient_steps_reservoir + tau_steps
total_steps_after_chaos = steps + transient_steps_reservoir + tau_steps
test_size = 0.2
test_steps = int(steps * test_size)
t = np.arange(0, total_steps)

henon_model = Henon()
henon_dataset = henon_model.make_trajectory(total_steps)
henon_dataset = henon_dataset[transient_steps_chaos:]
henon_scaler = StandardScaler()

henon_train_scaled = henon_scaler.fit_transform(henon_dataset[:-test_steps])
henon_test_scaled = henon_scaler.transform(henon_dataset[-test_steps:])
henon_scaled = np.concatenate((henon_train_scaled, henon_test_scaled), axis=0)
henon_scaled_cp = cp.asarray(henon_scaled)

dist_between = 2.5
x = np.zeros(N * 3)
for i in range(0, N):
    x[i * 3 : (i + 1) * 3] = np.array([0, 0, 0]) + dist_between * i
y = np.tile(np.array([0, 1, 2]), N)
nodes_pos = np.column_stack((x, y))
nodes_pos_cp = cp.asarray(nodes_pos)

dt = 0.01
N_step = int(N / (target_node_count + 1))
target_nodes = 1 + (np.arange(1, (target_node_count + 1)) * N_step) * 3
starts = 3 * np.arange(N)
wall_nodes = np.column_stack([starts, starts + 2]).flatten()
num_nodes = nodes_pos.shape[0]
dims = nodes_pos.shape[1]
matrix_size = num_nodes * dims

node_ids = np.arange(x.size)
starts = 3 * np.arange(N)
wall_src = np.column_stack([starts, starts + 1]).flatten()
wall_dst = np.column_stack([starts + 1, starts + 2]).flatten()
betweens = np.arange(1, node_ids[-1], 3)
between_src = betweens[:-1]
between_dst = betweens[1:]
src_nodes = np.concatenate([wall_src, between_src])
dst_nodes = np.concatenate([wall_dst, between_dst])
connections_list = np.column_stack((src_nodes, dst_nodes))
connections_list_cp = cp.asarray(connections_list)

init_vecs = nodes_pos[connections_list[:, 0]] - nodes_pos[connections_list[:, 1]]
rest_lens = np.sqrt(np.sum(init_vecs**2, axis=1))
rest_lens_cp = cp.asarray(rest_lens)

total_steps_with_free = steps + transient_steps_reservoir + tau_steps
U = np.zeros((total_steps_with_free, matrix_size))
for i, node_index in enumerate(target_nodes):
    U[:, node_index * dims] = henon_scaled[:, i % dims]
U_cp = cp.asarray(U)

movement_nodes = 1 + np.arange(N) * 3
movement_idx = movement_nodes * dims
movement_idx_cp = cp.asarray(movement_idx)
movement_nodes_pos = nodes_pos_cp[movement_nodes, 0]
base_wall_rest_cp = rest_lens_cp[: wall_src.shape[0]]
base_between_rest_cp = rest_lens_cp[wall_src.shape[0] :]
random_rest_lens_cp = cp.empty(rest_lens.shape, dtype=float)


def get_spring_forces(
    connections_list, disp, initial_pos, rest_lens, k_vals, num_nodes, dims
):
    forces = cp.zeros((num_nodes, dims))

    idx_a, idx_b = connections_list[:, 0], connections_list[:, 1]

    disp_reshaped = disp.reshape(num_nodes, dims)

    pos_a = initial_pos[idx_a] + disp_reshaped[idx_a]
    pos_b = initial_pos[idx_b] + disp_reshaped[idx_b]

    r_vecs = pos_b - pos_a
    current_lens = cp.sqrt(cp.sum(r_vecs**2, axis=1))

    force_magnitudes = k_vals * (current_lens - rest_lens)

    unit_dirs = r_vecs / current_lens.reshape(-1, 1)

    cupyx.scatter_add(forces, idx_a, force_magnitudes[:, cp.newaxis] * unit_dirs)
    cupyx.scatter_add(forces, idx_b, -force_magnitudes[:, cp.newaxis] * unit_dirs)

    return forces.reshape(-1)


def run_simulation(
    steps,
    dt,
    m_inv_diag,
    c_diag,
    U,
    initial_pos,
    connections_list,
    k_vals,
    rest_lens,
    wall_nodes=[-1],
    disp=None,
    v=None,
):
    if disp is None:
        disp = cp.zeros((steps, matrix_size))
    if v is None:
        v = cp.zeros((steps, matrix_size))

    disp.fill(0)
    v.fill(0)

    mask = cp.ones(matrix_size)
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

        acc_next[:] = m_inv_diag * (
            F_spring - c_diag * (v[i - 1] + 0.5 * acc * dt) + U[i]
        )
        acc_next *= mask

        v[i] = v[i - 1] + 0.5 * (acc + acc_next) * dt

    return disp, v


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
    cp.random.seed(rng_seed)

    mu = cp.log(m_val) - (m_spread**2 / 2)
    m_nodes = cp.random.lognormal(mean=mu, sigma=m_spread, size=num_nodes)
    m_diag = cp.repeat(m_nodes, dims)
    m_inv_diag_cp = cp.asarray(1.0 / m_diag)

    mu = cp.log(c_val) - (c_spread**2 / 2)
    c_nodes = cp.random.lognormal(mean=mu, sigma=c_spread, size=num_nodes)
    c_diag_cp = cp.asarray(cp.repeat(c_nodes, dims))

    mu = cp.log(k_wall_val) - (k_wall_spread**2 / 2)
    k_wall_vals = cp.random.lognormal(
        mean=mu, sigma=k_wall_spread, size=wall_src.shape[0] // 2
    ).repeat(2)
    mu = cp.log(k_between_val) - (k_between_spread**2 / 2)
    k_between_vals = cp.random.lognormal(
        mean=mu, sigma=k_between_spread, size=between_src.shape[0]
    )
    k_vals = cp.concatenate([k_wall_vals, k_between_vals])
    k_vals_cp = cp.asarray(k_vals)

    rest_val = 1.0
    mu = cp.log(rest_val) - (wall_rest_spread**2 / 2)
    wall_rest_len_randomness = cp.random.lognormal(
        mean=mu, sigma=wall_rest_spread, size=(wall_src.shape[0]) // 2
    ).repeat(2)
    mu = cp.log(rest_val) - (between_rest_spread**2 / 2)
    between_rest_len_randomness = cp.random.lognormal(
        mean=mu, sigma=between_rest_spread, size=between_src.shape[0]
    )
    random_rest_lens_cp[: wall_src.shape[0]] = (
        base_wall_rest_cp * wall_rest_len_randomness
    )
    random_rest_lens_cp[wall_src.shape[0] :] = (
        base_between_rest_cp * between_rest_len_randomness
    )

    displacement_cp, velocity_cp = run_simulation(
        steps=total_steps_with_free,
        dt=dt,
        m_inv_diag=m_inv_diag_cp,
        c_diag=c_diag_cp,
        U=U_cp * input_force,
        initial_pos=nodes_pos_cp,
        connections_list=connections_list_cp,
        k_vals=k_vals_cp,
        rest_lens=rest_lens_cp,
        wall_nodes=wall_nodes,
    )

    movement_disp = displacement_cp[:, movement_idx_cp]
    movement_vel = velocity_cp[:, movement_idx_cp]
    X = cp.column_stack((movement_disp, movement_vel))

    positions = movement_nodes_pos + movement_disp.reshape(-1, len(movement_nodes))
    node_spacings = cp.diff(positions, axis=1)
    overlap_mask = node_spacings <= 0.01
    is_finite = cp.isfinite(X).all(axis=1)
    invalid_steps_mask = ~is_finite
    has_overlap = cp.any(overlap_mask)
    has_invalid = cp.any(invalid_steps_mask)
    if has_overlap or has_invalid:
        first_overlap = int(cp.where(overlap_mask)[0][0]) if has_overlap else len(X)
        first_invalid = (
            int(cp.where(invalid_steps_mask)[0][0]) if has_invalid else len(X)
        )
        first_failure = min(len(X), first_overlap, first_invalid)
        survival_fraction = first_failure / len(X)
        penalized_r2 = -1.0 * (2.0 - survival_fraction)
        penalized_mse = 1e6 / (survival_fraction + 1e-3)
        return penalized_r2, penalized_mse

    X_delayed = X[:-tau_steps]
    X_data = X_delayed[transient_steps_reservoir:]
    Y_data = henon_scaled_cp[transient_steps_reservoir + tau_steps :]
    X_train, X_test = (
        X_data[:-test_steps],
        X_data[-test_steps:],
    )
    Y_train, Y_test = (
        Y_data[:-test_steps],
        Y_data[-test_steps:],
    )

    try:
        X_mean = cp.mean(X_train, axis=0)
        Y_mean = cp.mean(Y_train, axis=0)
        X_train_centered = X_train - X_mean
        Y_train_centered = Y_train - Y_mean
        XtX = X_train_centered.T @ X_train_centered
        XtX.flat[:: X_train_centered.shape[1] + 1] += ridge_alpha
        w = cp.linalg.solve(XtX, X_train_centered.T @ Y_train_centered)
        bias = Y_mean - (X_mean @ w)
        Y_pred = (X_test @ w) + bias
    except cp.linalg.LinAlgError:
        return -1.0, 1e9

    ss_res = cp.sum((Y_test - Y_pred) ** 2)
    ss_tot = cp.sum((Y_test - cp.mean(Y_test)) ** 2)
    r_2 = 1.0 - (ss_res / ss_tot)

    mse = cp.sqrt(cp.mean((Y_test - Y_pred) ** 2))

    r_2 = float(r_2.get())
    mse = float(mse.get())
    return r_2, mse


rng = np.random.default_rng(42)


def objective(trial):
    # if trial.number % 100 == 0:
    print(f"\r[Optuna] Processing Trial #{trial.number}...", end="", flush=True)

    trial_seed = rng.integers(0, 2**31 - 1)
    trial.set_user_attr("trial_seed", trial_seed)

    if trial.number % 1000 == 0:
        df = study.trials_dataframe()
        df.to_csv(trial_result_path, index=False)
        best_trials_df = pd.DataFrame(
            [
                {"trial": t.number, "values": t.values, "params": t.params}
                for t in study.best_trials
            ]
        )
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

    print(f"\r[Optuna] Ending Trial #{trial.number}...", end="", flush=True)
    return r_2, mse


if __name__ == "__main__":
    study = optuna.create_study(directions=["maximize", "minimize"])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, timeout=timeout, n_jobs=1)

    df = study.trials_dataframe()
    df.to_csv(trial_result_path, index=False)
    best_trials_df = pd.DataFrame(
        [
            {"trial": t.number, "values": t.values, "params": t.params}
            for t in study.best_trials
        ]
    )
    best_trials_df.to_csv(best_results_path, index=False)
    print("Finished Trials")

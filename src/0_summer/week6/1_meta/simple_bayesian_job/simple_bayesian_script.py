import numpy as np

from numba import njit

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error

import pandas as pd

import optuna

# Henon Stuff
steps = 20000
tau_steps = 1
transient_steps_henon = int(steps * 0.1)
transient_steps_reservoir = int(steps * 0.1)
total_steps = steps + transient_steps_henon + transient_steps_reservoir + tau_steps
total_steps_after_henon = steps + transient_steps_reservoir + tau_steps
test_size = 0.2
test_steps = int(steps * test_size)
t = np.arange(0, total_steps)

henon_dataset = np.zeros((total_steps, 2))
rng = np.random.default_rng(42)
henon_dataset[0] = rng.random(2)
a = 1.4
b = 0.3


@njit(fastmath=True, cache=True)
def henon_numba(steps, a=1.4, b=0.3, x0=0.0, y0=0.0):
    X = np.zeros(steps)
    Y = np.zeros(steps)
    X[0] = x0
    Y[0] = y0

    for i in range(1, steps):
        X[i] = 1 - a * X[i - 1] ** 2 + Y[i - 1]
        Y[i] = b * X[i - 1]

    return X, Y


henon_data_x, henon_data_y = henon_numba(total_steps)
henon_dataset = np.column_stack((henon_data_x, henon_data_y))
henon_dataset = henon_dataset[transient_steps_henon:]

henon_scaler = StandardScaler()
henon_train_scaled = henon_scaler.fit_transform(henon_dataset[:-test_steps])
henon_test_scaled = henon_scaler.transform(henon_dataset[-test_steps:])
henon_scaled = np.concatenate((henon_train_scaled, henon_test_scaled), axis=0)


# Calc stuff
@njit(fastmath=True, cache=True)
def create_stiffness_matrix(node_positions, connections, k_vals):
    num_nodes = node_positions.shape[0]
    dims = node_positions.shape[1]
    K = np.zeros((num_nodes * dims, num_nodes * dims))

    for node_conn, k_val in zip(connections, k_vals):
        node_pos = node_positions[node_conn]
        diff_vec = node_pos[1] - node_pos[0]
        unit_dir = diff_vec / np.linalg.norm(diff_vec)
        sub_block = np.outer(unit_dir, unit_dir)

        idx1 = node_conn[0] * dims
        idx2 = node_conn[1] * dims

        K[idx1 : idx1 + 2, idx1 : idx1 + 2] += k_val * sub_block
        K[idx2 : idx2 + 2, idx2 : idx2 + 2] += k_val * sub_block
        K[idx1 : idx1 + 2, idx2 : idx2 + 2] += k_val * -sub_block
        K[idx2 : idx2 + 2, idx1 : idx1 + 2] += k_val * -sub_block
    return K


@njit(fastmath=True, cache=True)
def run_simulation(
    steps,
    dt,
    matrix_size,
    M_INV,
    C,
    U,
    initial_pos,
    connections_list,
    k_vals,
    constrained_nodes,
    constrained_values,
):
    disp = np.zeros((steps, matrix_size))
    v = np.zeros((steps, matrix_size))
    acc = np.zeros(matrix_size)

    dims = initial_pos.shape[1]

    for i in range(1, steps):
        actual_pos = initial_pos + disp[i - 1].reshape(-1, dims)
        K = create_stiffness_matrix(actual_pos, connections_list, k_vals)
        for node in constrained_nodes:
            idx = node * dims
            K[idx, idx] += constrained_values
            K[idx + 1, idx + 1] += constrained_values

        acc = M_INV @ (-K @ disp[i - 1] - C @ v[i - 1] + U[i - 1])

        disp[i] = disp[i - 1] + v[i - 1] * dt + acc * 0.5 * dt**2

        acc_next = M_INV @ (-K @ disp[i] - C @ (v[i - 1] + acc * dt) + U[i])

        v[i] = v[i - 1] + 0.5 * (acc + acc_next) * dt

    return disp, v


# Single Hex
side_len = 1
x = np.array(
    [-side_len / 2, side_len / 2, side_len, side_len / 2, -side_len / 2, -side_len]
)
y = np.array(
    [
        0,
        0,
        side_len * np.sqrt(3) / 2,
        side_len * np.sqrt(3),
        side_len * np.sqrt(3),
        side_len * np.sqrt(3) / 2,
    ]
)
nodes_pos = np.column_stack((x, y))

rng = np.random.default_rng(42)

tau_steps = 1
num_nodes = nodes_pos.shape[0]
dims = nodes_pos.shape[1]
matrix_size = num_nodes * dims
node_m = rng.uniform(0.1, 0.3, size=num_nodes)
m_diag = np.repeat(node_m, dims)
m_inv_diag = 1.0 / m_diag
M = np.diag(m_diag)
M_INV = np.diag(m_inv_diag)
node_c = rng.uniform(0.05, 0.3, size=num_nodes)
c_diag = np.repeat(node_c, dims)
DAMP = np.diag(c_diag)
U = np.zeros((steps + transient_steps_reservoir + tau_steps, matrix_size))
target_nodes = np.array([2, 5])
col_indices = (target_nodes[:, None] * dims + np.arange(dims)).reshape(-1)
vectorized_force = np.zeros((U.shape[0], len(col_indices)))
vectorized_force[:, 0] = henon_scaled[:, 0]
vectorized_force[:, 2] = henon_scaled[:, 1]
U[:, col_indices] = vectorized_force

node_ids = np.arange(x.size)
src_nodes = node_ids
dst_nodes = np.roll(node_ids, 1)
k_vals = rng.uniform(0.5, 8, size=src_nodes.shape[0])
connections_list = np.column_stack((src_nodes, dst_nodes))


def hyper_param_input(input_force):
    displacement, velocity = run_simulation(
        steps + transient_steps_reservoir + tau_steps,
        0.01,
        matrix_size,
        M_INV,
        DAMP,
        U * input_force,
        nodes_pos,
        connections_list,
        k_vals,
        [0, 1, 3, 4],
        100,
    )
    X = np.column_stack((displacement, velocity))

    X_delayed = X[:-tau_steps]
    X_data = X_delayed[transient_steps_reservoir:]
    x_scaler = StandardScaler()
    X_train_scaled, X_test = (
        x_scaler.fit_transform(X_data[:-test_steps]),
        x_scaler.transform(X_data[-test_steps:]),
    )
    Y_train_scaled, Y_test_scaled = (
        henon_train_scaled[transient_steps_reservoir + tau_steps :],
        henon_test_scaled,
    )
    model = RidgeCV()
    model.fit(X_train_scaled, Y_train_scaled)
    Y_pred_scaled = model.predict(X_test)
    Y_pred = henon_scaler.inverse_transform(Y_pred_scaled)
    Y_test = henon_scaler.inverse_transform(Y_test_scaled)

    return Y_test, Y_pred


def objective(trial):
    print(f"\r[Optuna] Processing Trial #{trial.number}...", end="", flush=True)

    input_force = trial.suggest_float("input_force", 0.1, 100.0)

    Y_test, Y_pred = hyper_param_input(input_force)

    r_2 = r2_score(Y_test, Y_pred)
    mse = root_mean_squared_error(Y_test, Y_pred)

    return r_2, mse


study = optuna.create_study(directions=["maximize", "minimize"])
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=5000, n_jobs=-1)


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
df.to_csv("all_trials_results.csv", index=False)
best_trials_df.to_csv("best_pareto_trials.csv", index=False)

print("Finished Trials")

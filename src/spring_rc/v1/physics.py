"""
Physics engine that uses Velocity-Verlet integration for a 2D mass-spring lattice
"""

import numpy as np
from numba import njit

import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@njit(cache=True)
def compute_spring_forces(
    conn: IntArray,
    disp: FloatArray,
    init_pos: FloatArray,
    rest: FloatArray,
    k: FloatArray,
    f: FloatArray,
) -> None:
    """Compute spring forces (-kx) in place.

    Parameters
    ----------
    conn : ndarray of shape (n_springs, 2)
        Edge connection list indexing connected pairs ``[node_a, node_b]``.
    disp : ndarray of shape (n, d)
        Current displacement of each node relative to rest position.
    init_pos : ndarray of shape (n, d)
        Initial starting positions of all the nodes.
    rest : ndarray of shape (n_springs,)
        Equilibrium rest length of each spring. Values greater than the initial
        node separation caused compression and bistability.
    k : ndarray of shape (n_springs,)
        Stiffness coefficient for each spring.
    f : ndarray of shape (n, d)
        An output buffer where computed forces are accumulated in place.

    Returns
    -------
    None
        Results are written directly to the parameter buffer ``f``.
    """

    f.fill(0.0)

    for s in range(len(conn)):
        a = conn[s, 0]
        b = conn[s, 1]

        dx = (init_pos[b, 0] + disp[b, 0]) - (init_pos[a, 0] + disp[a, 0])
        dy = (init_pos[b, 1] + disp[b, 1]) - (init_pos[a, 1] + disp[a, 1])

        dist = np.sqrt(dx * dx + dy * dy)
        inv_dist = 1.0 / (dist + 1e-12)
        mag = k[s] * (dist - rest[s])

        fx = mag * (dx * inv_dist)
        fy = mag * (dy * inv_dist)

        f[a, 0] += fx
        f[a, 1] += fy
        f[b, 0] -= fx
        f[b, 1] -= fy


@njit(cache=True)
def run_simulation(
    steps: int,
    free_steps: int,
    dt: float,
    m_inv: FloatArray,
    c_diag: FloatArray,
    F_in: FloatArray,
    target_nodes: IntArray,
    init_pos: FloatArray,
    conn: IntArray,
    k: FloatArray,
    rest: FloatArray,
    fixed_nodes: IntArray,
    rec_nodes: IntArray,
) -> tuple[FloatArray, FloatArray]:
    """Integrates using velocity verlet with impulses at each macro step and records states at the end of substeps.

    Parameters
    ----------
    steps : int
        Number of discrete macro steps (e.g., Hénon map iterations).
    free_steps : int
        Number of numerical integration substeps per macro step (ringdown window).
    dt : float
        Discretized integration timestep duration.
    m_inv : ndarray of shape (n, d)
        Inverse mass matrix (:math:`1/m`) per node and degree of freedom.
    c_diag : ndarray of shape (n, d)
        Viscous damping coefficient per node and degree of freedom.
    F_in : ndarray of shape (steps, len(target_nodes), d)
        External force impulse vector per driving node per macro step.
    target_nodes : ndarray of shape (n_inputs,)
        Indices of nodes receiving driving input kicks.
    init_pos : ndarray of shape (n, d)
        Equilibrium reference coordinates of all nodes.
    conn : ndarray of shape (n_springs, 2)
        Edge list of connected node pairs ``[node_a, node_b]``.
    k : ndarray of shape (n_springs,)
        Stiffness coefficient for each spring.
    rest : ndarray of shape (n_springs,)
        Equilibrium rest length for each spring.
    fixed_nodes : ndarray of shape (n_fixed,)
        Indices of boundary nodes pinned to rigid walls.
    rec_nodes : ndarray of shape (n_rec,)
        Indices of nodes whose horizontal states are saved as features.

    Returns
    -------
    Xd : ndarray of shape (steps, len(rec_nodes))
        Horizontal displacements recorded at the terminal substep of each macro step.
    Xv : ndarray of shape (steps, len(rec_nodes))
        Horizontal velocities recorded at the terminal substep of each macro step.
    """
    n, d = init_pos.shape

    mask = np.ones((n, d))
    for node in fixed_nodes:
        mask[node, :] = 0.0

    # 18.75 KB active state buffers
    x = np.zeros((n, d))
    v = np.zeros((n, d))
    F = np.zeros((n, d))
    F_next = np.zeros((n, d))
    U = np.zeros((n, d))

    # Output which doesn't count the free steps
    Xd = np.zeros((steps, len(rec_nodes)))
    Xv = np.zeros((steps, len(rec_nodes)))

    compute_spring_forces(conn, x, init_pos, rest, k, F)
    damping_inv = 1.0 / (1.0 + 0.5 * dt * c_diag * m_inv)

    total_substeps = steps * free_steps
    for i in range(total_substeps):
        h = i // free_steps
        sub = i % free_steps

        if sub == 0:
            for j in range(len(target_nodes)):
                U[target_nodes[j]] = F_in[h, j]

        acc = m_inv * (F - c_diag * v + U) * mask
        x += v * dt + 0.5 * acc * dt**2

        if sub == 0:
            for j in range(len(target_nodes)):
                U[target_nodes[j], :] = 0.0

        compute_spring_forces(conn, x, init_pos, rest, k, F_next)

        v = (v + 0.5 * dt * (acc + m_inv * F_next * mask)) * damping_inv * mask

        if sub == free_steps - 1:
            for j in range(len(rec_nodes)):
                Xd[h, j] = x[rec_nodes[j], 0]
                Xv[h, j] = v[rec_nodes[j], 0]

    return Xd, Xv

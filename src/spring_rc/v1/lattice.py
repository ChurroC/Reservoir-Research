"""
Build a lattice
"""

import numpy as np


import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def lognormal_mu(mean_val: float, sigma: float) -> float:
    if mean_val <= 0:
        return -np.inf
    return np.log(mean_val) - (sigma**2 / 2)


def lognormal(
    rng: np.random.Generator, mean_val: float, sigma: float, size: int
) -> float:
    return rng.lognormal(mean=lognormal_mu(mean_val, sigma), sigma=sigma, size=size)


def build_von_mises_lattice(
    rng: np.random.Generator,
    num_cells: int = 100,
    cell_spacing: float = 1.0,
    wall_dist: float = 1.0,
    mass_val: float = 1.0,
    damping_val: float = 0.1,
    k_wall_val: float = 10.0,
    k_wall_spread: float = 0.0,
    k_inter_val: float = 1.0,
    k_inter_spread: float = 0.0,
    rest_wall_ratio: float = 1.0,
    rest_wall_ratio_spread: float = 0.0,
    rest_inter_ratio: float = 1.0,
    rest_inter_ratio_spread: float = 0.0,
) -> tuple[
    IntArray, FloatArray, FloatArray, IntArray, IntArray, FloatArray, FloatArray
]:
    """Build a 1D chain of 2D von Mises mass-spring cells.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for generating stochastic properties.
    num_cells : int
        Number of von Mises cells in the lattice.
    cell_spacing : float
        Spacing between the cells.
    wall_dist : float
        Distance of the walls from the center.
    mass_val : float
        Mass value for the cells.
    damping_val : float
        Damping value for the cells.
    k_wall_val : float
        Spring constant for the wall springs.
    k_wall_spread : float
        Spread of the wall spring constants.
    k_inter_val : float
        Spring constant for the inter-cell springs.
    k_inter_spread : float
        Spread of the inter-cell spring constants.
    rest_wall_ratio : float
        Rest length ratio for the wall springs.
    rest_wall_ratio_spread : float
        Spread of the rest length ratios for the wall springs.
    rest_inter_ratio : float
        Rest length ratio for the inter-cell springs.
    rest_inter_ratio_spread : float
        Spread of the rest length ratios for the inter-cell springs.

    Returns
    -------
    init_pos : ndarray of shape (n, 2)
        Initial positions of the nodes in the lattice.
    m_inv_diag : ndarray of shape (n,)
        Inverse mass diagonal elements for each node.
    c_diag : ndarray of shape (n,)
        Damping diagonal elements for each node.
    connections_list : ndarray of shape (n_springs, 2)
        Edge connection list indexing connected pairs ``[node_a, node_b]``.
    wall_nodes : ndarray of shape (n_walls,)
        Indices of the wall nodes.
    k_vals : ndarray of shape (n_springs,)
        Stiffness coefficients for each spring.
    rest_lens : ndarray of shape (n_springs,)
        Rest lengths for each spring.

    """

    x = np.zeros(num_cells * 3)
    for i in range(0, num_cells):
        x[i * 3 : (i + 1) * 3] = np.array([0, 0, 0]) + cell_spacing * i
    y = np.tile(np.array([0, 1 * wall_dist, 2 * wall_dist]), num_cells)
    init_pos = np.column_stack((x, y))

    node_ids = np.arange(x.size)
    starts = 3 * np.arange(num_cells)
    connection_src = np.column_stack([starts, starts + 1]).flatten()
    connection_dst = np.column_stack([starts + 1, starts + 2]).flatten()
    betweens = np.arange(1, node_ids[-1], 3)
    between_src = betweens[:-1]
    between_dst = betweens[1:]
    src_nodes = np.concatenate([connection_src, between_src])
    dst_nodes = np.concatenate([connection_dst, between_dst])
    connections_list = np.column_stack((src_nodes, dst_nodes))

    wall_nodes = np.column_stack([starts, starts + 2]).flatten()

    m_nodes = np.ones(len(x)) * mass_val
    m_diag = np.repeat(m_nodes[:, np.newaxis], 2, axis=1)
    m_inv_diag = 1.0 / m_diag

    c_nodes = np.ones(len(x)) * damping_val
    c_diag = np.repeat(c_nodes[:, np.newaxis], 2, axis=1)

    k_wall_connections = lognormal(
        rng, k_wall_val, k_wall_spread, connection_src.shape[0] // 2
    ).repeat(2)
    k_inter_connections = lognormal(
        rng, k_inter_val, k_inter_spread, connection_src.shape[0] // 2
    ).repeat(2)
    k_vals = np.concatenate([k_wall_connections, k_inter_connections])

    # We'll subtract and add 1 so the rest lengths ratio are always greater than 1
    rest_wall_connections = (
        lognormal(
            rng,
            rest_wall_ratio - 1,
            rest_wall_ratio_spread,
            connection_src.shape[0] // 2,
        ).repeat(2)
        + 1
    ) * wall_dist
    rest_inter_connections = (
        lognormal(
            rng,
            rest_inter_ratio - 1,
            rest_inter_ratio_spread,
            connection_src.shape[0] // 2,
        ).repeat(2)
        + 1
    ) * cell_spacing
    rest_lens = np.concatenate([rest_wall_connections, rest_inter_connections])

    return (
        init_pos,
        m_inv_diag,
        c_diag,
        connections_list,
        wall_nodes,
        k_vals,
        rest_lens,
    )

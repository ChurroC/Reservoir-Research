"""
Plotting functions
"""

from typing import Iterable, Optional, Union, Literal
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots

FloatArray = npt.NDArray[np.float64]


def get_dim(data: np.ndarray) -> int:
    if data.ndim == 1:
        return 1
    elif data.ndim == 2:
        return data.shape[1]


def scatter_plot(
    data_list: Union[FloatArray, Iterable[FloatArray]],
    data_names: Optional[Iterable[str]] = None,
    axis_names: Optional[Iterable[str]] = None,
    colors: Optional[Iterable[str]] = None,
    title: str = "",
    marker_size: float = 1.5,
    opacity: float = 0.8,
) -> go.Figure:
    """Plotly scatter plot 1D time series or 2D or 3D phase planes,

    Parameters
    ----------
    data_list : ndarray or iterable of ndarrays
        Data to plot. Can be a single array or a list/iterable of arrays.
        - 1D: shape (n_steps,) -> Plots value vs. time step.
        - 2D: shape (n_steps, 2) -> Plots 2D phase space (x vs. y).
        - 3D: shape (n_steps, >=3) -> Plots 3D state space (x, y, z).
    data_names : iterable of str, optional
        Labels for each data trace. Default is "Data 0", "Data 1", etc.
    axis_names : iterable of str, optional
        Labels for each axis. Default is "X Axis", "Y Axis", "Z Axis". But 1D has "Time Step" and "Data Values".
    colors : iterable of str, optional
        Marker colors. Has default of white, magenta, cyan, etc.
    title : str, optional
        The title of the figure.
    marker_size : float, optional
        Size of the scatter markers. Default is 1.5.
    opacity : float, optional
        Opacity of the markers. Default is 0.8.

    Returns
    -------
    go.Figure
        The Plotly figure object. So you can still edit or turn into subplots or save it using the save_figure func.
    """

    # Make sure we have the data in something iterable
    if isinstance(data_list, np.ndarray):
        data_list = [data_list]
    else:
        # Here if it's a tuple or something
        data_list = list(data_list)

    if isinstance(data_names, str):
        data_names = [data_names]
    elif data_names is not None:
        data_names = list(data_names)
    else:
        data_names = [f"Data {i}" for i in range(len(data_list))]

    if isinstance(colors, str):
        colors = [colors]
    elif colors is not None:
        colors = list(colors)
    else:
        colors = ["white", "magenta", "cyan", "lime", "yellow", "orange"]

    fig = go.Figure()

    max_dim = 0
    for i, data in enumerate(data_list):
        name = data_names[i % len(data_names)]
        color = colors[i % len(colors)]
        dim = get_dim(data)
        max_dim = max(max_dim, dim)

        if dim == 1:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(data)),
                    y=data,
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=marker_size, opacity=opacity),
                )
            )
        elif dim == 2:
            fig.add_trace(
                go.Scatter(
                    x=data[:, 0],
                    y=data[:, 1],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=marker_size, opacity=opacity),
                )
            )
        elif dim == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=data[:, 0],
                    y=data[:, 1],
                    z=data[:, 2],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=marker_size, opacity=opacity),
                )
            )
        else:
            raise ValueError(f"Unsupported data shape {data.shape} at index {i}.")

    if isinstance(axis_names, str):
        axis_names = [axis_names]
    elif axis_names is not None:
        axis_names = list(axis_names)
    else:
        axis_names = []
    if max_dim == 1:
        default_axes = ["Time Step", "Data Values", ""]
    elif max_dim == 2 or max_dim == 3:
        default_axes = ["X", "Y", "Z"]
    x_label = axis_names[0] if len(axis_names) > 0 else default_axes[0]
    y_label = axis_names[1] if len(axis_names) > 1 else default_axes[1]
    z_label = axis_names[2] if len(axis_names) > 2 else default_axes[2]

    layout_update = {
        "template": "plotly_dark",
        "title": title,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    }

    if max_dim <= 2:
        layout_update.update({"xaxis_title": x_label, "yaxis_title": y_label})
        fig.update_layout(**layout_update)
    else:
        fig.update_layout(**layout_update)
        fig.update_layout(
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label,
            )
        )

    fig.update_layout(width=1000, height=500).update_xaxes(nticks=6).update_yaxes(
        nticks=5
    )

    return fig


def r_2_plots_grid(
    actual_list: FloatArray,
    predicted_list: FloatArray,
    r2_vals: Iterable[float],
    subplot_titles: Optional[Iterable[str]] = None,
    title: str = "",
    n_bins: int = 10,
) -> go.Figure:
    """Create a multi-panel grid comparing actual vs predicted states with

    binned error statistics and R² performance metrics integrated into titles.

    Parameters
    ----------
    actual_list : ndarray
        Ground truth data array of shape (n_samples, n_dims) or (n_samples,).
    predicted_list : ndarray
        Model prediction data array of shape (n_samples, n_dims) or (n_samples,).
    subplot_titles : iterable of str, optional
        Labels for each subplot dimension. Default is "Dim 0", "Dim 1", etc.
    title : str, optional
        The overall figure title.
    n_bins : int, optional
        Number of bins for statistical error bars. Default is 10.

    Returns
    -------
    go.Figure
        The generated Plotly figure object.
    """

    actual_list = actual_list.reshape(-1, 1) if actual_list.ndim == 1 else actual_list
    predicted_list = (
        predicted_list.reshape(-1, 1) if predicted_list.ndim == 1 else predicted_list
    )

    n_cols = actual_list.shape[1]

    # Normalize subplot_titles similar to scatter_plot
    if isinstance(subplot_titles, str):
        subplot_titles = [subplot_titles]
    elif subplot_titles is not None:
        subplot_titles = list(subplot_titles)
    else:
        subplot_titles = [f"Dim {i}" for i in range(n_cols)]

    # Precompute R² scores and format subplot titles cleanly
    subplot_titles = []
    for i in range(n_cols):
        name = subplot_titles[i % len(subplot_titles)]
        r2 = r2_vals[i % len(r2_vals)]
        subplot_titles.append(f"{name} (R²: {r2:.4f})")

    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=subplot_titles)

    for i in range(n_cols):
        actual = actual_list[:, i]
        predicted = predicted_list[:, i]
        col = i + 1

        # 1. Background scatter points (semi-transparent)
        fig.add_trace(
            go.Scatter(
                x=actual,
                y=predicted,
                mode="markers",
                name="Data",
                marker=dict(color="rgba(150, 150, 150, 0.3)", size=4),
                showlegend=False,
            ),
            row=1,
            col=col,
        )

        # 2. Compute binned error/spread statistics
        bin_edges = np.linspace(predicted.min(), predicted.max(), n_bins + 1)
        bin_centers = []
        mean_actuals = []
        std_actuals = []

        for b in range(n_bins):
            mask = (predicted >= bin_edges[b]) & (predicted < bin_edges[b + 1])
            if np.any(mask):
                bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                mean_actuals.append(np.mean(actual[mask]))
                std_actuals.append(np.std(actual[mask]))

        # 3. Add binned error bar summary trace
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=mean_actuals,
                mode="markers+lines",
                name="Binned Mean",
                marker=dict(color="firebrick", size=8, symbol="circle"),
                line=dict(color="firebrick", width=1.5, dash="dot"),
                error_y=dict(
                    type="data",
                    array=std_actuals,
                    visible=True,
                    color="firebrick",
                    thickness=1.5,
                    width=4,
                ),
                showlegend=False,
            ),
            row=1,
            col=col,
        )

        # 4. Ideal y = x line
        min_val, max_val = min(actual.min(), predicted.min()), max(
            actual.max(), predicted.max()
        )
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Ideal",
                line=dict(color="white", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=col,
        )

        fig.update_xaxes(title_text="Actual", row=1, col=col)
        fig.update_yaxes(title_text="Predicted", row=1, col=col)

    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=500,
        width=1000,
        showlegend=False,
    ).update_xaxes(nticks=6).update_yaxes(nticks=5)

    return fig


def save_figure(
    fig: go.Figure,
    filepath: str,
    format: Literal["png", "svg", "pdf", "jpeg", ""] = "svg",
    include_html: bool = False,
    scale: float = 3.0,
) -> None:
    """Save a Plotly figure as HTML or image"""
    dir_name = os.path.dirname(filepath)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    base_path, _ = os.path.splitext(filepath)

    if include_html:
        html_path = f"{base_path}.html"
        fig.write_html(html_path)
        # print(f"Saved interactive plot to: {html_path}")

    if format:
        format = format.lower().strip(".")
        if format not in ["png", "svg", "pdf", "jpeg"]:
            raise ValueError(f"Unsupported format: {format}")

        img_path = f"{base_path}.{format}"
        try:
            fig.write_image(img_path, scale=scale)
            # print(f"Saved presentation image to: {img_path}")
        except Exception as e:
            print(f"Static export failed need Kaleido and Chrome: {e}")


if __name__ == "__main__":
    # 1D plot
    time_steps = 1000
    true_signal = np.sin(np.linspace(0, 20, time_steps))
    pred_signal = np.sin(np.linspace(0, 20, time_steps)) + 0.05 * np.random.randn(
        time_steps
    )
    fig = scatter_plot(
        data_list=[true_signal, pred_signal],
        data_names=["Ground Truth", "Model Prediction"],
        colors=["white", "cyan"],
        title="1D Time Series Comparison",
    )
    save_figure(fig=fig, filepath="./plots/1D_plot")

    # 2D plot
    trajectory_actual = np.random.randn(1000, 2)
    trajectory_pred = trajectory_actual + 0.1 * np.random.randn(1000, 2)
    fig = scatter_plot(
        data_list=[trajectory_actual, trajectory_pred],
        data_names=["Actual Attractor", "Predicted Attractor"],
        colors=["white", "magenta"],
        title="2D Phase Plane Attractor",
    )
    save_figure(
        fig=fig,
        filepath="./plots/2D_plot",
    )

    # 3D plot
    def generate_lorenz(num_steps=5000, dt=0.01):
        xs = np.empty(num_steps)
        ys = np.empty(num_steps)
        zs = np.empty(num_steps)

        xs[0], ys[0], zs[0] = (1.0, 1.0, 1.0)

        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        for i in range(num_steps - 1):
            dx = sigma * (ys[i] - xs[i])
            dy = xs[i] * (rho - zs[i]) - ys[i]
            dz = xs[i] * ys[i] - beta * zs[i]
            xs[i + 1] = xs[i] + dx * dt
            ys[i + 1] = ys[i] + dy * dt
            zs[i + 1] = zs[i] + dz * dt

        return np.column_stack((xs, ys, zs))

    true_lorenz = generate_lorenz(4000)
    pred_lorenz = true_lorenz + np.random.normal(0, 0.2, true_lorenz.shape)
    fig = scatter_plot(
        data_list=[true_lorenz, pred_lorenz],
        data_names=["True Lorenz", "Reservoir Prediction"],
        axis_names=["X Axis", "Y Axis", "Z Axis"],
        colors=["cyan", "magenta"],
        title="Chaos Prediction: Lorenz Attractor",
        marker_size=1.2,
        opacity=0.6,
    )
    save_figure(
        fig=fig,
        filepath="./plots/3D_plot",
    )

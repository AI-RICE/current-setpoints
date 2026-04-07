import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from typing import Any, Optional


def plot_grid_segments(data_obj: Any) -> None:
    """
    Plots the motor operating regions (MTPA, FW, etc.) as a scatter plot
    on the torque-speed plane.

    Args:
        data_obj: An object (typically MachineData) containing unique_segments,
            segments, omega_grid, and torq_grid.
    """
    colors_map = {
        0: "#1f77b4",
        1: "#ff7f0e",
        2: "#2ca02c",
        3: "#d62728",
        4: "#9467bd",
        5: "#8c564b",
        6: "#bcbd22",
        7: "#17becf",
        8: "#7f7f7f",
    }

    # Map the integer values to Matplotlib's mathtext for subscripts
    labels_map = {
        0: r"MTPA$_{0}$",
        1: r"MTPA$_{1}$",
        2: r"MTPA$_{2}$",
        3: r"FW$_{I,0}$",
        4: r"FW$_{I,1}$",
        5: r"FW$_{I,2}$",
        6: r"FW$_{II,0}$",
        7: r"FW$_{II,1}$",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Segments Overview")
    ax.set_xlabel("omega")
    ax.set_ylabel("torque")

    for seg_val in data_obj.unique_segments:
        mask = data_obj.segments == seg_val

        color = colors_map.get(seg_val, "black")
        label_name = labels_map.get(seg_val, str(seg_val))

        ax.scatter(
            data_obj.omega_grid[mask],
            data_obj.torq_grid[mask],
            s=10,
            c=color,
            marker=".",
            label=label_name,
        )

    # Prevent duplicate legend entries if segments are plotted in fragments
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))

    ax.legend(unique_legend.values(), unique_legend.keys(), loc="best")
    ax.grid(True)
    plt.show()


def plot_global_performance(df: Optional[pd.DataFrame]) -> None:
    """
    Generates a comprehensive performance comparison between Analytical and
    Neural torque models, including linearity, residuals, and error heatmaps.

    Args:
        df: DataFrame containing 'torq_meas', 'torq_model', 'torq_neural',
            'Error_Analytical', 'Error_Neural', 'omega', and 'I_total'.
    """
    if df is None:
        return

    rmse_analytical = np.sqrt(np.mean(df["Error_Analytical"] ** 2))
    rmse_neural = np.sqrt(np.mean(df["Error_Neural"] ** 2))

    # FIG 1: Linearity
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    min_val = min(df["torq_meas"].min(), df["torq_model"].min())
    max_val = max(df["torq_meas"].max(), df["torq_model"].max())

    def plot_fit(ax: Axes, y_pred: pd.Series, title: str, color: str) -> None:
        """Helper to plot predicted vs measured torque scatter."""
        ax.scatter(df["torq_meas"], y_pred, alpha=0.5, s=10, c=color)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        ax.set_title(title)
        ax.set_xlabel("Measured Torque [Nm]")
        ax.grid(True, alpha=0.3)

    plot_fit(
        axes1[0], df["torq_model"], f"Analytical (RMSE={rmse_analytical:.3f})", "blue"
    )
    plot_fit(axes1[1], df["torq_neural"], f"Neural (RMSE={rmse_neural:.3f})", "green")
    axes1[0].set_ylabel("Predicted Torque [Nm]")
    plt.tight_layout()

    # FIG 2: Residuals vs Speed
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    sc2 = axes2[0].scatter(
        df["omega"],
        df["Error_Analytical"],
        c=df["I_total"],
        cmap="viridis",
        s=15,
        alpha=0.7,
    )
    axes2[0].axhline(0, color="r", linestyle="--")
    axes2[0].set_title("Analytical Residuals")

    axes2[1].scatter(
        df["omega"],
        df["Error_Neural"],
        c=df["I_total"],
        cmap="viridis",
        s=15,
        alpha=0.7,
    )
    axes2[1].axhline(0, color="r", linestyle="--")
    axes2[1].set_title("Neural Residuals")

    fig2.colorbar(sc2, ax=axes2.ravel().tolist(), label="Current Magnitude [A]")
    fig2.suptitle("Error vs Speed (Colored by Current)")

    # FIG 3: Heatmap
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    v_lim = df["Error_Analytical"].abs().quantile(0.98)

    sc3 = axes3[0].scatter(
        df["omega"],
        df["torq_meas"],
        c=df["Error_Analytical"],
        cmap="seismic",
        s=30,
        vmin=-v_lim,
        vmax=v_lim,
    )
    axes3[0].set_title("Analytical Signed Error")

    axes3[1].scatter(
        df["omega"],
        df["torq_meas"],
        c=df["Error_Neural"],
        cmap="seismic",
        s=30,
        vmin=-v_lim,
        vmax=v_lim,
    )
    axes3[1].set_title("Neural Signed Error")

    fig3.colorbar(sc3, ax=axes3.ravel().tolist(), label="Error Magnitude [Nm]")
    plt.show()

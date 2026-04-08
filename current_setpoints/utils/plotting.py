import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from typing import Any, Optional, List


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

        # Dynamic fallback for N-phase machines generating higher segment IDs
        color = colors_map.get(seg_val, np.random.rand(3,))
        label_name = labels_map.get(seg_val, f"Segment_{seg_val}")

        ax.scatter(
            data_obj.omega_grid[mask],
            data_obj.torq_grid[mask],
            s=10,
            c=[color] * mask.sum(), # Enforce color format to avoid Matplotlib warnings
            marker=".",
            label=label_name,
        )

    # Prevent duplicate legend entries if segments are plotted in fragments
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))

    ax.legend(unique_legend.values(), unique_legend.keys(), loc="best")
    ax.grid(True)
    plt.show()


def plot_global_performance(
    df: Optional[pd.DataFrame], models: Optional[List[str]] = None
) -> None:
    """
    Generates a comprehensive performance comparison for any number of torque models,
    including linearity, residuals, and error heatmaps.

    Args:
        df: DataFrame containing 'torq_meas', 'omega', 'I_total', and 
            model-specific columns (e.g., 'torq_Analytical', 'Error_Analytical').
        models: Optional list of model names to plot. If None, auto-detects based on 'Error_' columns.
    """
    if df is None or df.empty:
        print("Provided DataFrame is empty or None.")
        return

    # Auto-detect models based on columns starting with "Error_"
    if models is None:
        models = [col.replace("Error_", "") for col in df.columns if col.startswith("Error_")]

    if not models:
        print("No model error columns found (expected format: 'Error_ModelName').")
        return

    n_models = len(models)

    # --- FIG 1: Linearity ---
    fig1, axes1 = plt.subplots(1, n_models, figsize=(7 * n_models, 5), sharey=True)
    # Ensure axes1 is always iterable even if n_models == 1
    if n_models == 1:
        axes1 = [axes1]

    # Calculate global min/max for the diagonal reference line
    min_val = df["torq_meas"].min()
    max_val = df["torq_meas"].max()

    def plot_fit(ax: Axes, y_pred: pd.Series, title: str) -> None:
        """Helper to plot predicted vs measured torque scatter."""
        ax.scatter(df["torq_meas"], y_pred, alpha=0.5, s=10)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        ax.set_title(title)
        ax.set_xlabel("Measured Torque [Nm]")
        ax.grid(True, alpha=0.3)

    for i, model in enumerate(models):
        error_col = f"Error_{model}"
        
        # Support legacy naming ('torq_model') or standard naming ('torq_Analytical')
        pred_col = f"torq_{model}"
        if pred_col not in df.columns and model == "Analytical" and "torq_model" in df.columns:
            pred_col = "torq_model"
            
        if pred_col in df.columns:
            rmse = np.sqrt(np.mean(df[error_col] ** 2))
            plot_fit(axes1[i], df[pred_col], f"{model} (RMSE={rmse:.3f})")
            
            # Update global min/max for the diagonal line based on predictions
            min_val = min(min_val, df[pred_col].min())
            max_val = max(max_val, df[pred_col].max())

    axes1[0].set_ylabel("Predicted Torque [Nm]")
    plt.tight_layout()

    # --- FIG 2: Residuals vs Speed ---
    fig2, axes2 = plt.subplots(1, n_models, figsize=(7 * n_models, 5), sharey=True)
    if n_models == 1:
        axes2 = [axes2]

    sc2 = None
    for i, model in enumerate(models):
        sc2 = axes2[i].scatter(
            df["omega"],
            df[f"Error_{model}"],
            c=df["I_total"],
            cmap="viridis",
            s=15,
            alpha=0.7,
        )
        axes2[i].axhline(0, color="r", linestyle="--")
        axes2[i].set_title(f"{model} Residuals")
        axes2[i].set_xlabel("Speed [rad/s]")

    axes2[0].set_ylabel("Error [Nm]")
    
    # Explicit 'is not None' check to satisfy Pylance typing
    if sc2 is not None:
        fig2.colorbar(sc2, ax=axes2, label="Current Magnitude [A]")
        
    fig2.suptitle("Error vs Speed (Colored by Current)")

    # --- FIG 3: Heatmap ---
    fig3, axes3 = plt.subplots(1, n_models, figsize=(7 * n_models, 5), sharey=True)
    if n_models == 1:
        axes3 = [axes3]

    # Calculate global symmetric color limit based on the worst 98th percentile error across all models
    v_lim = max([df[f"Error_{m}"].abs().quantile(0.98) for m in models])

    sc3 = None
    for i, model in enumerate(models):
        sc3 = axes3[i].scatter(
            df["omega"],
            df["torq_meas"],
            c=df[f"Error_{model}"],
            cmap="seismic",
            s=30,
            vmin=-v_lim,
            vmax=v_lim,
        )
        axes3[i].set_title(f"{model} Signed Error")
        axes3[i].set_xlabel("Speed [rad/s]")

    axes3[0].set_ylabel("Measured Torque [Nm]")
    
    # Explicit 'is not None' check to satisfy Pylance typing
    if sc3 is not None:
        fig3.colorbar(sc3, ax=axes3, label="Error Magnitude [Nm]")
    
    plt.show()
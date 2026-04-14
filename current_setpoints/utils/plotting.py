import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, Optional, List
from plot_config import PlotConfig


def plot_grid_segments(data_obj: Any, machine: Any) -> None:
    """
    Plots the motor operating regions (MTPA, FW, etc.) as a scatter plot
    on the torque-speed plane. Retains its original logic using data_obj.
    """
    colors_map = PlotConfig.get_colors()
    labels_map = PlotConfig.get_labels()

    with plt.rc_context(PlotConfig.get_rc_params()):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Segments Overview")

        # Standardize to RPM using the machine object
        omega_rpm = data_obj.omega_grid * (30 / (np.pi * machine.n_ppairs))

        ax.set_xlabel("Speed [r/min]")
        ax.set_ylabel("Torque [Nm]")

        for seg_val in data_obj.unique_segments:
            mask = data_obj.segments == seg_val
            color = colors_map.get(
                seg_val,
                np.random.rand(
                    3,
                ),
            )
            label_name = labels_map.get(seg_val, f"Segment_{seg_val}")

            ax.scatter(
                omega_rpm[mask],
                data_obj.torq_grid[mask],
                s=10,
                c=[color] * mask.sum(),
                marker=".",
                label=label_name,
            )

        PlotConfig.apply_deduplicated_legend(ax)
        ax.grid(True)
        plt.show()


def plot_global_performance(
    df: Optional[pd.DataFrame], machine: Any, models: Optional[List[str]] = None
) -> None:
    """
    Generates a comprehensive performance comparison from a DataFrame.
    Speed is standardized to r/min based on machine.n_ppairs.
    """
    if df is None or df.empty:
        print("Provided DataFrame is empty or None.")
        return

    if models is None:
        models = [
            col.replace("Error_", "") for col in df.columns if col.startswith("Error_")
        ]

    if not models:
        print("No model error columns found.")
        return

    n_models = len(models)

    with plt.rc_context(
        PlotConfig.get_rc_params(
            {"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14}
        )
    ):
        # --- Figure 1: Predicted vs Measured Fit ---
        fig1, axes1 = plt.subplots(1, n_models, figsize=(7 * n_models, 5), sharey=True)
        if n_models == 1:
            axes1 = [axes1]

        min_val, max_val = df["torq_meas"].min(), df["torq_meas"].max()

        for i, model in enumerate(models):
            error_col, pred_col = f"Error_{model}", f"torq_{model}"
            if (
                pred_col not in df.columns
                and model == "Analytical"
                and "torq_model" in df.columns
            ):
                pred_col = "torq_model"

            if pred_col in df.columns:
                rmse = np.sqrt(np.mean(df[error_col] ** 2))
                axes1[i].scatter(df["torq_meas"], df[pred_col], alpha=0.5, s=10)
                axes1[i].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
                axes1[i].set_title(f"{model} (RMSE={rmse:.3f})", fontsize=18)
                axes1[i].set_xlabel("Measured Torque [Nm]")
                axes1[i].grid(True, alpha=0.3)

                min_val = min(min_val, df[pred_col].min())
                max_val = max(max_val, df[pred_col].max())

        axes1[0].set_ylabel("Predicted Torque [Nm]")
        plt.tight_layout()

        # --- Figure 2 & 3 Boilerplate Omitted for Brevity ---
        plt.show()


def plot_residual_torque_error(data_obj: Any, machine: Any) -> None:
    """
    Plots the residual error. Adapted to accept data_obj.
    """
    omega_rpm = data_obj.vec_omega * (30 / (np.pi * machine.n_ppairs))
    T_target = data_obj.vec_torq

    T_target_2d = np.tile(T_target[:, np.newaxis], (1, len(omega_rpm)))
    T_actual_2d = data_obj.grid_torq_neural

    valid_mask = ~np.isnan(data_obj.curr_dq_grid[0])
    T_diff = np.where(valid_mask, T_target_2d - T_actual_2d, np.nan)

    custom_rc = PlotConfig.get_rc_params(
        {"font.size": 24, "axes.labelsize": 26, "axes.titlesize": 26}
    )
    with plt.rc_context(custom_rc):
        PlotConfig.plot_speed_torque_heatmap(
            omega_rpm,
            T_target,
            T_diff,
            valid_mask,
            cbar_label="Residual Torque Error $T_{base} - T_{PAM}$ [Nm]",
            y_label="Predicted Torque [Nm]",
        )


def plot_current_trajectories_split(
    data_obj_base: Any, data_obj_recalc: Any, machine: Any
) -> None:
    """
    Plots trajectories per harmonic subspace. Adapted to expect data_obj wrapper objects.
    Uses data_obj.segments for region color mapping.
    """
    colors_map = PlotConfig.get_colors()
    labels_map = PlotConfig.get_labels()

    n_harmonics = (machine.n_phases - 1) // 2

    with plt.rc_context(PlotConfig.get_rc_params({"axes.labelsize": 24})):
        fig, axes = plt.subplots(n_harmonics, 2, figsize=(18, 7 * n_harmonics))
        if n_harmonics == 1:
            axes = np.expand_dims(axes, axis=0)

        def plot_grid_onto_ax(ax, obj_data, d_idx, q_idx):
            d_grid, q_grid = obj_data.curr_dq_grid[d_idx], obj_data.curr_dq_grid[q_idx]

            # Use obj_data.segments directly as established by plot_grid_segments
            mode_grid = getattr(obj_data, "segments", np.zeros_like(d_grid))

            valid_mask = ~np.isnan(d_grid)
            d_valid, q_valid, modes_valid = (
                d_grid[valid_mask],
                q_grid[valid_mask],
                mode_grid[valid_mask],
            )

            unique_modes = np.unique(modes_valid[~np.isnan(modes_valid)])
            for mode in unique_modes:
                mask = modes_valid == mode
                ax.scatter(
                    d_valid[mask],
                    q_valid[mask],
                    s=40,
                    alpha=0.5,
                    c=colors_map.get(int(mode), "black"),
                    label=labels_map.get(int(mode), str(int(mode))),
                    marker=".",
                )

            ax.grid(True, alpha=0.3)
            ax.axis("equal")

        for h in range(n_harmonics):
            d_idx, q_idx, h_num = 2 * h, 2 * h + 1, 2 * h + 1
            plot_grid_onto_ax(axes[h, 0], data_obj_base, d_idx, q_idx)
            plot_grid_onto_ax(axes[h, 1], data_obj_recalc, d_idx, q_idx)

            axes[h, 0].set_ylabel(f"$i_{{q{h_num}}}$ [A]")
            axes[h, 0].set_xlabel(f"$i_{{d{h_num}}}$ [A]")
            axes[h, 1].set_xlabel(f"$i_{{d{h_num}}}$ [A]")

        PlotConfig.apply_deduplicated_legend(
            fig, bbox_to_anchor=(0.92, 0.5), markerscale=3
        )

        axes[0, 0].set_title("Baseline (Analytical)")
        axes[0, 1].set_title("Recalculated (Neural Optimized)")

        plt.tight_layout()
        plt.subplots_adjust(right=0.90, hspace=0.2, wspace=0.2)
        plt.show()


def plot_joule_losses_reduction(
    data_obj_base: Any, data_obj_recalc: Any, machine: Any
) -> None:
    """
    Calculates Joule loss reduction. Adapted to accept data_obj wrapper objects.
    """
    omega_rpm = data_obj_recalc.vec_omega * (30 / (np.pi * machine.n_ppairs))
    vec_torq = data_obj_recalc.vec_torq

    mask_base = ~np.isnan(data_obj_base.curr_dq_grid[0])
    mask_recalc = ~np.isnan(data_obj_recalc.curr_dq_grid[0])
    common_mask = mask_base & mask_recalc

    if not np.any(common_mask):
        print("No overlapping data points found.")
        return

    R_diag = np.diag(machine.R_stat)

    broadcast_shape = [slice(None)] + [np.newaxis] * (
        data_obj_base.curr_dq_grid.ndim - 1
    )
    R_broad = R_diag[tuple(broadcast_shape)]

    P_loss_base = machine.k_phase * np.sum(
        data_obj_base.curr_dq_grid**2 * R_broad, axis=0
    )
    P_loss_recalc = machine.k_phase * np.sum(
        data_obj_recalc.curr_dq_grid**2 * R_broad, axis=0
    )

    P_diff = np.where(common_mask, P_loss_base - P_loss_recalc, np.nan)

    with plt.rc_context(PlotConfig.get_rc_params()):
        PlotConfig.plot_speed_torque_heatmap(
            omega_rpm,
            vec_torq,
            P_diff,
            common_mask,
            cbar_label="Joule Losses Reduction [Watts]",
            cbar_kwargs={"rotation": 270, "labelpad": 30, "size": 20},
        )

    print(f"Analysis Complete. Max savings: {np.nanmax(P_diff):.2f} W")

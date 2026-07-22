from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from ..models.machines import DriveModel
from ..optimization.data import MachineData


# ─────────────────────────────────────────────────────────────────────────────
# PlotConfig
# ─────────────────────────────────────────────────────────────────────────────

class PlotConfig:

    @classmethod
    def get_colors(cls) -> dict[int, str]:
        return {
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

    @classmethod
    def get_labels(cls) -> dict[int, str]:
        # index = 3*n_volt + n_curr (grid.py's grid_segments), n_volt/n_curr
        # each capped at 2 by count_peaks_at_limit -- 9 combinations, 0-8.
        return {
            0: r"MTPA$_{0}$",
            1: r"MTPA$_{1}$",
            2: r"MTPA$_{2}$",
            3: r"FW$_{I,0}$",
            4: r"FW$_{I,1}$",
            5: r"FW$_{I,2}$",
            6: r"FW$_{II,0}$",
            7: r"FW$_{II,1}$",
            8: r"FW$_{II,2}$",
        }

    @classmethod
    def get_rc_params(cls, custom_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        base = {
            "font.size": 20,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "legend.title_fontsize": 22,
        }
        if custom_overrides:
            base.update(custom_overrides)
        return base

    @classmethod
    def apply_deduplicated_legend(
        cls,
        figure_or_ax: Figure | Axes | np.ndarray,
        loc: str = "best",
        **kwargs: Any,
    ) -> None:
        if isinstance(figure_or_ax, Figure):
            axes_list: list[Axes] = list(figure_or_ax.axes)
            legend_target: Figure | Axes = figure_or_ax
        elif isinstance(figure_or_ax, Axes):
            axes_list = [figure_or_ax]
            legend_target = figure_or_ax
        elif isinstance(figure_or_ax, np.ndarray):
            axes_list = [ax for ax in figure_or_ax.flatten() if isinstance(ax, Axes)]
            if not axes_list:
                return
            legend_target = axes_list[0]
        else:
            raise TypeError(
                f"figure_or_ax must be a Figure, Axes, or ndarray of Axes; "
                f"got {type(figure_or_ax).__name__}."
            )

        by_label: dict[str, Any] = {}
        for ax in axes_list:
            handles, labels = ax.get_legend_handles_labels()
            for h, lbl in zip(handles, labels, strict=True):
                if lbl not in by_label:
                    by_label[lbl] = h

        labels_map = cls.get_labels()
        label_to_idx = {v: k for k, v in labels_map.items()}
        sorted_labels = sorted(by_label, key=lambda lbl: label_to_idx.get(lbl, 999))
        sorted_handles = [by_label[lbl] for lbl in sorted_labels]
        legend_target.legend(sorted_handles, sorted_labels, loc=loc, **kwargs)

    @classmethod
    def plot_speed_torque_heatmap(
        cls,
        omega_rpm: np.ndarray,
        vec_torq: np.ndarray,
        z_grid: np.ndarray,
        mask: np.ndarray,
        cbar_label: str,
        y_label: str = "Torque [Nm]",
        cbar_kwargs: dict[str, Any] | None = None,
    ) -> None:
        omega_2d, torq_2d = np.meshgrid(omega_rpm, vec_torq)
        fig, ax = plt.subplots(figsize=(14, 9))
        sc = ax.scatter(
            omega_2d[mask], torq_2d[mask], c=z_grid[mask],
            cmap="Reds", s=40, marker="s", linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax)
        if cbar_kwargs:
            cbar.set_label(cbar_label, **cbar_kwargs)
        else:
            cbar.set_label(cbar_label, size=24, labelpad=20)
        ax.set_xlabel("Speed [r/min]", labelpad=10)
        ax.set_ylabel(y_label, labelpad=10)
        ax.grid(True, which="both", linestyle="-", alpha=0.4)
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_COLORS: dict[str, str] = {"Analytical": "tab:blue", "Neural": "tab:green"}
_DEFAULT_COLOR: str = "tab:gray"


def plot_grid_segments(
    data_obj: MachineData,
    title: str = "Segments Overview",
    override_torq_2d: np.ndarray | None = None,
) -> None:
    colors_map = PlotConfig.get_colors()
    labels_map = PlotConfig.get_labels()

    with plt.rc_context(PlotConfig.get_rc_params()):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(title)
        ax.set_xlabel("Speed [r/min]")
        ax.set_ylabel("Torque [Nm]")

        omega_2d, default_torq_2d = np.meshgrid(data_obj.omega, data_obj.torq)
        torq_2d = override_torq_2d if override_torq_2d is not None else default_torq_2d

        modes = data_obj.segments
        valid_mask = ~np.isnan(modes)
        omega_v = omega_2d[valid_mask]
        torq_v = torq_2d[valid_mask]
        modes_v = modes[valid_mask]

        for seg_val in data_obj.unique_segments:
            mask = modes_v == seg_val
            ax.scatter(
                omega_v[mask], torq_v[mask],
                s=10, c=[colors_map.get(seg_val, "black")] * mask.sum(),
                marker=".", label=labels_map.get(seg_val, f"Segment_{seg_val}"),
            )

        PlotConfig.apply_deduplicated_legend(ax)
        ax.grid(True)
        plt.show()


def plot_baseline_vs_compensated(
    baseline_data: MachineData,
    compensated_data: MachineData,
    override_torq_2d_top: np.ndarray | None = None,
    override_torq_2d_bottom: np.ndarray | None = None,
    figsize: tuple[float, float] = (12.0, 18.5),
    marker_size: float = 40.0,
) -> None:
    colors_map = PlotConfig.get_colors()
    labels_map = PlotConfig.get_labels()

    def _plot(ax: Axes, data: MachineData, override: np.ndarray | None) -> None:
        omega_2d, default_torq_2d = np.meshgrid(data.omega, data.torq)
        torq_2d = override if override is not None else default_torq_2d
        modes = data.segments
        valid_mask = ~np.isnan(modes)
        omega_v = omega_2d[valid_mask]
        torq_v = torq_2d[valid_mask]
        modes_v = modes[valid_mask]
        for seg_val in data.unique_segments:
            mask = modes_v == seg_val
            ax.scatter(
                omega_v[mask], torq_v[mask],
                s=marker_size, c=[colors_map.get(seg_val, "black")] * mask.sum(),
                marker=".", label=labels_map.get(seg_val, f"Segment_{seg_val}"),
            )
        ax.set_ylabel("Torque [Nm]")
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlim(float(data.omega.min()), float(data.omega.max()))
        ax.set_ylim(float(data.torq.min()), float(data.torq.max()))
        ax.grid(True)

    with plt.rc_context(PlotConfig.get_rc_params()):
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)
        _plot(axes[0], baseline_data, override_torq_2d_top)
        _plot(axes[1], compensated_data, override_torq_2d_bottom)
        axes[1].set_xlabel("Speed [r/min]")
        fig.subplots_adjust(hspace=0.08, right=0.86)
        PlotConfig.apply_deduplicated_legend(
            fig, loc="center left", bbox_to_anchor=(0.88, 0.5), markerscale=3,
        )
        plt.show()


def plot_global_performance(
    df: pd.DataFrame | None,
    models: list[str] | None = None,
    test_mask: np.ndarray | None = None,
) -> None:
    if df is None or df.empty:
        print("Provided DataFrame is empty or None.")
        return

    if models is None:
        models = [col.replace("Error_", "") for col in df.columns if col.startswith("Error_")]
    if not models:
        print("No model error columns found.")
        return

    if test_mask is not None:
        test_mask = np.asarray(test_mask, dtype=bool)
        if test_mask.shape != (len(df),):
            raise ValueError(
                f"test_mask shape {test_mask.shape} does not match DataFrame length ({len(df)})."
            )

    n_models = len(models)
    with plt.rc_context(PlotConfig.get_rc_params({"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})):
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), sharey=True)
        if n_models == 1:
            axes = [axes]

        min_val, max_val = df["torq_meas"].min(), df["torq_meas"].max()

        for i, model in enumerate(models):
            error_col = f"Error_{model}"
            pred_col = f"torq_{model}"
            if pred_col not in df.columns and model == "Analytical" and "torq_model" in df.columns:
                pred_col = "torq_model"
            if pred_col not in df.columns:
                continue

            color = _MODEL_COLORS.get(model, _DEFAULT_COLOR)

            if test_mask is None:
                rmse = np.sqrt(np.mean(df[error_col] ** 2))
                axes[i].scatter(df["torq_meas"], df[pred_col], alpha=0.5, s=10, color=color)
                axes[i].set_title(f"{model} (RMSE={rmse:.3f})", fontsize=18)
            else:
                rmse_test = np.sqrt(np.mean(df.loc[test_mask, error_col] ** 2))
                axes[i].scatter(
                    df.loc[~test_mask, "torq_meas"], df.loc[~test_mask, pred_col],
                    s=30, facecolors="none", edgecolors=color, linewidths=0.8,
                    alpha=0.6, label="Train",
                )
                axes[i].scatter(
                    df.loc[test_mask, "torq_meas"], df.loc[test_mask, pred_col],
                    s=30, color=color, label=f"Test (RMSE: {rmse_test:.3f})",
                )
                axes[i].legend(loc="upper left", fontsize=14)

            axes[i].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
            axes[i].set_xlabel("Measured Torque [Nm]")
            axes[i].grid(True, alpha=0.3)
            min_val = min(min_val, df[pred_col].min())
            max_val = max(max_val, df[pred_col].max())

        axes[0].set_ylabel("Predicted Torque [Nm]")
        plt.tight_layout()
        plt.show()


def plot_residual_torque_error(
    dict_grid: dict[str, Any],
    drive: DriveModel,
) -> None:
    const_mech_speed = 30.0 / (np.pi * drive.n_ppairs)
    omega_rpm = dict_grid["vec_omega"] * const_mech_speed
    T_target = dict_grid["vec_torq"]
    T_actual_2d = dict_grid["grid_torq_neural"]
    T_target_2d = np.tile(T_target[:, np.newaxis], (1, len(omega_rpm)))

    valid_mask = ~np.isnan(dict_grid["curr_dq_grid"][0]) & ~np.isnan(T_actual_2d)
    T_diff = T_target_2d - T_actual_2d
    omega_2d, _ = np.meshgrid(omega_rpm, T_target)

    with plt.rc_context(PlotConfig.get_rc_params({"font.size": 24, "axes.labelsize": 26, "axes.titlesize": 26})):
        fig, ax = plt.subplots(figsize=(14, 9))
        sc = ax.scatter(
            omega_2d[valid_mask], T_actual_2d[valid_mask], c=T_diff[valid_mask],
            cmap="Reds", s=40, marker="s", linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"Residual Torque Error $T_{base} - T_{NTM}$ [Nm]", size=24, labelpad=20)
        ax.set_xlabel("Speed [r/min]", labelpad=10)
        ax.set_ylabel("Torque [Nm]", labelpad=10)
        ax.grid(True, which="both", linestyle="-", alpha=0.4)
        ax.set_xlim(0.0, float(omega_rpm.max()))
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()


def plot_current_trajectories_split(
    dict_grid_base: dict[str, Any],
    dict_grid_recalc: dict[str, Any],
    drive: DriveModel,
) -> None:
    colors_map = PlotConfig.get_colors()
    labels_map = PlotConfig.get_labels()
    n_harmonics = drive.dim // 2

    with plt.rc_context(PlotConfig.get_rc_params(
        {"axes.labelsize": 28, "xtick.labelsize": 24, "ytick.labelsize": 24,
         "legend.fontsize": 24, "legend.title_fontsize": 26}
    )):
        fig, axes = plt.subplots(n_harmonics, 2, figsize=(18, 7 * n_harmonics))
        if n_harmonics == 1:
            axes = np.expand_dims(axes, axis=0)

        def _plot_onto(ax: Axes, dict_grid: dict[str, Any], d_idx: int, q_idx: int) -> None:
            d_grid = dict_grid["curr_dq_grid"][d_idx]
            q_grid = dict_grid["curr_dq_grid"][q_idx]
            mode_grid = dict_grid.get("grid_segments", np.zeros_like(d_grid))
            valid_mask = ~np.isnan(d_grid)
            d_v, q_v = d_grid[valid_mask], q_grid[valid_mask]
            modes_v = mode_grid[valid_mask]
            for mode in np.unique(modes_v[~np.isnan(modes_v)]):
                mask = modes_v == mode
                ax.scatter(
                    d_v[mask], q_v[mask], s=40, alpha=0.5,
                    c=colors_map.get(int(mode), "black"),
                    label=labels_map.get(int(mode), str(int(mode))),
                    marker=".",
                )
            ax.grid(True, alpha=0.3)
            ax.axis("equal")

        for h in range(n_harmonics):
            d_idx, q_idx, h_num = 2 * h, 2 * h + 1, 2 * h + 1
            _plot_onto(axes[h, 0], dict_grid_base, d_idx, q_idx)
            _plot_onto(axes[h, 1], dict_grid_recalc, d_idx, q_idx)
            axes[h, 0].set_ylabel(f"$i_{{q{h_num}}}$ [A]")
            axes[h, 0].set_xlabel(f"$i_{{d{h_num}}}$ [A]")
            axes[h, 1].set_xlabel(f"$i_{{d{h_num}}}$ [A]")

        plt.tight_layout()
        plt.subplots_adjust(right=0.85, hspace=0.2, wspace=0.2)
        PlotConfig.apply_deduplicated_legend(
            fig, loc="center left", bbox_to_anchor=(0.86, 0.5), markerscale=3,
        )
        plt.show()


def plot_joule_losses_reduction(
    dict_grid_base: dict[str, Any],
    dict_grid_recalc: dict[str, Any],
    drive: DriveModel,
) -> None:
    const_mech_speed = 30.0 / (np.pi * drive.n_ppairs)
    omega_rpm = dict_grid_recalc["vec_omega"] * const_mech_speed
    vec_torq = dict_grid_recalc["vec_torq"]
    T_actual_2d = dict_grid_recalc["grid_torq_neural"]

    mask_base = ~np.isnan(dict_grid_base["curr_dq_grid"][0])
    mask_recalc = ~np.isnan(dict_grid_recalc["curr_dq_grid"][0])
    common_mask = mask_base & mask_recalc & ~np.isnan(T_actual_2d)

    if not np.any(common_mask):
        print("No overlapping data points found.")
        return

    R_diag = np.diag(drive.R_stat)
    broadcast_shape = [slice(None)] + [np.newaxis] * (dict_grid_base["curr_dq_grid"].ndim - 1)
    R_broad = R_diag[tuple(broadcast_shape)]

    P_base = drive.k_phase * np.sum(dict_grid_base["curr_dq_grid"] ** 2 * R_broad, axis=0)
    P_recalc = drive.k_phase * np.sum(dict_grid_recalc["curr_dq_grid"] ** 2 * R_broad, axis=0)
    P_diff = P_base - P_recalc

    omega_2d, _ = np.meshgrid(omega_rpm, vec_torq)

    with plt.rc_context(PlotConfig.get_rc_params()):
        fig, ax = plt.subplots(figsize=(14, 9))
        sc = ax.scatter(
            omega_2d[common_mask], T_actual_2d[common_mask], c=P_diff[common_mask],
            cmap="Reds", s=40, marker="s", linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(
            r"Joule Losses Reduction $P_{j,base} - P_{j,eval}$ [W]", size=24, labelpad=20,
        )
        ax.set_xlabel("Speed [r/min]", labelpad=10)
        ax.set_ylabel("Torque [Nm]", labelpad=10)
        ax.grid(True, which="both", linestyle="-", alpha=0.4)
        ax.set_xlim(0.0, float(omega_rpm.max()))
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

    print(f"Analysis Complete. Max savings: {np.nanmax(P_diff[common_mask]):.2f} W")

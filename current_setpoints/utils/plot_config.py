from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


class PlotConfig:
    """
    Centralized configuration and plotting helper class.
    Houses common styles, colors, labels, and shared boilerplate logic.
    """

    @classmethod
    def get_colors(cls) -> Dict[int, str]:
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
    def get_labels(cls) -> Dict[int, str]:
        return {
            0: r"MTPA$_{0}$",
            1: r"MTPA$_{1}$",
            2: r"MTPA$_{2}$",
            3: r"FW$_{I,0}$",
            4: r"FW$_{I,1}$",
            5: r"FW$_{I,2}$",
            6: r"FW$_{II,0}$",
            7: r"FW$_{II,1}$",
        }

    @classmethod
    def get_rc_params(
        cls, custom_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        base_params = {
            "font.size": 20,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "legend.title_fontsize": 22,
        }
        if custom_overrides:
            base_params.update(custom_overrides)
        return base_params

    @classmethod
    def apply_deduplicated_legend(
        cls, figure_or_ax: Any, loc: str = "best", **kwargs
    ) -> None:
        by_label = {}

        if hasattr(figure_or_ax, "get_axes") and callable(figure_or_ax.get_axes):
            axes_list = figure_or_ax.axes
        else:
            # Safely force single Axes or n-dimensional arrays of Axes into a flat 1D iterable
            axes_list = np.atleast_1d(figure_or_ax).flatten()

        for ax in axes_list:
            handles, labels = ax.get_legend_handles_labels()
            for h, lbl in zip(handles, labels):
                if lbl not in by_label:
                    by_label[lbl] = h

        labels_map = cls.get_labels()

        def get_sort_key(lbl: str) -> int:
            for k, v in labels_map.items():
                if v == lbl:
                    return k
            return 999

        sorted_labels = sorted(by_label.keys(), key=get_sort_key)
        sorted_handles = [by_label[lbl] for lbl in sorted_labels]

        target = figure_or_ax if hasattr(figure_or_ax, "legend") else axes_list[0]
        target.legend(sorted_handles, sorted_labels, loc=loc, **kwargs)

    @classmethod
    def plot_speed_torque_heatmap(
        cls,
        omega_rpm: np.ndarray,
        vec_torq: np.ndarray,
        z_grid: np.ndarray,
        mask: np.ndarray,
        cbar_label: str,
        y_label: str = "Torque [Nm]",
        cbar_kwargs: Optional[Dict] = None,
    ) -> None:
        omega_2d, torq_2d = np.meshgrid(omega_rpm, vec_torq)

        x_vals = omega_2d[mask]
        y_vals = torq_2d[mask]
        c_vals = z_grid[mask]

        fig, ax = plt.subplots(figsize=(14, 9))
        sc = ax.scatter(
            x_vals, y_vals, c=c_vals, cmap="Reds", s=40, marker="s", linewidths=0
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

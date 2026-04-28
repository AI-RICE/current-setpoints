from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class PlotConfig:
    """
    Centralized configuration and plotting helper class.
    Houses common styles, colors, labels, and shared boilerplate logic.
    """

    @classmethod
    def get_colors(cls) -> dict[int, str]:
        """Returns the color palette mapping segment index to hex color."""
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
        """Returns the label mapping for operating regime segment indices."""
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
        cls, custom_overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Returns a default matplotlib rcParams dict, optionally merged with caller overrides.

        Args:
            custom_overrides: Optional rcParams entries that take precedence over defaults.
        """
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
        cls, figure_or_ax: Figure | Axes | np.ndarray, loc: str = "best", **kwargs: Any
    ) -> None:
        """
        Builds a single legend on the figure (or single axes) with each label
        appearing at most once, and orders entries to match
        ``PlotConfig.get_labels`` so segment regimes appear in canonical order.

        Args:
            figure_or_ax: A Figure, an Axes, or an ndarray of Axes.
            loc: Legend location, forwarded to matplotlib's ``legend``.
            **kwargs: Additional keyword arguments forwarded to ``legend``.
        """
        legend_target: Figure | Axes
        axes_list: list[Axes]

        if isinstance(figure_or_ax, Figure):
            axes_list = list(figure_or_ax.axes)
            legend_target = figure_or_ax
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
        # Reverse map for O(1) sort key lookup (was O(n) per call inside sorted()).
        label_to_idx = {v: k for k, v in labels_map.items()}

        sorted_labels = sorted(
            by_label.keys(), key=lambda lbl: label_to_idx.get(lbl, 999)
        )
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
        """
        Renders a scatter heatmap of a scalar field over the (speed, torque) plane,
        masking entries where ``mask`` is False.

        Args:
            omega_rpm: Speed axis values [r/min].
            vec_torq: Torque axis values [Nm].
            z_grid: 2D field to colorize, shape (n_torq, n_omega).
            mask: 2D boolean mask of points to plot, same shape as z_grid.
            cbar_label: Colorbar label string.
            y_label: Y-axis label.
            cbar_kwargs: Optional kwargs forwarded to ``cbar.set_label``.
        """
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

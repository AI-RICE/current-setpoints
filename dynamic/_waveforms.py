"""
Dense-grid dq-trajectory reconstruction and the combined dq/phase-current/
phase-voltage summary plot used by the numbered dynamic/ demo scripts and
several experiments/ scripts.

Ported from the pre-refactor ``current_setpoints.utils.plotting_dynamic``
(deleted in 500edad). Simplified: the old ``_phase_currents_all`` rebuilt
every phase by rolling a phase-A waveform because ``Transform`` only stored
phase-A's dq-to-phase matrix; ``ForwardModel`` stores every phase's matrix
directly (``_mat_dq_to_ph_all``), so no rolling reconstruction is needed --
this returns all phases in one shot instead of one phase to be rolled.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from current_setpoints.models.forward_model import ForwardModel
from current_setpoints.utils.plotting import PlotConfig


def evaluate_dq_on_grid(
    theta_grid: np.ndarray,
    curr_dq_grid: np.ndarray,
    theta_dense: np.ndarray,
) -> np.ndarray:
    """
    Periodic linear interpolation of a per-node dq trajectory onto a dense
    angular grid. ``theta_grid`` need not include the periodic wrap-point;
    the function adds it internally.

    Args:
        theta_grid: shape ``(M,)`` in ``[0, 2*pi)``.
        curr_dq_grid: shape ``(M, dim)``.
        theta_dense: shape ``(P,)`` -- target evaluation angles.

    Returns:
        Array of shape ``(P, dim)``.
    """
    theta_aug = np.concatenate([theta_grid, [theta_grid[0] + 2 * np.pi]])
    curr_aug = np.vstack([curr_dq_grid, curr_dq_grid[:1]])
    out = np.empty((theta_dense.size, curr_dq_grid.shape[1]))
    for j in range(curr_dq_grid.shape[1]):
        out[:, j] = np.interp(theta_dense, theta_aug, curr_aug[:, j], period=2 * np.pi)
    return out


def dynamic_phase_waveforms(
    fwd: ForwardModel,
    omega: float,
    theta_grid: np.ndarray,
    curr_dq_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes all-phase current and voltage waveforms from a per-node dq
    trajectory. The voltage includes the inductive derivative term that the
    static formulation omits:

        v(theta) = P(theta) @ ( U @ curr_dq(theta) + bemf_dq(theta)
                     + omega * L_stat @ d curr_dq / d theta )

    Args:
        fwd: configured ForwardModel (drive's L is assumed curr-independent
            over the trajectory, matching the original Transform-based
            reconstruction, which fixed U/L at a single omega).
        omega: electrical speed [rad/s].
        theta_grid: shape ``(M,)`` node angles in ``[0, 2*pi)``.
        curr_dq_grid: shape ``(M, dim)``.

    Returns:
        ``(curr_ph_all, volt_ph_all)`` each of shape ``(n_phases, n_theta + 1)``.
    """
    drive = fwd.drive
    dim = drive.dim
    theta_dense = fwd.vec_theta
    curr_dense = evaluate_dq_on_grid(theta_grid, curr_dq_grid, theta_dense)  # (n_t, dim)

    mean_curr = curr_dq_grid.mean(axis=0)
    U = drive.voltage_operator(omega, np.zeros(dim))
    L = drive.inductance(omega, np.zeros(dim))
    bemf_dq = drive.bemf_dq(omega, mean_curr)

    static_dq = curr_dense @ U.T + bemf_dq[None, :]

    dtheta = theta_dense[1] - theta_dense[0]
    dxdtheta = np.gradient(curr_dense, dtheta, axis=0)
    inductive_dq = omega * dxdtheta @ L.T

    volt_dq_total = static_dq + inductive_dq

    P = fwd._mat_dq_to_ph_all  # (n_t, n_phases, dim)
    curr_ph_all = np.einsum("tpj,tj->pt", P, curr_dense)
    volt_ph_all = np.einsum("tpj,tj->pt", P, volt_dq_total)
    return curr_ph_all, volt_ph_all


def _harmonic_dq_labels(dim: int) -> list[str]:
    """Returns ``[i_{d1}, i_{q1}, i_{d3}, i_{q3}, ...]`` for the given dim."""
    labels: list[str] = []
    for i in range(dim // 2):
        h = 2 * i + 1
        labels.append(rf"$i_{{d^{h}}}$")
        labels.append(rf"$i_{{q^{h}}}$")
    return labels


def plot_dq_phase_combined(
    fwd: ForwardModel,
    omega: float,
    curr_dq: np.ndarray,
    *,
    theta_dq: np.ndarray | None = None,
    curr_dq_static: np.ndarray | None = None,
    curr_max: float | None = None,
    volt_max: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 11),
) -> tuple[Figure, np.ndarray]:
    """
    Plots a three-panel summary of one electrical period:

      (a) dq currents vs theta (lines, with optional dashed static overlay),
      (b) phase currents vs theta with optional +/- ``curr_max`` guides,
      (c) phase voltages vs theta with optional +/- ``volt_max`` guides.

    Args:
        fwd: configured ForwardModel.
        omega: electrical speed [rad/s] at which to evaluate phase waveforms.
        curr_dq: either a 1D vector of length ``fwd.drive.dim`` (static) or
            a 2D array of shape ``(M, fwd.drive.dim)`` (dynamic, per-node).
        theta_dq: angles for the dynamic case; defaults to
            ``linspace(0, 2*pi, M, endpoint=False)``. Ignored when ``curr_dq``
            is 1D.
        curr_dq_static: optional 1D reference overlaid on the dq panel as
            dashed horizontal lines. Useful in the dynamic case for showing
            departure from the static MTPA solution.
        curr_max, volt_max: optional limits, drawn as dashed horizontal lines.
        title: optional figure suptitle.
        figsize: figure size.

    Returns:
        ``(fig, axes)``: the created Figure and an array of the three Axes.
    """
    curr_dq = np.asarray(curr_dq)
    is_dynamic = curr_dq.ndim == 2
    dim = fwd.drive.dim
    n_phases = fwd.drive.n_phases

    if is_dynamic:
        if theta_dq is None:
            theta_dq = np.linspace(0.0, 2 * np.pi, curr_dq.shape[0], endpoint=False)
        curr_ph_all, volt_ph_all = dynamic_phase_waveforms(fwd, omega, theta_dq, curr_dq)
    else:
        if curr_dq.shape[0] != dim:
            raise ValueError(f"static curr_dq must have length {dim}, got {curr_dq.shape[0]}.")
        curr_ph_all = fwd.curr_ph(omega, curr_dq)
        volt_ph_all, _, _ = fwd.volt_ph(omega, curr_dq)

    theta_dense = fwd.vec_theta
    dq_labels = _harmonic_dq_labels(dim)
    rc = PlotConfig.get_rc_params({"axes.labelsize": 18, "axes.titlesize": 18, "legend.fontsize": 14})

    with plt.rc_context(rc):
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        ax_dq, ax_i, ax_v = axes

        cmap_dq = plt.get_cmap("tab10")
        if is_dynamic:
            for j, lbl in enumerate(dq_labels):
                ax_dq.plot(theta_dq, curr_dq[:, j], color=cmap_dq(j), lw=2.0, label=lbl)
        else:
            for j, lbl in enumerate(dq_labels):
                ax_dq.axhline(curr_dq[j], color=cmap_dq(j), lw=2.0, label=lbl)
        if curr_dq_static is not None:
            for j in range(dim):
                ax_dq.axhline(curr_dq_static[j], color=cmap_dq(j), lw=1.0, ls="--", alpha=0.7)
        ax_dq.set_ylabel("dq current [A]")
        ax_dq.grid(True, alpha=0.3)
        ax_dq.legend(ncols=max(1, dim // 2), loc="upper right")
        if title is not None:
            ax_dq.set_title(title)

        cmap_ph = plt.get_cmap("tab10")
        for k in range(n_phases):
            ax_i.plot(theta_dense, curr_ph_all[k], color=cmap_ph(k), lw=1.5, label=f"phase {k}")
        if curr_max is not None:
            ax_i.axhline(curr_max, color="k", lw=1.0, ls="--", alpha=0.6)
            ax_i.axhline(-curr_max, color="k", lw=1.0, ls="--", alpha=0.6)
        ax_i.set_ylabel("phase current [A]")
        ax_i.grid(True, alpha=0.3)
        ax_i.legend(ncols=n_phases, loc="upper right")

        for k in range(n_phases):
            ax_v.plot(theta_dense, volt_ph_all[k], color=cmap_ph(k), lw=1.5)
        if volt_max is not None:
            ax_v.axhline(volt_max, color="k", lw=1.0, ls="--", alpha=0.6)
            ax_v.axhline(-volt_max, color="k", lw=1.0, ls="--", alpha=0.6)
        ax_v.set_ylabel("phase voltage [V]")
        ax_v.set_xlabel(r"rotor angle $\theta$ [rad]")
        ax_v.grid(True, alpha=0.3)
        ax_v.set_xlim(0.0, 2 * np.pi)
        ax_v.set_xticks(np.linspace(0, 2 * np.pi, 5))
        ax_v.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

        fig.tight_layout()
    return fig, axes


__all__ = ["evaluate_dq_on_grid", "dynamic_phase_waveforms", "plot_dq_phase_combined"]

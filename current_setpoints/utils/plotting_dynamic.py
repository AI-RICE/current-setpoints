"""
Combined per-period visualization: dq currents, phase currents, phase
voltages, all as line plots vs rotor angle theta on a shared x-axis.

Works in two modes, distinguished by the shape of ``curr_dq``:

* **Static**  -- ``curr_dq`` is a 1D array of length ``transform.dim``.
  The dq panel shows ``dim`` horizontal constant lines; phase panels show
  the waveforms returned by ``Transform.get_curr_ph`` /
  ``Transform.get_volt_ph``. This reproduces information already inside
  the existing static solver that was previously discarded after the
  peak-over-theta reduction in ``constraints.py``.

* **Dynamic** -- ``curr_dq`` is a 2D array of shape ``(M, dim)`` sampled
  at angles ``theta_dq`` (defaults to a uniform grid ``[0, 2*pi)``).
  The dq panel shows the per-angle trajectory as lines; the phase panels
  are recomputed row-by-row from the densely interpolated ``curr_dq``,
  with the inductive ``omega * L_stat * d(curr_dq)/d(theta)`` term added
  to the phase voltage so the displayed waveform matches the dynamic
  stator equation from ``Dynamic.tex``.

If ``curr_dq_static`` is supplied in dynamic mode, it is overlaid on the
dq panel as dashed horizontal lines so the dynamic departure from the
static MTPA point is read off directly.

Import:

    from current_setpoints.utils.plotting_dynamic import plot_dq_phase_combined
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..simulation import Transform
from .plot_config import PlotConfig


def _harmonic_dq_labels(dim: int) -> list[str]:
    """Returns ``[i_{d1}, i_{q1}, i_{d3}, i_{q3}, ...]`` for the given dim."""
    labels: list[str] = []
    for i in range(dim // 2):
        h = 2 * i + 1
        labels.append(rf"$i_{{d^{h}}}$")
        labels.append(rf"$i_{{q^{h}}}$")
    return labels


def _phase_currents_all(transform: Transform, curr_ph_a: np.ndarray) -> np.ndarray:
    """
    Reconstructs all ``n_phases`` phase current waveforms by rolling the
    phase-A waveform with the integer phase-shift used inside ``Transform``.
    Drops the periodic wrap sample so the roll is exact.

    Args:
        transform: provides ``n_phases`` and ``_phase_shift_samples``.
        curr_ph_a: phase-A waveform of length ``n_theta + 1``.

    Returns:
        Array of shape ``(n_phases, n_theta + 1)``.
    """
    shift = transform._phase_shift_samples
    core = curr_ph_a[:-1]  # n_theta samples on [0, 2*pi)
    rows = [np.roll(core, k * shift) for k in range(transform.n_phases)]
    full = np.stack([np.concatenate([r, r[:1]]) for r in rows], axis=0)
    return full


def _evaluate_dq_on_grid(
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


def _dynamic_phase_waveforms(
    transform: Transform,
    omega: float,
    theta_grid: np.ndarray,
    curr_dq_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes phase-A current and phase-A voltage waveforms from a per-node
    dq trajectory. The voltage includes the inductive derivative term that
    the static formulation omits:

        v_a(theta) = mat_dq_to_ph(theta) @ ( mat_curr_dq_to_volt_dq @ curr_dq(theta)
                       + bemf_dq(theta)
                       + omega * L_stat @ d curr_dq / d theta )

    Args:
        transform: provides ``mat_dq_to_ph``, ``mat_curr_dq_to_volt_dq``,
            and machine matrices.
        omega: electrical speed [rad/s].
        theta_grid: shape ``(M,)`` node angles in ``[0, 2*pi)``.
        curr_dq_grid: shape ``(M, dim)``.

    Returns:
        ``(curr_ph_a, volt_ph_a)`` each of length ``n_theta + 1``.
    """
    transform._set_omega(omega)
    theta_dense = transform.vec_theta
    curr_dense = _evaluate_dq_on_grid(theta_grid, curr_dq_grid, theta_dense)

    curr_ph_a = np.einsum("ij,ij->i", transform.mat_dq_to_ph, curr_dense)

    static_dq = curr_dense @ transform.mat_curr_dq_to_volt_dq.T
    flux_volt, _ = transform.flux.get_flux(omega, curr_dq_grid.mean(axis=0))
    volt_bemf_dq = omega * transform.machine.mat_crossc @ flux_volt
    static_dq = static_dq + volt_bemf_dq[None, :]

    dtheta = theta_dense[1] - theta_dense[0]
    dxdtheta = np.gradient(curr_dense, dtheta, axis=0)
    inductive_dq = omega * dxdtheta @ transform.machine.L_stat.T

    volt_dq_total = static_dq + inductive_dq
    volt_ph_a = np.einsum("ij,ij->i", transform.mat_dq_to_ph, volt_dq_total)
    return curr_ph_a, volt_ph_a


def plot_dq_phase_combined(
    transform: Transform,
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
        transform: ``Transform`` instance configured for the machine.
        omega: electrical speed [rad/s] at which to evaluate phase waveforms.
        curr_dq: either a 1D vector of length ``transform.dim`` (static) or
            a 2D array of shape ``(M, transform.dim)`` (dynamic, per-node).
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

    if is_dynamic:
        if theta_dq is None:
            theta_dq = np.linspace(0.0, 2 * np.pi, curr_dq.shape[0], endpoint=False)
        curr_ph_a, volt_ph_a = _dynamic_phase_waveforms(transform, omega, theta_dq, curr_dq)
    else:
        if curr_dq.shape[0] != transform.dim:
            raise ValueError(f"static curr_dq must have length {transform.dim}, got {curr_dq.shape[0]}.")
        curr_ph_a = transform.get_curr_ph(omega, curr_dq)
        volt_ph_a, _, _ = transform.get_volt_ph(omega, curr_dq)

    theta_dense = transform.vec_theta
    curr_ph_all = _phase_currents_all(transform, curr_ph_a)
    volt_ph_all = _phase_currents_all(transform, volt_ph_a)

    dq_labels = _harmonic_dq_labels(transform.dim)
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
            for j in range(transform.dim):
                ax_dq.axhline(curr_dq_static[j], color=cmap_dq(j), lw=1.0, ls="--", alpha=0.7)
        ax_dq.set_ylabel("dq current [A]")
        ax_dq.grid(True, alpha=0.3)
        ax_dq.legend(ncols=max(1, transform.dim // 2), loc="upper right")
        if title is not None:
            ax_dq.set_title(title)

        cmap_ph = plt.get_cmap("tab10")
        for k in range(transform.n_phases):
            ax_i.plot(theta_dense, curr_ph_all[k], color=cmap_ph(k), lw=1.5, label=f"phase {k}")
        if curr_max is not None:
            ax_i.axhline(curr_max, color="k", lw=1.0, ls="--", alpha=0.6)
            ax_i.axhline(-curr_max, color="k", lw=1.0, ls="--", alpha=0.6)
        ax_i.set_ylabel("phase current [A]")
        ax_i.grid(True, alpha=0.3)
        ax_i.legend(ncols=transform.n_phases, loc="upper right")

        for k in range(transform.n_phases):
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


__all__ = ["plot_dq_phase_combined"]

"""
Regime descriptors for dynamic-MTPA trajectories.

In the static formulation regimes were labelled by counting discrete peaks
of the phase current / voltage waveforms at the limit
(``transform.count_peaks``). In the dynamic / position-dependent setting
limits can be touched on entire arcs, so the peak count is ill-defined.
We replace it with continuous descriptors:

* ``alpha_k^I`` / ``alpha_k^V``: fraction of the period within ``eps`` of
  the current / voltage limit on phase ``k`` (arc length).
* ``n_k^I`` / ``n_k^V``: number of contiguous at-limit arcs per period on
  phase ``k`` (the dynamic generalisation of the static "1 peak" vs
  "2 peaks" distinction).

The *active set* is the set of ``(kind, phase, n_arcs)`` triples whose arc
fraction is above a threshold; a *regime change* in a sweep is any change
in the active set between consecutive operating points.
"""

from __future__ import annotations

import numpy as np

from current_setpoints.simulation import Transform
from current_setpoints.utils.plotting_dynamic import (
    _dynamic_phase_waveforms,
    _phase_currents_all,
)


def _phase_waveforms_all(transform: Transform, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns ``(curr_ph_all, volt_ph_all)`` of shape ``(n_phases, n_theta + 1)``,
    handling both static (1D ``curr_dq``) and dynamic (2D ``curr_dq``) modes.
    """
    if curr_dq.ndim == 2:
        theta_grid = np.linspace(0.0, 2 * np.pi, curr_dq.shape[0], endpoint=False)
        curr_ph_a, volt_ph_a = _dynamic_phase_waveforms(transform, omega, theta_grid, curr_dq)
    else:
        curr_ph_a = transform.get_curr_ph(omega, curr_dq)
        volt_ph_a, _, _ = transform.get_volt_ph(omega, curr_dq)
    return (
        _phase_currents_all(transform, curr_ph_a),
        _phase_currents_all(transform, volt_ph_a),
    )


def _count_arcs_periodic(mask: np.ndarray) -> int:
    """
    Number of contiguous True arcs in a 1D periodic boolean mask. Counts
    rising edges of the periodic signal (wrap last sample to first).
    """
    if mask.size == 0 or not mask.any():
        return 0
    m = mask.astype(np.int8)
    transitions = np.diff(np.r_[m[-1], m])
    return int((transitions == 1).sum())


def fingerprint(
    transform: Transform,
    omega: float,
    curr_dq: np.ndarray,
    curr_max: float,
    volt_max: float,
    eps: float = 5e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns ``(alpha_I, alpha_V, n_I, n_V)``, each of shape ``(n_phases,)``.
    """
    curr_all, volt_all = _phase_waveforms_all(transform, omega, curr_dq)
    mask_I = np.abs(curr_all) >= (1.0 - eps) * curr_max
    mask_V = np.abs(volt_all) >= (1.0 - eps) * volt_max
    alpha_I = mask_I.mean(axis=1)
    alpha_V = mask_V.mean(axis=1)
    n_I = np.array([_count_arcs_periodic(mask_I[k, :-1]) for k in range(mask_I.shape[0])])
    n_V = np.array([_count_arcs_periodic(mask_V[k, :-1]) for k in range(mask_V.shape[0])])
    return alpha_I, alpha_V, n_I, n_V


def active_set(
    alpha_I: np.ndarray,
    alpha_V: np.ndarray,
    n_I: np.ndarray,
    n_V: np.ndarray,
    thresh: float = 1e-2,
) -> frozenset[tuple[str, int, int]]:
    """
    Active set as ``(kind, phase, n_arcs)`` triples with arc fraction above
    ``thresh`` (a fraction of the period, e.g. 0.01 = 1%).
    """
    items: list[tuple[str, int, int]] = []
    for k, (a, n) in enumerate(zip(alpha_I, n_I, strict=True)):
        if a > thresh:
            items.append(("I", k, int(n)))
    for k, (a, n) in enumerate(zip(alpha_V, n_V, strict=True)):
        if a > thresh:
            items.append(("V", k, int(n)))
    return frozenset(items)


def active_set_tag(active: frozenset[tuple[str, int, int]] | None) -> str:
    """Filename-safe tag, e.g. ``free``, ``I0x1``, ``I0x2_I1x2``, or ``FAIL``."""
    if active is None:
        return "FAIL"
    if not active:
        return "free"
    return "_".join(f"{kind}{ph}x{n}" for (kind, ph, n) in sorted(active))


__all__ = [
    "fingerprint",
    "active_set",
    "active_set_tag",
]

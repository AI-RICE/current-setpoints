"""
Unified, validated per-phase voltage reconstruction for an explicit
i_dq(theta)/di_dq(theta) trajectory (healthy and faulted) -- shared by the
fourier head-to-head experiments (e15, e17) and their diagnostics.

Ported from the pre-refactor ``dynamic/phase_voltage.py`` (deleted during
the library port: it duplicated ForwardModel for the single fixed-current
case, but NOT for this trajectory-with-derivative case, which
``ForwardModel.volt_ph``/``volt_map_at_theta`` don't cover on their own).

  v_dq(theta) = (R + omega*J*L) i_dq(theta) + omega*L di_dq/dtheta + e_dq
  v_phase_k(theta) = P_k(theta) . v_dq(theta)            [inverse Park]

Per-phase *voltage* uses inverse-Park and is fault-independent; current
under fault uses the reduced-Clarke map instead (``ForwardModel.curr_ph``).
See docs/sota.md §3 and tests/test_new_forward_model.py's h_k^I/h_k^V tests.
"""

from __future__ import annotations

import numpy as np

from current_setpoints.models.forward_model import ForwardModel


def phase_basis(fwd: ForwardModel, idx: np.ndarray, phases: tuple[int, ...] | None = None) -> np.ndarray:
    """
    Inverse-Park (dq->phase) rows P_k(theta) for the requested physical
    phases at sample indices ``idx`` into ``fwd.vec_theta``.

    Returns:
        Array ``(n_sel, n_idx, dim)`` with row [p, n, :] = P_{phases[p]}(idx[n]).
    """
    if phases is None:
        phases = tuple(range(fwd.drive.n_phases))
    return fwd._mat_dq_to_ph_all[idx][:, list(phases), :].transpose(1, 0, 2)


def voltage_dq(fwd: ForwardModel, omega: float, i_dq: np.ndarray, di_dq: np.ndarray) -> np.ndarray:
    """
    dq voltage trajectory ``v_dq = (R + omega J L) i_dq + omega L di_dq + e_dq``.

    Args:
        i_dq, di_dq: (n, dim) dq current trajectory and its d/dtheta.

    Returns:
        (n, dim) dq voltage trajectory.
    """
    dim = fwd.drive.dim
    zeros = np.zeros(dim)
    U = fwd.drive.voltage_operator(omega, zeros)
    L = fwd.drive.inductance(omega, zeros)
    bemf_dq = fwd.drive.bemf_dq(omega, zeros)
    return i_dq @ U.T + omega * di_dq @ L.T + bemf_dq[None, :]


def phase_voltage(
    fwd: ForwardModel,
    omega: float,
    i_dq: np.ndarray,
    di_dq: np.ndarray,
    idx: np.ndarray,
    phases: tuple[int, ...] | None = None,
) -> np.ndarray:
    """
    Per-phase terminal voltage v_k(theta) = P_k(theta) . v_dq(theta).

    Returns:
        (n_sel, n) per-phase voltage, n = idx.size, n_sel = len(phases).
    """
    v_dq = voltage_dq(fwd, omega, i_dq, di_dq)  # (n, dim)
    Pk = phase_basis(fwd, idx, phases)          # (n_sel, n, dim)
    return np.einsum("pnj,nj->pn", Pk, v_dq)


def voltage_linear_maps(
    fwd: ForwardModel, omega: float, idx: np.ndarray, phases: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear maps for optimiser constraints: per-phase voltage is
    ``v_k(theta) = gU[k,theta] . i_dq(theta) + gL[k,theta] . di_dq(theta) + b[k,theta]``.

    Returns:
        gU, gL: (n_sel, n, dim) acting on i_dq / omega*di_dq respectively.
        b: (n_sel, n) back-EMF offset.
    """
    dim = fwd.drive.dim
    zeros = np.zeros(dim)
    U = fwd.drive.voltage_operator(omega, zeros)
    L = fwd.drive.inductance(omega, zeros)
    bemf_dq = fwd.drive.bemf_dq(omega, zeros)
    Pk = phase_basis(fwd, idx, phases)          # (n_sel, n, dim)
    gU = np.einsum("pnj,jl->pnl", Pk, U)
    gL = np.einsum("pnj,jl->pnl", Pk, L)
    b = np.einsum("pnj,j->pn", Pk, bemf_dq)
    return gU, gL, b


__all__ = ["phase_basis", "voltage_dq", "phase_voltage", "voltage_linear_maps"]

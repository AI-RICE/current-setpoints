"""
Unified, validated per-phase voltage reconstruction (healthy and faulted).

Single source of truth for "what voltage must the inverter supply to phase
k(theta)" given a dq current reference. Replaces the ad-hoc reconstructions
that diverged between experiments (envelope_solver used the reduced-Clarke
*current* map on v_dq; e17's Fall arm used inverse-Park; e15 used the
reduced-Clarke map too).

Physics / convention (matches the identified dq model and the validated
``Transform.get_volt_ph``):

  v_dq(theta) = (R + omega*J*L) i_dq + omega*L di_dq/dtheta + e_dq
  v_phase_k(theta) = P_k(theta) . v_dq(theta)            [inverse Park]

where P_k is the inverse-Park (dq->phase) row for physical phase k, i.e.
``Transform.mat_dq_to_ph`` sampled at theta shifted by k phase positions.

Key point that fixes the high-speed fault bug: the per-phase *voltage*
transform is inverse-Park and is **fault-independent** -- an open phase does
not change the machine's voltage transform, it only forces that phase's
*current* to zero. The current realisation under fault therefore uses the
reduced-Clarke map (see ``envelope_solver.fault_phase_map``), but the
voltage uses inverse-Park for every (still-connected) phase. Using the
current map on v_dq (the old behaviour) over-counts surviving-phase voltage
once omega*L di/dtheta becomes significant.

Resistive-term note: ``R`` here is the identified dq resistance
(``machine.R_stat``), so the resistive drop is taken in the dq frame
(P_k @ R i_dq). At the speeds where voltage binds this term is small versus
the inductive + back-EMF terms; the dominant correction is the map (P_k vs
reduced-Clarke), not the resistive sub-term.
"""

from __future__ import annotations

import numpy as np

from current_setpoints.simulation import Transform


def phase_basis(transform: Transform, idx: np.ndarray, phases: tuple[int, ...] | None = None) -> np.ndarray:
    """
    Inverse-Park (dq->phase) rows P_k(theta) for the requested physical phases
    at sample indices ``idx`` into ``transform.vec_theta``.

    Args:
        transform: configured Transform (provides mat_dq_to_ph, shift, dim).
        idx: integer sample indices (any shape, here treated 1-D), into the
            periodic grid of ``n_theta = vec_theta.size - 1`` samples.
        phases: physical phase indices to return; default all ``n_phases``.

    Returns:
        Array ``(n_sel, n_idx, dim)`` with row [p, n, :] = P_{phases[p]}(idx[n]).
    """
    if phases is None:
        phases = tuple(range(transform.n_phases))
    shift = transform._phase_shift_samples
    n_theta = transform.vec_theta.size - 1
    rows = [transform.mat_dq_to_ph[(idx - k * shift) % n_theta] for k in phases]
    return np.stack(rows, axis=0)  # (n_sel, n_idx, dim)


def voltage_dq(transform: Transform, omega: float, i_dq: np.ndarray, di_dq: np.ndarray) -> np.ndarray:
    """
    dq voltage trajectory ``v_dq = (R + omega J L) i_dq + omega L di_dq + e_dq``.

    Args:
        transform: configured Transform.
        omega: electrical speed [rad/s].
        i_dq: (n, dim) dq current trajectory.
        di_dq: (n, dim) d i_dq / d theta trajectory (same grid as i_dq).

    Returns:
        (n, dim) dq voltage trajectory.
    """
    transform._set_omega(omega)
    U = transform.mat_curr_dq_to_volt_dq          # R + omega * crossc @ L
    L = transform.machine.L_stat
    flux_volt, _ = transform.flux.get_flux(omega, np.zeros(transform.dim))
    bemf_dq = omega * transform.machine.mat_crossc @ flux_volt
    return i_dq @ U.T + omega * di_dq @ L.T + bemf_dq[None, :]


def phase_voltage(
    transform: Transform,
    omega: float,
    i_dq: np.ndarray,
    di_dq: np.ndarray,
    idx: np.ndarray,
    phases: tuple[int, ...] | None = None,
) -> np.ndarray:
    """
    Per-phase terminal voltage v_k(theta) = P_k(theta) . v_dq(theta).

    Args:
        transform, omega, i_dq, di_dq: see ``voltage_dq``. i_dq/di_dq are
            sampled at the angles ``transform.vec_theta[idx]``.
        idx: sample indices locating i_dq/di_dq on the periodic grid.
        phases: physical phases to evaluate; default all.

    Returns:
        (n_sel, n) per-phase voltage, n = idx.size, n_sel = len(phases).
    """
    v_dq = voltage_dq(transform, omega, i_dq, di_dq)  # (n, dim)
    Pk = phase_basis(transform, idx, phases)          # (n_sel, n, dim)
    return np.einsum("pnj,nj->pn", Pk, v_dq)


def voltage_linear_maps(
    transform: Transform,
    omega: float,
    idx: np.ndarray,
    phases: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear maps for optimiser constraints: per-phase voltage is
    ``v_k(theta) = gU[k,theta] . i_dq(theta) + gL[k,theta] . di_dq(theta) + b[k,theta]``.

    Returns:
        gU: (n_sel, n, dim) acting on i_dq         (= P_k @ U)
        gL: (n_sel, n, dim) acting on omega*di_dq  (= P_k @ L)   [caller scales by omega]
        b:  (n_sel, n)      back-EMF offset         (= P_k @ e_dq)
    """
    transform._set_omega(omega)
    U = transform.mat_curr_dq_to_volt_dq
    L = transform.machine.L_stat
    flux_volt, _ = transform.flux.get_flux(omega, np.zeros(transform.dim))
    bemf_dq = omega * transform.machine.mat_crossc @ flux_volt
    Pk = phase_basis(transform, idx, phases)          # (n_sel, n, dim)
    gU = np.einsum("pnj,jl->pnl", Pk, U)
    gL = np.einsum("pnj,jl->pnl", Pk, L)
    b = np.einsum("pnj,j->pn", Pk, bemf_dq)
    return gU, gL, b

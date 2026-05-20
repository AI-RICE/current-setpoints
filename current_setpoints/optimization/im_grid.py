"""
Grid calculation engine for the 9-phase induction motor. Mirrors the
PMSM ``calculate_grid`` in ``grid.py`` but with three adjustments:

1. State dimension is ``machine.dim`` (= 4 for IM9Phase with 1st+3rd
   harmonics), not ``machine.n_phases - 1`` (which would be 8).
2. Warm starts and global-max-torque probes use a small positive
   ``omega_s`` rather than zero, because the rotor-flux-orientation
   slip law ``omega_r = (R_r^1/L_r^1)(i_sd^1/i_sq^1)`` is degenerate
   when ``i_sq^1 = 0`` (the candidate i_s = [1,0,0,0] used by the
   PMSM grid produces no torque for IM).
3. Initial guess seeds both d and q components so the slip law is
   well-defined from the start.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..simulation import IMTransform
from .im_model import ModelIMAnalytical
from .grid import _init_grid_arrays, _fill_grid_point
from .optimizer import MotorOptimizer


def calculate_grid_im(
    optimizer: MotorOptimizer,
    transform: IMTransform,
    opts: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a (T*, omega_s) grid of minimum-current setpoints for a
    multiphase IM.

    Args:
        optimizer: ``MotorOptimizer`` instance wrapping a
            ``ModelIMAnalytical``.
        transform: ``IMTransform`` instance for the target machine.
        opts: dict with keys
            ``n_torq`` (int), ``n_omega`` (int),
            ``torq_min`` (float, Nm),
            ``omega_min`` (float, mechanical RPM upper bound for the
            sweep lower-end, in mechanical RPM; the upper bound is
            ``transform.machine.omega_max`` in mechanical RPM as in
            the PMSM grid),
            ``omega_probe`` (optional float, electrical rad/s; default
            50.0): stator electrical speed at which the global maximum
            torque is computed.

    Returns:
        Same dict schema as ``calculate_grid``.
    """
    print("Starting IM Grid Calculation...")

    machine = transform.machine
    dim = machine.dim

    grid: dict[str, Any] = {}
    grid["const_mech_speed"] = 30.0 / (np.pi * machine.n_ppairs)

    omega_probe = float(opts.get("omega_probe", 50.0))
    _, torq_max_global, ok_global = optimizer.maximize_torque(
        omega=omega_probe, transform=transform
    )
    if not ok_global:
        raise RuntimeError(
            f"Global max-torque probe at omega={omega_probe} failed; "
            "adjust omega_probe in opts or extend candidate seeds."
        )
    print(f"  Global T_max at omega_s={omega_probe:.1f} rad/s: {torq_max_global:.3f} Nm")

    n_torq, n_omega = opts["n_torq"], opts["n_omega"]
    grid["vec_torq"] = np.linspace(opts["torq_min"], torq_max_global, n_torq)
    grid["vec_omega"] = np.linspace(
        opts["omega_min"] / grid["const_mech_speed"],
        machine.omega_max / grid["const_mech_speed"],
        n_omega,
    )

    grid.update(_init_grid_arrays(dim, n_torq, n_omega))

    for idx_omega in range(n_omega):
        omega_target = grid["vec_omega"][idx_omega]
        rpm = omega_target * grid["const_mech_speed"]
        print(f"  Calculating: Omega step {idx_omega + 1}/{n_omega} ({rpm:.1f} RPM)")

        _, torq_max_local, ok_local = optimizer.maximize_torque(
            omega=omega_target, transform=transform
        )
        if ok_local:
            grid["vec_torq_max"][0, idx_omega] = torq_max_local
        else:
            torq_max_local = 0.0

        # IM warm-start: small balanced (d, q) so the slip law is
        # well-defined. Updated to the previous successful solution
        # as we sweep up in torque.
        vec_curr_dq_prev = np.zeros(dim)
        vec_curr_dq_prev[0] = 0.1
        vec_curr_dq_prev[1] = 0.1

        for idx_torq in range(n_torq):
            torq_target = grid["vec_torq"][idx_torq]
            if torq_target > torq_max_local or torq_target > torq_max_global:
                break

            vec_curr_dq_opt, success = optimizer.minimize_current(
                torq_target=torq_target,
                omega=omega_target,
                transform=transform,
                vec_curr_guess=vec_curr_dq_prev.copy(),
            )

            if success:
                vec_curr_dq_prev = vec_curr_dq_opt
                _fill_grid_point(
                    grid,
                    transform,
                    omega_target,
                    vec_curr_dq_opt,
                    idx_torq,
                    idx_omega,
                )

    print("IM Grid Calculation Complete.")
    return grid

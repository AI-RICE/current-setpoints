from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..models.forward_model import ForwardModel
from .optimizer import BaseOptimizer

if TYPE_CHECKING:
    from ..models.machines import NeuralPMSM5Phase


def count_peaks(
    fwd: ForwardModel,
    omega: float,
    curr_dq: np.ndarray,
    rel_tol: float = 1e-3,
) -> tuple[int, int]:
    return fwd.count_peaks(omega, curr_dq, rel_tol)


def _init_grid_arrays(dim: int, n_torq: int, n_omega: int) -> dict[str, np.ndarray]:
    grid: dict[str, np.ndarray] = {
        k: np.full((n_torq, n_omega), np.nan) for k in ("grid_curr_peak", "grid_volt_peak", "grid_segments")
    }
    grid["vec_torq_max"] = np.full((1, n_omega), np.nan)
    grid["curr_dq_grid"] = np.full((dim, n_torq, n_omega), np.nan)
    return grid


def _fill_grid_point(
    grid: dict[str, Any],
    fwd: ForwardModel,
    omega: float,
    curr_dq: np.ndarray,
    idx_torq: int,
    idx_omega: int,
) -> None:
    curr_peak, volt_peak = fwd.peak_vals(omega, curr_dq)
    n_curr, n_volt = fwd.count_peaks(omega, curr_dq)

    grid["curr_dq_grid"][:, idx_torq, idx_omega] = curr_dq
    grid["grid_curr_peak"][idx_torq, idx_omega] = curr_peak
    grid["grid_volt_peak"][idx_torq, idx_omega] = volt_peak
    grid["grid_segments"][idx_torq, idx_omega] = 3 * n_volt + n_curr


def calculate_grid(
    optimizer: BaseOptimizer,
    fwd: ForwardModel,
    opts: dict[str, Any],
    mode: str = "standard",
    dict_grid_corr: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in ("standard", "recalculated"):
        raise ValueError(f"mode must be 'standard' or 'recalculated', got {mode!r}")
    if mode == "recalculated" and dict_grid_corr is None:
        raise ValueError("dict_grid_corr must be provided for recalculated mode.")

    drive = fwd.drive
    dim = drive.dim
    const_mech_speed = 30.0 / (np.pi * drive.n_ppairs)
    omega_probe = float(opts.get("omega_probe", 0.0))

    grid: dict[str, Any] = {"const_mech_speed": const_mech_speed}

    if mode == "standard":
        if "torq_max" in opts and opts["torq_max"] is not None:
            torq_max_global = float(opts["torq_max"])
        else:
            sol = optimizer.maximize_torque(omega_probe)
            if not sol.success:
                raise RuntimeError(
                    f"Global T_max probe at omega={omega_probe} rad/s failed. "
                    "Adjust omega_probe in opts or extend seeds."
                )
            torq_max_global = sol.torque

        n_torq, n_omega = opts["n_torq"], opts["n_omega"]
        grid["vec_torq"] = np.linspace(opts["torq_min"], torq_max_global, n_torq)
        grid["vec_omega"] = np.linspace(
            opts["omega_min"] / const_mech_speed,
            drive.omega_max / const_mech_speed,
            n_omega,
        )
        grid_torq_targets = None
    else:
        n_torq = len(dict_grid_corr["vec_torq"])  # type: ignore[index]
        n_omega = len(dict_grid_corr["vec_omega"])  # type: ignore[index]
        grid["vec_torq"] = dict_grid_corr["vec_torq"]  # type: ignore[index]
        grid["vec_omega"] = dict_grid_corr["vec_omega"]  # type: ignore[index]
        grid_torq_targets = dict_grid_corr["grid_torq_neural"]  # type: ignore[index]
        grid["grid_torq_neural"] = grid_torq_targets
        torq_max_global = float(grid["vec_torq"][-1])

    grid.update(_init_grid_arrays(dim, n_torq, n_omega))

    print(f"Starting {mode} grid: {n_torq} torques × {n_omega} speeds")

    # Warm-start for maximize_torque carried across speeds (FW corner convergence)
    prev_max_curr: np.ndarray | None = None

    for idx_omega in range(n_omega):
        omega = float(grid["vec_omega"][idx_omega])
        rpm = omega * const_mech_speed
        print(f"  Speed {idx_omega + 1}/{n_omega}: {rpm:.1f} RPM")

        # Per-speed T_max probe (warm-started from previous speed)
        sol_max = optimizer.maximize_torque(omega, guess=prev_max_curr)
        if sol_max.success:
            torq_max_local = sol_max.torque
            prev_max_curr = sol_max.curr_dq
        else:
            # Fall back to global estimate so cells aren't silently skipped
            torq_max_local = torq_max_global if mode == "standard" else float("inf")
        grid["vec_torq_max"][0, idx_omega] = torq_max_local

        # Initial warm-start for the torque sweep: first seed from drive
        prev_curr = drive.seeds()[0]

        for idx_torq in range(n_torq):
            if mode == "standard":
                torq_target = float(grid["vec_torq"][idx_torq])
                guess = prev_curr.copy()
                if torq_target > min(torq_max_local, torq_max_global):
                    break
            else:
                torq_target = float(grid_torq_targets[idx_torq, idx_omega])  # type: ignore[index]
                corr_curr = dict_grid_corr["curr_dq_grid"][:, idx_torq, idx_omega]  # type: ignore[index]
                if np.isnan(torq_target) or np.any(np.isnan(corr_curr)) or torq_target > torq_max_local:
                    continue
                guess = corr_curr

            sol = optimizer.minimize_current(torq_target, omega, guess=guess)

            if mode == "standard":
                if sol.success:
                    prev_curr = sol.curr_dq
                    _fill_grid_point(grid, fwd, omega, sol.curr_dq, idx_torq, idx_omega)
            else:
                # Recalculated: use optimizer result if converged, else keep baseline
                curr_final = sol.curr_dq if sol.success else guess
                _fill_grid_point(grid, fwd, omega, curr_final, idx_torq, idx_omega)

    print(f"{mode.capitalize()} grid complete.")
    return grid


def get_correction_grid(
    dict_grid: dict[str, Any],
    neural: "NeuralPMSM5Phase",
) -> dict[str, Any]:
    print("Computing neural correction grid...")

    out = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in dict_grid.items()}

    dim, n_torq, n_omega = dict_grid["curr_dq_grid"].shape
    vec_omega = dict_grid["vec_omega"]
    const_mech_speed = dict_grid["const_mech_speed"]
    grid_torq_neural = np.full((n_torq, n_omega), np.nan)

    for idx_omega in range(n_omega):
        omega = float(vec_omega[idx_omega])
        for idx_torq in range(n_torq):
            curr = dict_grid["curr_dq_grid"][:, idx_torq, idx_omega]
            if np.any(np.isnan(curr)):
                continue
            grid_torq_neural[idx_torq, idx_omega] = neural.torque(omega, curr)

    out["grid_torq_neural"] = grid_torq_neural
    print("Neural correction grid complete.")
    return out

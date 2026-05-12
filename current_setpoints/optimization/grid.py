from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..simulation import MachineData, Transform
from .models import ModelAnalytical, ModelNeural
from .optimizer import MotorOptimizer


def grid_to_data(grid: dict[str, Any], k_skip: int) -> MachineData:
    """
    Converts a grid dictionary into a structured MachineData object.

    Args:
        grid: Dictionary containing optimization results and vectors.
        k_skip: Step size for downsampling the grid; forwarded to
            ``MachineData.select_k``. A value of 1 keeps every sample.

    Returns:
        MachineData: Encapsulated motor data for training or analysis.
    """
    torq = grid["vec_torq"]
    omega = grid["const_mech_speed"] * grid["vec_omega"]
    return MachineData(
        torq=torq,
        omega=omega,
        segments=grid["grid_segments"],
        curr_dq_grid=grid["curr_dq_grid"],
        k_skip=k_skip,
    )


def _init_grid_arrays(dim: int, n_torq: int, n_omega: int) -> dict[str, np.ndarray]:
    """
    Initializes empty 2D and 3D matrices for all physical parameters in the motor grid.
    """
    keys = [
        "grid_curr_peak",
        "grid_volt_peak",
        "grid_curr_ang_diff",
        "grid_volt_ang_diff",
        "grid_segments",
    ]

    grid: dict[str, np.ndarray] = {k: np.full((n_torq, n_omega), np.nan) for k in keys}

    grid["vec_torq_max"] = np.full((1, n_omega), np.nan)
    grid["curr_dq_grid"] = np.full((dim, n_torq, n_omega), np.nan)

    return grid


def _fill_grid_point(
    grid: dict[str, Any],
    transform: Transform,
    omega: float,
    vec_curr_dq: np.ndarray,
    idx_torq: int,
    idx_omega: int,
) -> None:
    """
    Calculates physical metrics for a specific DQ current vector and stores
    them in the grid.

    Args:
        grid: The results dictionary to update.
        transform: The Transform instance used for voltage/peak calculations.
        omega: Electrical speed [rad/s] at this operating point.
        vec_curr_dq: DQ current vector for this operating point.
        idx_torq: Current torque index.
        idx_omega: Current speed index.
    """
    curr_peak, curr_ang_diff, volt_peak, volt_ang_diff = transform.get_max_vals(omega, vec_curr_dq)

    n_curr_peaks, n_volt_peaks = transform.count_peaks(omega, vec_curr_dq)

    grid["curr_dq_grid"][:, idx_torq, idx_omega] = vec_curr_dq

    grid["grid_curr_peak"][idx_torq, idx_omega] = curr_peak
    grid["grid_volt_peak"][idx_torq, idx_omega] = volt_peak
    grid["grid_curr_ang_diff"][idx_torq, idx_omega] = curr_ang_diff
    grid["grid_volt_ang_diff"][idx_torq, idx_omega] = volt_ang_diff

    grid["grid_segments"][idx_torq, idx_omega] = 3 * n_volt_peaks + n_curr_peaks


def calculate_grid(
    optimizer: MotorOptimizer,
    transform: Transform,
    opts: dict[str, Any],
    mode: str = "standard",
    dict_grid_corr: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Unified grid calculation engine for motor mapping.

    Modes:
      - "standard": Generates a new grid (Baseline or Compensated) using warm-starts.
      - "recalculated": Re-runs an existing grid through a different model/optimizer.

    Args:
        optimizer: MotorOptimizer instance containing the torque model.
        transform: Transform instance handling speed-dependent matrices.
        opts: Configuration dictionary with keys:
            ``n_torq`` (int), ``n_omega`` (int),
            ``torq_min`` (float, Nm),
            ``omega_min`` (float, mechanical RPM; ``machine.omega_max`` is
            used as the upper bound and is also in mechanical RPM).
        mode: Calculation strategy choice.
        dict_grid_corr: Baseline result dictionary required for "recalculated" mode.

    Returns:
        Dict: Completed motor grid dictionary.
    """
    if mode not in ("standard", "recalculated"):
        raise ValueError("Mode must be either 'standard' or 'recalculated'")

    print(f"Starting {mode.capitalize()} Grid Calculation...")

    machine = transform.machine
    dim = machine.n_phases - 1

    grid: dict[str, Any] = {}
    grid["const_mech_speed"] = 30 / (np.pi * machine.n_ppairs)

    torq_max_global: float = 0.0
    grid_torq_targets: Any = None

    if mode == "standard":
        _, torq_max_global, _ = optimizer.maximize_torque(omega=0.0, transform=transform)

        n_torq, n_omega = opts["n_torq"], opts["n_omega"]
        grid["vec_torq"] = np.linspace(opts["torq_min"], torq_max_global, n_torq)
        grid["vec_omega"] = np.linspace(
            opts["omega_min"] / grid["const_mech_speed"],
            machine.omega_max / grid["const_mech_speed"],
            n_omega,
        )
    else:
        if dict_grid_corr is None:
            raise ValueError("dict_grid_corr must be provided for recalculated mode.")
        n_torq = len(dict_grid_corr["vec_torq"])
        n_omega = len(dict_grid_corr["vec_omega"])
        grid["vec_torq"] = dict_grid_corr["vec_torq"]
        grid["vec_omega"] = dict_grid_corr["vec_omega"]
        grid_torq_targets = dict_grid_corr["grid_torq_neural"]
        grid["grid_torq_neural"] = grid_torq_targets

    grid.update(_init_grid_arrays(dim, n_torq, n_omega))

    for idx_omega in range(n_omega):
        omega_target = grid["vec_omega"][idx_omega]
        print(f"Calculating: Omega step {idx_omega + 1}/{n_omega} ({omega_target * grid['const_mech_speed']:.1f} RPM)")

        _, torq_max_local, _ = optimizer.maximize_torque(omega=omega_target, transform=transform)
        grid["vec_torq_max"][0, idx_omega] = torq_max_local

        vec_curr_dq_prev = np.zeros(dim)
        vec_curr_dq_prev[0] = 1.0

        for idx_torq in range(n_torq):
            if mode == "standard":
                torq_target = grid["vec_torq"][idx_torq]
                vec_curr_dq_guess = vec_curr_dq_prev.copy()

                if torq_target > torq_max_local or torq_target > torq_max_global:
                    break

            else:
                assert dict_grid_corr is not None
                torq_target = grid_torq_targets[idx_torq, idx_omega]

                vec_curr_dq_guess = dict_grid_corr["curr_dq_grid"][:, idx_torq, idx_omega]

                if np.isnan(torq_target) or np.any(np.isnan(vec_curr_dq_guess)) or torq_target > torq_max_local:
                    continue

            vec_curr_dq_opt, success = optimizer.minimize_current(
                torq_target=torq_target,
                omega=omega_target,
                transform=transform,
                vec_curr_guess=vec_curr_dq_guess,
            )

            if mode == "standard":
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
            else:
                vec_curr_dq_final = vec_curr_dq_opt if success else vec_curr_dq_guess
                _fill_grid_point(
                    grid,
                    transform,
                    omega_target,
                    vec_curr_dq_final,
                    idx_torq,
                    idx_omega,
                )

    print(f"{mode.capitalize()} Grid Calculation Complete.")
    return grid


def get_correction_grid(
    dict_grid: dict[str, Any],
    model: ModelNeural,
) -> dict[str, Any]:
    """
    Computes the correction grid by predicting torque using the neural torque
    model on the analytical baseline currents.

    The new ``ModelNeural`` already returns (analytical + neural-residual) from
    its ``calculate_torque`` method, so this function simply walks every valid
    grid cell and asks the model for the total predicted torque. The neural
    residual itself is also computed batched for speed.

    Args:
        dict_grid: Dictionary containing the baseline analytical grid.
        model: A ``ModelNeural`` instance combining analytical + residual logic.
            It exposes ``neural_model``, ``scaler``, ``device``, plus
            ``calculate_torque(omega, curr_dq)`` for the analytical part.

    Returns:
        Dict: A copy of ``dict_grid`` with an added ``grid_torq_neural`` field
        containing the model's torque prediction at every valid cell.
    """
    print("Computing ML Correction Grid...")

    dict_grid_corr = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in dict_grid.items()}

    dim, n_torq, n_omega = dict_grid_corr["curr_dq_grid"].shape
    grid_torq_neural = np.full((n_torq, n_omega), np.nan)

    omega_2d = np.tile(dict_grid_corr["vec_omega"], (n_torq, 1))

    curr_flat = dict_grid_corr["curr_dq_grid"].reshape(dim, -1)
    omega_flat = omega_2d.flatten()

    valid_mask = ~np.isnan(curr_flat[0, :])

    if np.any(valid_mask):
        omega_valid = omega_flat[valid_mask]
        curr_valid = curr_flat[:, valid_mask]
        n_valid = omega_valid.size

        X_in = np.vstack([omega_valid, curr_valid]).T
        X_scaled = model.scaler.transform(X_in)
        X_tensor = torch.from_numpy(X_scaled).float().to(model.device)

        with torch.no_grad():
            residuals = model.neural_model(X_tensor).cpu().numpy().flatten()

        analyticals = np.empty(n_valid, dtype=np.float64)
        for i in range(n_valid):
            analyticals[i] = ModelAnalytical.calculate_torque(model, float(omega_valid[i]), curr_valid[:, i])

        torq_neural_flat = np.full(n_torq * n_omega, np.nan)
        torq_neural_flat[valid_mask] = analyticals + residuals

        grid_torq_neural = torq_neural_flat.reshape(n_torq, n_omega)

    dict_grid_corr["grid_torq_neural"] = grid_torq_neural

    return dict_grid_corr

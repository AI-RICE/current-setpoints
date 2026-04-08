import numpy as np
from typing import Dict, Any, Optional
from ..model import MachineData


def grid_to_data(grid: Dict[str, Any], k_skip: int) -> MachineData:
    """
    Converts a grid dictionary into a structured MachineData object.

    Args:
        grid: Dictionary containing optimization results and vectors.
        k_skip: Number of initial samples to skip for training/processing.

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


def _init_grid_arrays(dim: int, n_torq: int, n_omega: int) -> Dict[str, np.ndarray]:
    """
    Initializes empty 2D and 3D matrices for all physical parameters in the motor grid.

    Args:
        dim: Number of DQ current components.
        n_torq: Number of torque steps.
        n_omega: Number of speed steps.

    Returns:
        Dict: Dictionary of initialized NaN arrays.
    """
    keys = [
        "grid_curr_peak",
        "grid_volt_peak",
        "grid_curr_ang_diff",
        "grid_volt_ang_diff",
        "grid_segments",
        "grid_volt_raw_peak",
        "grid_volt_0_rms",
        "grid_volt_0_peak",
    ]

    grid: Dict[str, np.ndarray] = {k: np.full((n_torq, n_omega), np.nan) for k in keys}

    grid["vec_torq_max"] = np.full((1, n_omega), np.nan)

    grid["curr_dq_grid"] = np.full((dim, n_torq, n_omega), np.nan)

    return grid


def _fill_grid_point(
    grid: Dict[str, Any],
    transform: Any,
    vec_curr_dq: np.ndarray,
    idx_torq: int,
    idx_omega: int,
    machine: Any,
) -> None:
    """
    Calculates physical metrics for a specific DQ current vector and stores them in the grid.

    Args:
        grid: The results dictionary to update.
        transform: The Transform instance used for voltage/peak calculations.
        vec_curr_dq: DQ current vector for this operating point.
        idx_torq: Current torque index.
        idx_omega: Current speed index.
        machine: The Machine object for physical limits and peak counting.
    """
    (
        curr_peak,
        curr_ang_diff,
        _,
        volt_peak,
        volt_ang_diff,
        volt_raw_peak,
        volt_0_rms,
        volt_0_peak,
    ) = transform.get_max_vals(vec_curr_dq)

    n_curr_peaks, n_volt_peaks = transform.count_peaks(vec_curr_dq, machine)

    grid["curr_dq_grid"][:, idx_torq, idx_omega] = vec_curr_dq

    grid["grid_curr_peak"][idx_torq, idx_omega] = curr_peak
    grid["grid_volt_peak"][idx_torq, idx_omega] = volt_peak
    grid["grid_curr_ang_diff"][idx_torq, idx_omega] = curr_ang_diff
    grid["grid_volt_ang_diff"][idx_torq, idx_omega] = volt_ang_diff

    grid["grid_segments"][idx_torq, idx_omega] = 3 * n_volt_peaks + n_curr_peaks


def calculate_grid(
    optimizer: Any,
    transform: Any,
    opts: Dict[str, Any],
    mode: str = "standard",
    dict_grid_corr: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Unified grid calculation engine for motor mapping.

    Modes:
      - "standard": Generates a new grid (Baseline or Compensated) using warm-starts.
      - "recalculated": Re-runs an existing grid through a different model/optimizer.

    Args:
        optimizer: MotorOptimizer instance containing the torque model.
        transform: Transform instance handling speed-dependent matrices.
        opts: Configuration dictionary (n_torq, n_omega, torq_min, omega_min).
        mode: Calculation strategy choice.
        dict_grid_corr: Baseline result dictionary required for "recalculated" mode.

    Returns:
        Dict: Completed motor grid dictionary.
    """
    if mode not in ["standard", "recalculated"]:
        raise ValueError("Mode must be either 'standard' or 'recalculated'")

    print(f"Starting {mode.capitalize()} Grid Calculation...")

    machine = optimizer.model.machine
    dim = machine.n_phases - 1

    grid: Dict[str, Any] = {}
    grid["const_mech_speed"] = 30 / (np.pi * machine.n_ppairs)

    torq_max_global: float = 0.0
    grid_torq_targets: Any = None

    if mode == "standard":
        transform.set_omega(0)
        _, torq_max_global, _ = optimizer.maximize_torque(transform=transform)

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
        grid_torq_targets = dict_grid_corr["grid_torq_pirn"]

    grid.update(_init_grid_arrays(dim, n_torq, n_omega))

    for idx_omega in range(n_omega):
        omega_target = grid["vec_omega"][idx_omega]
        print(
            f"Calculating: Omega step {idx_omega + 1}/{n_omega} ({omega_target * grid['const_mech_speed']:.1f} RPM)"
        )

        transform.set_omega(omega_target)

        _, torq_max_local, _ = optimizer.maximize_torque(transform=transform)
        grid["vec_torq_max"][0, idx_omega] = torq_max_local

        vec_curr_dq_prev = np.zeros(dim)
        vec_curr_dq_prev[0] = 1.0

        for idx_torq in range(n_torq):
            if mode == "standard":
                torq_target = grid["vec_torq"][idx_torq]
                vec_curr_dq_guess = vec_curr_dq_prev

                if torq_target > torq_max_local or torq_target > torq_max_global:
                    break

            else:
                assert dict_grid_corr is not None
                torq_target = grid_torq_targets[idx_torq, idx_omega]

                vec_curr_dq_guess = dict_grid_corr["curr_dq_grid"][
                    :, idx_torq, idx_omega
                ]

                if (
                    np.isnan(torq_target)
                    or np.any(np.isnan(vec_curr_dq_guess))
                    or torq_target > torq_max_local
                ):
                    continue

            vec_curr_dq_opt, success = optimizer.minimize_current(
                torq_target=torq_target,
                transform=transform,
                vec_curr_guess=vec_curr_dq_guess,
            )

            if mode == "standard":
                if success:
                    vec_curr_dq_prev = vec_curr_dq_opt
                    _fill_grid_point(
                        grid, transform, vec_curr_dq_opt, idx_torq, idx_omega, machine
                    )
            else:
                vec_curr_dq_final = vec_curr_dq_opt if success else vec_curr_dq_guess
                _fill_grid_point(
                    grid, transform, vec_curr_dq_final, idx_torq, idx_omega, machine
                )

    print(f"{mode.capitalize()} Grid Calculation Complete.")
    return grid

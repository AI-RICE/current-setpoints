import numpy as np
from typing import Dict, Any, Optional
from model.data import MachineData


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
        isd1=grid["isd1"],
        isd3=grid["isd3"],
        isq1=grid["isq1"],
        isq3=grid["isq3"],
        k_skip=k_skip,
    )


def _init_grid_arrays(n_torq: int, n_omega: int) -> Dict[str, np.ndarray]:
    """
    Initializes empty 2D matrices for all physical parameters in the motor grid.

    Args:
        n_torq: Number of torque steps.
        n_omega: Number of speed steps.

    Returns:
        Dict: Dictionary of initialized NaN arrays.
    """
    keys = [
        "isd1",
        "isd3",
        "isq1",
        "isq3",
        "grid_curr_peak",
        "grid_volt_peak",
        "grid_curr_ang_diff",
        "grid_volt_ang_diff",
        "grid_segments",
        "grid_volt_raw_peak",
        "grid_volt_0_rms",
        "grid_volt_0_peak",
    ]
    grid = {k: np.full((n_torq, n_omega), np.nan) for k in keys}
    grid["vec_torq_max"] = np.full((1, n_omega), np.nan)
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
        vec_curr_dq: 4-element current vector [id1, iq1, id3, iq3].
        idx_torq: Current torque index.
        idx_omega: Current speed index.
        machine: The Machine object for physical limits and peak counting.
    """
    # Extract peaks and angles from the transform
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

    # Determine which constraints are active (peaks)
    n_curr_peaks, n_volt_peaks = transform.count_peaks(vec_curr_dq, machine)

    # Store DQ currents
    grid["isd1"][idx_torq, idx_omega] = vec_curr_dq[0]
    grid["isq1"][idx_torq, idx_omega] = vec_curr_dq[1]
    grid["isd3"][idx_torq, idx_omega] = vec_curr_dq[2]
    grid["isq3"][idx_torq, idx_omega] = vec_curr_dq[3]

    # Store calculated physical quantities
    grid["grid_curr_peak"][idx_torq, idx_omega] = curr_peak
    grid["grid_volt_peak"][idx_torq, idx_omega] = volt_peak
    grid["grid_curr_ang_diff"][idx_torq, idx_omega] = curr_ang_diff
    grid["grid_volt_ang_diff"][idx_torq, idx_omega] = volt_ang_diff

    # Segment logic: unique ID representing which constraints are active
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

    # Extract machine properties from the unified model hierarchy
    machine = optimizer.model.machine
    grid: Dict[str, Any] = {}
    grid["const_mech_speed"] = 30 / (np.pi * machine.n_ppairs)

    torq_max_global: float = 0.0
    grid_torq_targets: Any = None

    if mode == "standard":
        # Determine global torque scale at zero speed (Maximum capability)
        transform.set_omega(0)
        _, torq_max_global, _ = optimizer.maximize_torque(transform=transform)

        n_torq, n_omega = opts["n_torq"], opts["n_omega"]
        grid["vec_torq"] = np.linspace(opts["torq_min"], torq_max_global, n_torq)
        grid["vec_omega"] = np.linspace(
            opts["omega_min"] / grid["const_mech_speed"],
            machine.omega_max / grid["const_mech_speed"],
            n_omega,
        )
    else:  # recalculated
        if dict_grid_corr is None:
            raise ValueError("dict_grid_corr must be provided for recalculated mode.")
        n_torq = len(dict_grid_corr["vec_torq"])
        n_omega = len(dict_grid_corr["vec_omega"])
        grid["vec_torq"] = dict_grid_corr["vec_torq"]
        grid["vec_omega"] = dict_grid_corr["vec_omega"]
        grid_torq_targets = dict_grid_corr["grid_torq_pirn"]

    # Allocate memory for results
    grid.update(_init_grid_arrays(n_torq, n_omega))

    # Main calculation loop
    for idx_omega in range(n_omega):
        omega_target = grid["vec_omega"][idx_omega]
        print(
            f"Calculating: Omega step {idx_omega + 1}/{n_omega} ({omega_target * grid['const_mech_speed']:.1f} RPM)"
        )

        # Synchronize transform matrices with current speed
        transform.set_omega(omega_target)

        # Find Physical Max Torque ceiling for this specific speed (voltage limited)
        _, torq_max_local, _ = optimizer.maximize_torque(transform=transform)
        grid["vec_torq_max"][0, idx_omega] = torq_max_local

        # Initial guess for the first torque step in a speed column
        vec_curr_dq_prev = np.array([1.0, 0.0, 0.0, 0.0])

        for idx_torq in range(n_torq):
            # --- A. Setup Target and Guess based on Mode ---
            if mode == "standard":
                torq_target = grid["vec_torq"][idx_torq]
                vec_curr_dq_guess = vec_curr_dq_prev

                # Boundary check: Exit if target is physically impossible at this speed
                if torq_target > torq_max_local or torq_target > torq_max_global:
                    break

            else:  # recalculated
                # Type-checker hint: We know it's not None if mode is 'recalculated'
                assert dict_grid_corr is not None
                torq_target = grid_torq_targets[idx_torq, idx_omega]
                # Use baseline current as the warm-start guess
                vec_curr_dq_guess = np.array(
                    [
                        dict_grid_corr["isd1"][idx_torq, idx_omega],
                        dict_grid_corr["isq1"][idx_torq, idx_omega],
                        dict_grid_corr["isd3"][idx_torq, idx_omega],
                        dict_grid_corr["isq3"][idx_torq, idx_omega],
                    ]
                )

                # Skip invalid or unreachable data points
                if (
                    np.isnan(torq_target)
                    or np.any(np.isnan(vec_curr_dq_guess))
                    or torq_target > torq_max_local
                ):
                    continue

            # --- B. Execute Optimization (Find minimum current for target torque) ---
            vec_curr_dq_opt, success = optimizer.minimize_current(
                torq_target=torq_target,
                transform=transform,
                vec_curr_guess=vec_curr_dq_guess,
            )

            # --- C. Handle Results and Fallbacks ---
            if mode == "standard":
                if success:
                    # Successfully solved: update warm-start for the next torque step
                    vec_curr_dq_prev = vec_curr_dq_opt
                    _fill_grid_point(
                        grid, transform, vec_curr_dq_opt, idx_torq, idx_omega, machine
                    )
            else:  # recalculated
                # Use optimized result if successful, otherwise fallback to the corrected baseline guess
                vec_curr_dq_final = vec_curr_dq_opt if success else vec_curr_dq_guess
                _fill_grid_point(
                    grid, transform, vec_curr_dq_final, idx_torq, idx_omega, machine
                )

    print(f"{mode.capitalize()} Grid Calculation Complete.")
    return grid

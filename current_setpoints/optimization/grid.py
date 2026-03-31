import numpy as np
from model.data import MachineData

def grid_to_data(grid, n_skip):
    torq = grid['vec_torq']
    omega = grid['const_mech_speed'] * grid['vec_omega']
    return MachineData(
        torq=torq, omega=omega, segments=grid['grid_segments'],
        isd1=grid['isd1'], isd3=grid['isd3'], isq1=grid['isq1'], isq3=grid['isq3'],
        n_skip=n_skip
    )

def _init_grid_arrays(n_torq, n_omega):
    keys = ['isd1', 'isd3', 'isq1', 'isq3', 'grid_curr_peak', 'grid_volt_peak', 
            'grid_curr_ang_diff', 'grid_volt_ang_diff', 'grid_segments', 'grid_volt_raw_peak', 'grid_volt_0_rms', 'grid_volt_0_peak']
    grid = {k: np.full((n_torq, n_omega), np.nan) for k in keys}
    grid['vec_torq_max'] = np.full((1, n_omega), np.nan)
    return grid

def _fill_grid_point(grid, transform, vec_curr_dq, idx_torq, idx_omega, machine):
    curr_peak, curr_ang_diff, _, volt_peak, volt_ang_diff, volt_raw_peak, volt_0_rms, volt_0_peak = transform.get_max_vals(vec_curr_dq)
    n_curr_peaks, n_volt_peaks = transform.count_peaks(vec_curr_dq, machine)
    
    grid['isd1'][idx_torq, idx_omega] = vec_curr_dq[0]
    grid['isq1'][idx_torq, idx_omega] = vec_curr_dq[1]
    grid['isd3'][idx_torq, idx_omega] = vec_curr_dq[2]
    grid['isq3'][idx_torq, idx_omega] = vec_curr_dq[3]
    grid['grid_curr_peak'][idx_torq, idx_omega] = curr_peak
    grid['grid_volt_peak'][idx_torq, idx_omega] = volt_peak
    grid['grid_curr_ang_diff'][idx_torq, idx_omega] = curr_ang_diff
    grid['grid_volt_ang_diff'][idx_torq, idx_omega] = volt_ang_diff
    grid['grid_segments'][idx_torq, idx_omega] = 3 * n_volt_peaks + n_curr_peaks


# TODO: (DONE) this is an amazing example why to unify reg_maxTorque and reg_maxTorque_PIRN_compensated into a class.
# TODO: (DONE) functions like grid_calc and grid_calc_Compensated should never appear, there should be only one function, which takes the class as an argument
def calculate_grid(optimizer, transform, opts, mode="standard", dict_grid_corr=None):
    """
    Unified grid calculation engine. 
    Modes: 
      - "standard": Generates a new grid (Baseline or Compensated) using warm-starts.
      - "recalculated": Uses an existing dictionary for targets and stateless fallbacks.
    """
    if mode not in ["standard", "recalculated"]:
        raise ValueError("Mode must be either 'standard' or 'recalculated'")
        
    print(f"Starting {mode.capitalize()} Grid Calculation...")
    
    # Grab the machine object from the unified model
    machine = optimizer.model.machine
    grid = {}
    grid['const_mech_speed'] = 30 / (np.pi * machine.n_ppairs)

    if mode == "standard":
        # Find global max torque at zero speed
        transform.set_omega(0)
        _, torq_max_global, _ = optimizer.maximize_torque(transform=transform)
        
        n_torq, n_omega = opts['n_torq'], opts['n_omega']
        grid['vec_torq'] = np.linspace(opts['torq_min'], torq_max_global, n_torq)
        grid['vec_omega'] = np.linspace(
            opts['omega_min'] / grid['const_mech_speed'], 
            machine.omega_max / grid['const_mech_speed'], 
            n_omega
        )
    else:  # recalculated
        if dict_grid_corr is None:
            raise ValueError("dict_grid_corr must be provided for recalculated mode.")
        n_torq = len(dict_grid_corr['vec_torq'])
        n_omega = len(dict_grid_corr['vec_omega'])
        grid['vec_torq'] = dict_grid_corr['vec_torq']
        grid['vec_omega'] = dict_grid_corr['vec_omega']
        grid_torq_targets = dict_grid_corr['grid_torq_pirn']

    grid.update(_init_grid_arrays(n_torq, n_omega))

    #Main calculation loop
    for idx_omega in range(n_omega):
        omega_target = grid['vec_omega'][idx_omega]
        print(f"Calculating: Omega step {idx_omega + 1}/{n_omega} ({omega_target * grid['const_mech_speed']:.1f} RPM)")
        
        transform.set_omega(omega_target)

        # Find Physical Max Torque ceiling for this specific speed
        _, torq_max_local, _ = optimizer.maximize_torque(transform=transform)
        grid['vec_torq_max'][0, idx_omega] = torq_max_local
        
        # Warm start tracker (only used in standard mode)
        vec_curr_dq_prev = np.array([1.0, 0.0, 0.0, 0.0])

        for idx_torq in range(n_torq):
            
            # --- A. Setup Target and Guess based on Mode ---
            if mode == "standard":
                torq_target = grid['vec_torq'][idx_torq]
                vec_curr_dq_guess = vec_curr_dq_prev
                
                # Stop if requested torque exceeds physical limits
                if torq_target > torq_max_local or torq_target > torq_max_global:
                    break 
                    
            else: # recalculated
                torq_target = grid_torq_targets[idx_torq, idx_omega]
                vec_curr_dq_guess = np.array([
                    dict_grid_corr['isd1'][idx_torq, idx_omega], dict_grid_corr['isq1'][idx_torq, idx_omega],
                    dict_grid_corr['isd3'][idx_torq, idx_omega], dict_grid_corr['isq3'][idx_torq, idx_omega]
                ])
                
                # Skip invalid data points but continue the loop
                if np.isnan(torq_target) or np.isnan(vec_curr_dq_guess[0]) or torq_target > torq_max_local:
                    continue

            # --- B. Execute Optimization ---
            vec_curr_dq_opt, success = optimizer.minimize_current(
                torq_target=torq_target, 
                transform=transform, 
                vec_curr_guess=vec_curr_dq_guess
            )

            # --- C. Handle Results and Fallbacks ---
            if mode == "standard":
                if success:
                    vec_curr_dq_prev = vec_curr_dq_opt # Update warm start for the next loop
                    _fill_grid_point(grid, transform, vec_curr_dq_opt, idx_torq, idx_omega, machine)
            else: # recalculated
                # Safety feature: Revert to baseline guess if optimization fails
                vec_curr_dq_final = vec_curr_dq_opt if success else vec_curr_dq_guess
                _fill_grid_point(grid, transform, vec_curr_dq_final, idx_torq, idx_omega, machine)

    print(f"{mode.capitalize()} Grid Calculation Complete.")
    return grid
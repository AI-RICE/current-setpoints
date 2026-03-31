import numpy as np
from model.data import MachineData
from .optimization import (
    reg_maxTorque, reg_defTorque, 
    reg_maxTorque_PIRN_Compensated, reg_defTorque_PIRN_Compensated
)

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

# --- BASELINE GRID CALCULATION ---
# TODO: this is an amazing example why to unify reg_maxTorque and reg_maxTorque_PIRN_compensated into a class.
# TODO: functions like grid_calc and grid_calc_Compensated should never appear, there should be only one function, which takes the class as an argument
def grid_calc(machine, transform, dict_opts):
    print("Starting Baseline Grid Calculation...")
    transform.set_omega(0)
    _, torq_max_global, _ = reg_maxTorque(machine, transform, vec_curr_dq_guess=None, opts=dict_opts['opts'])    
    
    grid = {}
    grid['const_mech_speed'] = 30 / (np.pi * machine.pp)    
    grid['vec_torq'] = np.linspace(dict_opts['torq_min'], torq_max_global, dict_opts['n_torq'])
    grid['vec_omega'] = np.linspace(
        dict_opts['omega_min'] / grid['const_mech_speed'], 
        machine.nmax / grid['const_mech_speed'], 
        dict_opts['n_omega']
    )
    
    n_torq, n_omega = dict_opts['n_torq'], dict_opts['n_omega']
    grid.update(_init_grid_arrays(n_torq, n_omega))

    for idx_omega in range(n_omega):
        omega_target = grid['vec_omega'][idx_omega]
        print(f"Calculating: Omega step {idx_omega + 1}/{n_omega} ({omega_target * grid['const_mech_speed']:.1f} RPM)") 
        
        transform.set_omega(omega_target)
        
        # 1. Find Max Torque (Ceiling)
        # The optimization function now handles multi-start internally
        _, grid['vec_torq_max'][0, idx_omega], _ = reg_maxTorque(machine, transform, vec_curr_dq_guess=None, opts=dict_opts['opts'])
        
        # Initialize warm start variable
        vec_curr_dq = np.array([1.0, 0.0, 0.0, 0.0])

        for idx_torq in range(n_torq):
            torq_target = grid['vec_torq'][idx_torq]
            
            # Stop if the requested torque is physically impossible
            if torq_target > grid['vec_torq_max'][0, idx_omega] or torq_target > torq_max_global:
                break
                
            # 2. Find Optimal Vector
            # We pass 'vec_curr_dq' (result from previous torque step) as the primary guess.
            # If it fails, reg_defTorque will automatically try MTPA and Flux-Weakening guesses.
            vec_curr_dq, success = reg_defTorque(machine, torq_target, transform, vec_curr_dq_guess=vec_curr_dq, opts=dict_opts['opts'])
            
            if success:
                _fill_grid_point(grid, transform, vec_curr_dq, idx_torq, idx_omega, machine)
            
    print("Baseline Grid Calculation Complete.")
    return grid

# --- COMPENSATED (PIRN) GRID CALCULATION ---
def grid_calc_Compensated(machine, transform, dict_opts, pirn_model, scaler, device):
    print("Starting Compensated (PIRN) Grid Calculation...")
    transform.set_omega(0)
    _, torq_max_analytical, _ = reg_maxTorque(machine, transform, vec_curr_dq_guess=None, opts=dict_opts['opts'])
    
    grid = {}
    grid['const_mech_speed'] = 30 / (np.pi * machine.pp)     
    grid['vec_torq'] = np.linspace(dict_opts['torq_min'], torq_max_analytical, dict_opts['n_torq'])
    grid['vec_omega'] = np.linspace(
        dict_opts['omega_min'] / grid['const_mech_speed'], 
        machine.nmax / grid['const_mech_speed'], 
        dict_opts['n_omega']
    )
    
    n_torq, n_omega = dict_opts['n_torq'], dict_opts['n_omega']
    grid.update(_init_grid_arrays(n_torq, n_omega))

    for idx_omega in range(n_omega):
        omega_target = grid['vec_omega'][idx_omega]
        print(f"Calculating (PIRN): Omega step {idx_omega + 1}/{n_omega} ({omega_target * grid['const_mech_speed']:.1f} RPM)")
        
        transform.set_omega(omega_target)

        # 1. Find Max Torque
        _, torq_max_local, _ = reg_maxTorque_PIRN_Compensated(
            machine, transform, pirn_model, scaler, device, omega_target, vec_curr_dq_guess=None, opts=dict_opts['opts']
        )
        grid['vec_torq_max'][0, idx_omega] = torq_max_local
        
        # Initialize warm start
        vec_curr_dq_prev = None

        for idx_torq in range(n_torq):
            torq_target = grid['vec_torq'][idx_torq]
            
            if torq_target > torq_max_local: 
                continue 

            # Use previous result as primary guess
            vec_curr_dq_guess = vec_curr_dq_prev 
            
            # 2. Find Optimal Vector
            # The optimization function handles 3-candidate retry logic internally.
            vec_curr_dq_attempt, success = reg_defTorque_PIRN_Compensated(
                machine, torq_target, transform, pirn_model, scaler, device, omega_target, vec_curr_dq_guess=vec_curr_dq_guess, opts=dict_opts['opts']
            )
            
            if success:
                vec_curr_dq_prev = vec_curr_dq_attempt
                _fill_grid_point(grid, transform, vec_curr_dq_attempt, idx_torq, idx_omega, machine)

    print("Compensated Grid Calculation Complete.")
    return grid


def grid_calc_Recalculated(machine, transform, dict_opts, pirn_model, scaler, device, dict_grid_corr):
    print("Starting Recalculated Grid Calculation (Stateless / Always Baseline)...")
    
    n_torq = len(dict_grid_corr['vec_torq'])
    n_omega = len(dict_grid_corr['vec_omega'])
    
    grid = {}
    grid['const_mech_speed'] = dict_grid_corr['const_mech_speed']
    grid['vec_torq'] = dict_grid_corr['vec_torq']
    grid['vec_omega'] = dict_grid_corr['vec_omega']
    grid.update(_init_grid_arrays(n_torq, n_omega))
    
    grid_torq_targets = dict_grid_corr['grid_torq_pirn']

    for idx_omega in range(n_omega):
        omega_target = grid['vec_omega'][idx_omega]
        print(f"Recalculating: Omega step {idx_omega + 1}/{n_omega}")
        
        transform.set_omega(omega_target)

        # 1. Find Physical Max Torque ceiling
        _, torq_max_local, _ = reg_maxTorque_PIRN_Compensated(
            machine, transform, pirn_model, scaler, device, omega_target, vec_curr_dq_guess=None, opts=dict_opts['opts']
        )
        grid['vec_torq_max'][0, idx_omega] = torq_max_local
        
        # Note: We REMOVED 'vec_curr_dq_prev' entirely.

        for idx_torq in range(n_torq):
            torq_target = grid_torq_targets[idx_torq, idx_omega]
            
            # 2. ALWAYS Start from Baseline
            vec_curr_dq_base = np.array([
                dict_grid_corr['isd1'][idx_torq, idx_omega], dict_grid_corr['isq1'][idx_torq, idx_omega],
                dict_grid_corr['isd3'][idx_torq, idx_omega], dict_grid_corr['isq3'][idx_torq, idx_omega]
            ])
            
            if np.isnan(torq_target) or np.isnan(vec_curr_dq_base[0]) or torq_target > torq_max_local: 
                continue 

            # 3. Optimization
            # We strictly use 'vec_curr_dq_base' as the guess (vec_curr_dq_guess)
            vec_curr_dq_opt, success = reg_defTorque_PIRN_Compensated(
                machine, torq_target, transform, pirn_model, scaler, device, omega_target, vec_curr_dq_guess=vec_curr_dq_base, opts=dict_opts['opts']
            )
            
            # 4. Simple Safety Check
            # If optimization fails, we revert to baseline values (Safety)
            if not success:
                vec_curr_dq_final = vec_curr_dq_base
            else:
                vec_curr_dq_final = vec_curr_dq_opt

            _fill_grid_point(grid, transform, vec_curr_dq_final, idx_torq, idx_omega, machine)

    print("Recalculated Grid Calculation Complete.")
    return grid
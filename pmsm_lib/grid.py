import numpy as np
from .data import PMSMData
from .optimization import (
    reg_maxTorque, reg_defTorque, 
    reg_maxTorque_PIRN_Compensated, reg_defTorque_PIRN_Compensated
)

def grid_to_data(grid, k_skip):
    T = grid['T']
    omega = grid['c_mech_speed'] * grid['om_vec']
    return PMSMData(
        T=T, omega=omega, segments=grid['clr'],
        isd1=grid['isd1'], isd3=grid['isd3'], isq1=grid['isq1'], isq3=grid['isq3'],
        k_skip=k_skip
    )

def _init_grid_arrays(n_T, n_om):
    keys = ['isd1', 'isd3', 'isq1', 'isq3', 'mI', 'mU', 
            'epsIdiff', 'epsUdiff', 'clr', 'mU13', 'U0rms', 'mU0']
    grid = {k: np.full((n_T, n_om), np.nan) for k in keys}
    grid['max_T'] = np.full((1, n_om), np.nan)
    return grid

def _fill_grid_point(grid, W, is_vec, k_T, k_om, IPM):
    Im, eps_diff, _, Um, beta_diff, mU13, U0rms, mU0 = W.maximal_IU(is_vec)
    peaks_I, peaks_U = W.number_of_peaks(is_vec, IPM)
    
    grid['isd1'][k_T, k_om] = is_vec[0]
    grid['isq1'][k_T, k_om] = is_vec[1]
    grid['isd3'][k_T, k_om] = is_vec[2]
    grid['isq3'][k_T, k_om] = is_vec[3]
    grid['mI'][k_T, k_om] = Im
    grid['mU'][k_T, k_om] = Um
    grid['epsIdiff'][k_T, k_om] = eps_diff
    grid['epsUdiff'][k_T, k_om] = beta_diff
    grid['clr'][k_T, k_om] = 3 * peaks_U + peaks_I

# --- BASELINE GRID CALCULATION ---
def grid_calc(IPM, W, calc_opt):
    print("Starting Baseline Grid Calculation...")
    W.change_om(0)
    _, max_T_global, _ = reg_maxTorque(IPM, W, is0=None, opts=calc_opt['opts'])    
    
    grid = {}
    grid['c_mech_speed'] = 30 / (np.pi * IPM.pp)    
    grid['T'] = np.linspace(calc_opt['min_T'], max_T_global, calc_opt['n_T'])
    grid['om_vec'] = np.linspace(
        calc_opt['min_om'] / grid['c_mech_speed'], 
        IPM.nmax / grid['c_mech_speed'], 
        calc_opt['n_om']
    )
    
    n_T, n_om = calc_opt['n_T'], calc_opt['n_om']
    grid.update(_init_grid_arrays(n_T, n_om))

    for k_om in range(n_om):
        omega_val = grid['om_vec'][k_om]
        print(f"Calculating: Omega step {k_om + 1}/{n_om} ({omega_val * grid['c_mech_speed']:.1f} RPM)") 
        
        W.change_om(omega_val)
        
        # 1. Find Max Torque (Ceiling)
        # The optimization function now handles multi-start internally
        _, grid['max_T'][0, k_om], _ = reg_maxTorque(IPM, W, is0=None, opts=calc_opt['opts'])
        
        # Initialize warm start variable
        is_vec = np.array([1.0, 0.0, 0.0, 0.0])

        for k_T in range(n_T):
            torque_val = grid['T'][k_T]
            
            # Stop if the requested torque is physically impossible
            if torque_val > grid['max_T'][0, k_om] or torque_val > max_T_global:
                break
                
            # 2. Find Optimal Vector
            # We pass 'is_vec' (result from previous torque step) as the primary guess.
            # If it fails, reg_defTorque will automatically try MTPA and Flux-Weakening guesses.
            is_vec, success = reg_defTorque(IPM, torque_val, W, is0=is_vec, opts=calc_opt['opts'])
            
            if success:
                _fill_grid_point(grid, W, is_vec, k_T, k_om, IPM)
            
    print("Baseline Grid Calculation Complete.")
    return grid

# --- COMPENSATED (PIRN) GRID CALCULATION ---
def grid_calc_Compensated(IPM, W, calc_opt, pirn_model, scaler, device):
    print("Starting Compensated (PIRN) Grid Calculation...")
    W.change_om(0)
    _, max_T_analytical, _ = reg_maxTorque(IPM, W, is0=None, opts=calc_opt['opts'])
    
    grid = {}
    grid['c_mech_speed'] = 30 / (np.pi * IPM.pp)     
    grid['T'] = np.linspace(calc_opt['min_T'], max_T_analytical, calc_opt['n_T'])
    grid['om_vec'] = np.linspace(
        calc_opt['min_om'] / grid['c_mech_speed'], 
        IPM.nmax / grid['c_mech_speed'], 
        calc_opt['n_om']
    )
    
    n_T, n_om = calc_opt['n_T'], calc_opt['n_om']
    grid.update(_init_grid_arrays(n_T, n_om))

    for k_om in range(n_om):
        omega_val = grid['om_vec'][k_om]
        print(f"Calculating (PIRN): Omega step {k_om + 1}/{n_om} ({omega_val * grid['c_mech_speed']:.1f} RPM)")
        
        W.change_om(omega_val)

        # 1. Find Max Torque
        _, local_max_T, _ = reg_maxTorque_PIRN_Compensated(
            IPM, W, pirn_model, scaler, device, omega_val, is0=None, opts=calc_opt['opts']
        )
        grid['max_T'][0, k_om] = local_max_T
        
        # Initialize warm start
        previous_is_vec = None

        for k_T in range(n_T):
            torque_val = grid['T'][k_T]
            
            if torque_val > local_max_T: 
                continue 

            # Use previous result as primary guess
            is0_guess = previous_is_vec 
            
            # 2. Find Optimal Vector
            # The optimization function handles 3-candidate retry logic internally.
            is_vec_attempt, exitflag = reg_defTorque_PIRN_Compensated(
                IPM, torque_val, W, pirn_model, scaler, device, omega_val, is0=is0_guess, opts=calc_opt['opts']
            )
            
            if exitflag:
                previous_is_vec = is_vec_attempt
                _fill_grid_point(grid, W, is_vec_attempt, k_T, k_om, IPM)

    print("Compensated Grid Calculation Complete.")
    return grid


def grid_calc_Recalculated(IPM, W, calc_opt, pirn_model, scaler, device, corr_grid):
    print("Starting Recalculated Grid Calculation (Stateless / Always Baseline)...")
    
    n_T = len(corr_grid['T'])
    n_om = len(corr_grid['om_vec'])
    
    grid = {}
    grid['c_mech_speed'] = corr_grid['c_mech_speed']
    grid['T'] = corr_grid['T']
    grid['om_vec'] = corr_grid['om_vec']
    grid.update(_init_grid_arrays(n_T, n_om))
    
    T_targets = corr_grid['T_pirn_matrix']

    for k_om in range(n_om):
        omega_val = grid['om_vec'][k_om]
        print(f"Recalculating: Omega step {k_om + 1}/{n_om}")
        
        W.change_om(omega_val)

        # 1. Find Physical Max Torque ceiling
        _, local_max_T, _ = reg_maxTorque_PIRN_Compensated(
            IPM, W, pirn_model, scaler, device, omega_val, is0=None, opts=calc_opt['opts']
        )
        grid['max_T'][0, k_om] = local_max_T
        
        # Note: We REMOVED 'previous_is_vec' entirely.

        for k_T in range(n_T):
            torque_val = T_targets[k_T, k_om]
            
            # 2. ALWAYS Start from Baseline
            is_base = np.array([
                corr_grid['isd1'][k_T, k_om], corr_grid['isq1'][k_T, k_om],
                corr_grid['isd3'][k_T, k_om], corr_grid['isq3'][k_T, k_om]
            ])
            
            if np.isnan(torque_val) or np.isnan(is_base[0]) or torque_val > local_max_T: 
                continue 

            # 3. Optimization
            # We strictly use 'is_base' as the guess (is0)
            is_opt, exitflag = reg_defTorque_PIRN_Compensated(
                IPM, torque_val, W, pirn_model, scaler, device, omega_val, is0=is_base, opts=calc_opt['opts']
            )
            
            # 4. Simple Safety Check
            # If optimization fails, we revert to baseline values (Safety)
            if not exitflag:
                final_is = is_base
            else:
                final_is = is_opt

            _fill_grid_point(grid, W, final_is, k_T, k_om, IPM)

    print("Recalculated Grid Calculation Complete.")
    return grid
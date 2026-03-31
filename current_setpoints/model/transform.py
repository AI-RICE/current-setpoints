import numpy as np


# TODO: we need to think whether we want to work for 5 phases or more general. i would prefer the second option. that would require lots of changes though.
class Transform:
    # TODO: (DONE) use the same name for P and IPM. is it the best name?
    def __init__(self, machine, omega, add_volt_0, n_theta=700):
        self.n_phases = machine.n_phases
        self.add_volt_0 = add_volt_0  
        
        # Validate theta resolution
        if n_theta <= 0:
            raise ValueError("n_theta must be positive")
            
        n_theta = round(n_theta / (2 * machine.n_phases)) * 2 * machine.n_phases        
        self.vec_theta = np.linspace(0, 2 * np.pi, n_theta + 1)
        
        self.compute_matrices_init(machine)
        self.compute_matrices(omega)

    def compute_matrices_init(self, machine):
        self.mat_curr_dq_to_volt_dq_fixed = machine.R_stat @ np.eye(4)
        self.mat_curr_dq_to_volt_dq_omega = machine.mat_crossc @ machine.L_stat
        self.vec_volt_bemf_dq_omega = machine.mat_crossc @ machine.flux_volt        
        
        cos_theta = np.cos(self.vec_theta)
        sin_theta = np.sin(self.vec_theta)
        cos_3theta = np.cos(3 * self.vec_theta)
        sin_3theta = np.sin(3 * self.vec_theta)        
        self.mat_dq_to_ph = np.array([cos_theta, -sin_theta, cos_3theta, -sin_3theta]).T

    def compute_matrices(self, omega):
        self.omega = omega
        self.mat_curr_dq_to_volt_dq = self.mat_curr_dq_to_volt_dq_fixed + omega * self.mat_curr_dq_to_volt_dq_omega
        self.vec_volt_bemf_dq = omega * self.vec_volt_bemf_dq_omega        
        self.mat_curr_dq_to_volt_ph = self.mat_dq_to_ph @ self.mat_curr_dq_to_volt_dq
        self.vec_volt_bemf_ph = self.mat_dq_to_ph @ self.vec_volt_bemf_dq

    def set_omega(self, omega):
        if not np.isscalar(omega):
            raise TypeError(f"Omega must be a scalar, got {type(omega)}")
        self.compute_matrices(omega)
    
    def get_curr_ph(self, vec_curr_dq): 
        if len(vec_curr_dq) != 4:
            raise ValueError("Current vector must be length 4 (id1, iq1, id3, iq3)")
        return self.mat_dq_to_ph @ vec_curr_dq

    def get_volt_dq(self, vec_curr_dq):
        return self.mat_curr_dq_to_volt_dq @ vec_curr_dq + self.vec_volt_bemf_dq

    def get_volt_ph(self, vec_curr_dq):
        if vec_curr_dq.ndim != 1 or len(vec_curr_dq) != 4:
             # Critical check for optimization loops
             raise ValueError(f"Input current vector shape mismatch: {vec_curr_dq.shape}")

        vec_volt_raw = self.mat_curr_dq_to_volt_ph @ vec_curr_dq + self.vec_volt_bemf_ph
        if self.add_volt_0:          
            mat_volt_res = vec_volt_raw[:-1].reshape(-1, self.n_phases) 
            vec_volt_0 = -0.5 * (np.min(mat_volt_res, axis=1) + np.max(mat_volt_res, axis=1))            
            vec_volt_ph = mat_volt_res          
            vec_volt_ph = vec_volt_ph.flatten()
            vec_volt_ph = np.append(vec_volt_ph, vec_volt_ph[0])            
            vec_volt_0 = np.repeat(vec_volt_0, self.n_phases)
            vec_volt_0 = np.append(vec_volt_0, vec_volt_0[0])
        else:
            vec_volt_ph = vec_volt_raw
            vec_volt_0 = np.zeros_like(vec_volt_raw)
        return vec_volt_ph, vec_volt_0, vec_volt_raw

    def get_max_vals(self, vec_curr_dq):
        vec_volt_ph, vec_volt_0, vec_volt_raw = self.get_volt_ph(vec_curr_dq)        
        curr_peak = np.max(np.abs(self.get_curr_ph(vec_curr_dq)))
        volt_peak = np.max(np.abs(vec_volt_ph))
        volt_raw_peak = np.max(np.abs(vec_volt_raw))        
        volt_0_rms = np.sqrt(np.mean(vec_volt_0**2))
        volt_0_peak = np.max(np.abs(vec_volt_0))        
        vec_volt_dq = self.get_volt_dq(vec_curr_dq)        
        
        # Phase calculations
        curr_ang_1 = np.arctan2(vec_curr_dq[1], vec_curr_dq[0])
        curr_ang_3 = np.arctan2(vec_curr_dq[3], vec_curr_dq[2])
        curr_ang_diff = np.abs(np.mod(curr_ang_1 * 3 + np.pi, 2 * np.pi) - np.mod(curr_ang_3, 2 * np.pi))
        
        volt_ang_1 = np.arctan2(vec_volt_dq[1], vec_volt_dq[0])
        volt_ang_3 = np.arctan2(vec_volt_dq[3], vec_volt_dq[2])
        volt_ang_diff = np.abs(np.mod(volt_ang_1 * 3 + np.pi, 2 * np.pi) - np.mod(volt_ang_3, 2 * np.pi))
        
        return curr_peak, curr_ang_diff, vec_volt_dq, volt_peak, volt_ang_diff, volt_raw_peak, volt_0_rms, volt_0_peak

    def count_peaks(self, vec_curr_dq, machine, tol=1e-4):
        curr_peak, curr_ang_diff, _, volt_peak, volt_ang_diff, _, _, _ = self.get_max_vals(vec_curr_dq)        
        n_curr_peaks = self._count_peaks_helper(curr_peak, machine.curr_max, curr_ang_diff, tol)
        n_volt_peaks = self._count_peaks_helper(volt_peak, machine.volt_max, volt_ang_diff, tol)
        return n_curr_peaks, n_volt_peaks

    def _count_peaks_helper(self, value, val_max, angle_diff, tol):
        if np.abs(value - val_max) < tol:
            if angle_diff < tol * val_max:
                return 2
            else:
                return 1
        else:
            return 0
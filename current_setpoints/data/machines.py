import numpy as np

class BaseMachine:
    def __init__(self, n_phases, n_ppairs):
        self.n_phases = n_phases          
        self.n_ppairs = n_ppairs        
        self.k_phase = n_phases / 2 
        self.mat_crossc = np.array([ 
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, -3],
            [0, 0, 3, 0]
        ])

    def set_max_pars(self, curr_max, volt_max, omg_max):        
        self.curr_max = curr_max
        self.volt_max = volt_max
        self.omg_max = omg_max

    def check_data(self):
        self._check_matrix(self.mat_A, False, 'mat_A')
        self._check_matrix(self.L_stat, False, 'L_stat')
        self._check_matrix(self.R_stat, True, 'R_stat')
        self._check_vector(self.vec_b, 'vec_b')

    def _check_matrix(self, mat, allow_scalar, name):
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]: 
            raise ValueError(f"Matrix {name} must be square.")
        if not allow_scalar and mat.shape[0] != self.n_phases - 1:
            raise ValueError(f"Matrix {name} must be square and sized to n_phases-1 ({self.n_phases - 1}).")

    def _check_vector(self, vec, name):
        if vec.ndim > 1 and vec.shape[1] != 1:
            raise ValueError(f"Vector {name} must be a column vector.")
        if vec.shape[0] != self.n_phases - 1:
            raise ValueError(f"Vector {name} must be a column vector with length n_phases-1 ({self.n_phases - 1}).")


# TODO: (DONE) normal names
class GenericMachine(BaseMachine):
    def __init__(self, n_phases, n_ppairs, R_stat_vec, L_stat, flux_volt_vec, flux_torq_vec):
        super().__init__(n_phases, n_ppairs)
        
        R_stat_vec = np.array(R_stat_vec).flatten() 
        flux_volt_vec = np.array(flux_volt_vec).flatten()
        flux_torq_vec = np.array(flux_torq_vec).flatten()
        L_stat = np.array(L_stat)

        self.flux_volt = flux_volt_vec  
        self.flux_torq = flux_torq_vec
        self.R_stat = np.diag(R_stat_vec)
        self.L_stat = L_stat
        self.mat_A = np.zeros((4, 4))        
        
        self.vec_b = (self.n_phases * self.n_ppairs / 4) * (self.mat_crossc @ self.flux_torq)                        
        self.check_data()


class IEEEMachine2(GenericMachine):
    def __init__(self):
        n_phases = 5 
        n_ppairs = 8 
        R_stat_vec = [0.0191, 0.0514, 0.0805, 0.0801]         
        L_stat = np.array([ 
            [ 0.0920, -0.0286, -0.0141,  0.0010],
            [-0.0133,  0.1090, -0.0008, -0.0092],
            [-0.0088,  0.0037,  0.0725, -0.0466],
            [-0.0041, -0.0053,  0.0475,  0.0722]
        ]) * 1e-3      
        flux_volt_vec = [0.0115, 0.0018, 0, 0] 
        flux_torq_vec = [1.12810358e-02, -6.28421072e-04, 1.55053034e-04, -4.81016476e-05] 
        
        super().__init__(n_phases, n_ppairs, R_stat_vec, L_stat, flux_volt_vec, flux_torq_vec)
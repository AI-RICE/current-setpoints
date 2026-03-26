import numpy as np

class PMSMBase:
    def __init__(self, m, pp):
        self.m = m          
        self.pp = pp        
        self.kp = m / 2 
        self.J = np.array([ 
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, -3],
            [0, 0, 3, 0]
        ])

    def set_max_pars(self, Imax, Umax, nmax):        
        self.Imax = Imax
        self.Umax = Umax
        self.nmax = nmax

    def check_data(self):
        self._check_matrix(self.A, False, 'A')
        self._check_matrix(self.L, False, 'L')
        self._check_matrix(self.Rs, True, 'Rs')
        self._check_vector(self.b, 'b')

    def _check_matrix(self, A, scalar_allowed, name):
        if A.ndim != 2 or A.shape[0] != A.shape[1]: 
            raise ValueError(f"Matrix {name} must be square.")
        if not scalar_allowed and A.shape[0] != self.m - 1:
            raise ValueError(f"Matrix {name} must be square and sized to m-1 ({self.m - 1}).")

    def _check_vector(self, a, name):
        if a.ndim > 1 and a.shape[1] != 1:
            raise ValueError(f"Vector {name} must be a column vector.")
        if a.shape[0] != self.m - 1:
            raise ValueError(f"Vector {name} must be a column vector with length m-1 ({self.m - 1}).")


# TODO: normal names
class PMSMBase2(PMSMBase):
    def __init__(self, m, pp, Rs_vec, L, Psi_vec, TPsi_vec):
        super().__init__(m, pp)
        
        Rs_vec = np.array(Rs_vec).flatten() 
        Psi_vec = np.array(Psi_vec).flatten()
        TPsi_vec = np.array(TPsi_vec).flatten()
        L = np.array(L)

        self.Psi = Psi_vec  
        self.TPsi = TPsi_vec
        self.Rs = np.diag(Rs_vec)
        self.L = L
        self.A = np.zeros((4, 4))        
        
        self.b = (self.m * self.pp / 4) * (self.J @ self.TPsi)                        
        self.check_data()


class PMSM_IEEETIE_machine2(PMSMBase2):
    def __init__(self):
        m = 5 
        pp = 8 
        Rs_vec = [0.0191, 0.0514, 0.0805, 0.0801]         
        L_base = np.array([ 
            [ 0.0920, -0.0286, -0.0141,  0.0010],
            [-0.0133,  0.1090, -0.0008, -0.0092],
            [-0.0088,  0.0037,  0.0725, -0.0466],
            [-0.0041, -0.0053,  0.0475,  0.0722]
        ])
        L = L_base * 1e-3         
        Psi_vec = [0.0115, 0.0018, 0, 0] 
        TPsi_vec = [1.12810358e-02, -6.28421072e-04, 1.55053034e-04, -4.81016476e-05] 
        #TPsi_vec = [0.0122, 0.0005, 0.0001, 0]
        
        super().__init__(m, pp, Rs_vec, L, Psi_vec, TPsi_vec)
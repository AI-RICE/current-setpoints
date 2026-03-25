import numpy as np
import sys

class Transform:
    def __init__(self, P, om, add_u0, n_theta=700):
        self.n_phases = P.m
        self.add_u0 = add_u0  
        
        # Validate theta resolution
        if n_theta <= 0:
            raise ValueError("n_theta must be positive")
            
        n_theta = round(n_theta / (2 * P.m)) * 2 * P.m         
        self.th = np.linspace(0, 2 * np.pi, n_theta + 1)
        
        self.compute_matrices_init(P)
        self.compute_matrices(om)

    def compute_matrices_init(self, P):
        self.matrix_i_to_u_fixed0 = P.Rs @ np.eye(4)
        self.matrix_i_to_u_fixed0_om = P.J @ P.L
        self.vector_i_to_u0_om = P.J @ P.Psi        
        
        cos_th = np.cos(self.th)
        sin_th = np.sin(self.th)
        cos_3th = np.cos(3 * self.th)
        sin_3th = np.sin(3 * self.th)        
        self.matrix_s_to_a = np.array([cos_th, -sin_th, cos_3th, -sin_3th]).T

    def compute_matrices(self, om):
        self.om = om
        self.matrix_i_to_u = self.matrix_i_to_u_fixed0 + om * self.matrix_i_to_u_fixed0_om
        self.vector_i_to_u = om * self.vector_i_to_u0_om        
        self.matrix_is_to_ua = self.matrix_s_to_a @ self.matrix_i_to_u
        self.vector_is_to_ua = self.matrix_s_to_a @ self.vector_i_to_u

    def change_om(self, om):
        if not np.isscalar(om):
            raise TypeError(f"Omega must be a scalar, got {type(om)}")
        self.compute_matrices(om)
    
    def is_to_ia(self, is_vec):       
        if len(is_vec) != 4:
            raise ValueError("Current vector must be length 4 (id1, iq1, id3, iq3)")
        return self.matrix_s_to_a @ is_vec

    def is_to_us(self, is_vec):
        return self.matrix_i_to_u @ is_vec + self.vector_i_to_u

    def is_to_ua(self, is_vec):
        if is_vec.ndim != 1 or len(is_vec) != 4:
             # Critical check for optimization loops
             raise ValueError(f"Input current vector shape mismatch: {is_vec.shape}")

        Ua13 = self.matrix_is_to_ua @ is_vec + self.vector_is_to_ua
        if self.add_u0:           
            Uf = Ua13[:-1].reshape(-1, self.n_phases) 
            U0 = -0.5 * (np.min(Uf, axis=1) + np.max(Uf, axis=1))            
            Ua = Uf           
            Ua = Ua.flatten()
            Ua = np.append(Ua, Ua[0])             
            U0 = np.repeat(U0, self.n_phases)
            U0 = np.append(U0, U0[0])
        else:
            Ua = Ua13
            U0 = np.zeros_like(Ua13)
        return Ua, U0, Ua13

    def maximal_IU(self, is_vec):
        Ua, U0, Ua13 = self.is_to_ua(is_vec)        
        mI = np.max(np.abs(self.is_to_ia(is_vec)))
        mU = np.max(np.abs(Ua))
        mU13 = np.max(np.abs(Ua13))        
        U0rms = np.sqrt(np.mean(U0**2))
        mU0 = np.max(np.abs(U0))        
        U = self.is_to_us(is_vec)        
        
        # Phase calculations
        eps1 = np.arctan2(is_vec[1], is_vec[0])
        eps3 = np.arctan2(is_vec[3], is_vec[2])
        eps_diff = np.abs(np.mod(eps1 * 3 + np.pi, 2 * np.pi) - np.mod(eps3, 2 * np.pi))
        
        beta1 = np.arctan2(U[1], U[0])
        beta3 = np.arctan2(U[3], U[2])
        beta_diff = np.abs(np.mod(beta1 * 3 + np.pi, 2 * np.pi) - np.mod(beta3, 2 * np.pi))
        
        return mI, eps_diff, U, mU, beta_diff, mU13, U0rms, mU0

    def number_of_peaks(self, is_vec, IPM, tol=1e-4):
        mI, eps_diff, _, mU, beta_diff, _, _, _ = self.maximal_IU(is_vec)        
        n_I = self._number_of_peaks_helper(mI, IPM.Imax, eps_diff, tol)
        n_U = self._number_of_peaks_helper(mU, IPM.Umax, beta_diff, tol)
        return n_I, n_U

    def _number_of_peaks_helper(self, value, value_max, eps, tol):
        if np.abs(value - value_max) < tol:
            if eps < tol * value_max:
                return 2
            else:
                return 1
        else:
            return 0
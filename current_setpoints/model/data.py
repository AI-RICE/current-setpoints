import numpy as np
import warnings


class MachineData:
    def __init__(self, torq, omega, segments, isd1, isd3, isq1, isq3, k_skip=None):
        self.torq = np.array(torq).flatten()
        self.omega = np.array(omega).flatten()
        self.segments = np.array(segments)
        self.isd1 = np.array(isd1)
        self.isd3 = np.array(isd3)
        self.isq1 = np.array(isq1)
        self.isq3 = np.array(isq3)

        # 1. Check Dimensions
        self._check_dimensions()

        # TODO: (DONE) remove all the Matlab indexing artefacts        
        if k_skip is not None and k_skip > 1:
            self.select_k(k_skip)

        self.torq_grid, self.omega_grid = np.meshgrid(self.torq, self.omega, indexing='ij')    
        self.unique_segments = np.unique(self.segments[~np.isnan(self.segments)]).astype(int)    

        self.compute_data()

    def _check_dimensions(self):
        """
        Ensures all input arrays are compatible with torq and omega.
        """
        n_torq = len(self.torq)
        n_omega = len(self.omega)
        
        expected_shape = (n_torq, n_omega)
        
        if self.segments.shape != expected_shape:
            # TODO: (DONE) never ever use this. use raise instead. this is horrible for tests, when it crashes Python instead of a desired error.
            raise ValueError(f"Segments shape {self.segments.shape} != expected {expected_shape}")
             
        if self.isd1.shape != expected_shape:
            raise ValueError("Current component dimensions do not match grid (torq x omega).")

        # Sanity check on values (optional but recommended)
        # TODO: (DONE) should it be like this? Is negative speed ok? YES
        if np.any(self.omega < 0):
            warnings.warn(
                "Negative speeds detected in grid data. Ensure reverse rotation is intended.", 
                UserWarning
            )

    def select_k(self, k_skip):
        # TODO: (DONE) why is this here?
        self.omega = self.omega[::k_skip]
        self.torq = self.torq[::k_skip]
        
        self.segments = self.segments[::k_skip, ::k_skip]
        self.isd1 = self.isd1[::k_skip, ::k_skip]
        self.isd3 = self.isd3[::k_skip, ::k_skip]
        self.isq1 = self.isq1[::k_skip, ::k_skip]
        self.isq3 = self.isq3[::k_skip, ::k_skip]
        
import numpy as np
import sys


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

        # TODO: remove all the Matlab indexing artefacts
        self.i_to_seg = {}
        self.n_seg = 0
        self.rows = {}
        self.cols = {}
        
        if k_skip is not None:
            self.select_k(k_skip)
            
        self.compute_data()

    def _check_dimensions(self):
        """
        Ensures all input arrays are compatible with torq and omega.
        """
        n_torq = len(self.torq)
        n_omega = len(self.omega)
        
        expected_shape = (n_torq, n_omega)
        
        if self.segments.shape != expected_shape:
             print(f"Error: Segments shape {self.segments.shape} != expected {expected_shape}")
             # TODO: never ever use this. use raise instead. this is horrible for tests, when it crashes Python instead of a desired error.
             sys.exit()
             
        if self.isd1.shape != expected_shape:
             print("Error: Current component dimensions do not match grid (torq x omega).")
             sys.exit()

        # Sanity check on values (optional but recommended)
        # TODO: should it be like this? Is negative speed ok?
        if np.any(self.omega < 0):
             print("Warning: Negative speeds detected in grid data.")

    def select_k(self, k_skip):
        # TODO: why is this here?
        if k_skip < 1: return
        i_torq = np.arange(0, len(self.torq), k_skip)
        i_omega = np.arange(0, len(self.omega), k_skip)
        
        self.omega = self.omega[i_omega]
        self.torq = self.torq[i_torq]
        
        # Use np.ix_ for clean slicing of 2D arrays
        mesh = np.ix_(i_torq, i_omega)
        self.segments = self.segments[mesh]
        self.isd1 = self.isd1[mesh]
        self.isd3 = self.isd3[mesh]
        self.isq1 = self.isq1[mesh]
        self.isq3 = self.isq3[mesh]

    def compute_data(self):
        # Filter NaNs from unique segments check
        segs = np.unique(self.segments[~np.isnan(self.segments)])
        segs = np.sort(segs).astype(int)
        
        self.n_seg = len(segs)
        self.i_to_seg = {i + 1: seg for i, seg in enumerate(segs)}
        
        for i_seg in range(1, self.n_seg + 1):
            seg_val = self.i_to_seg[i_seg]
            # Boolean mask for this segment
            idx = (self.segments == seg_val)
            row, col = np.where(idx)
            
            if len(row) == 0:
                print(f"Warning: Segment {seg_val} exists but has no data points.")
                
            self.rows[i_seg] = row
            self.cols[i_seg] = col
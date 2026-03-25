import os
import numpy as np
import pandas as pd
import sys

# TODO: move somewhere else. is it going to be used for all data? if yes, move it to utils. if not, move it to ../../notebooks
def load_aggregated_csv_data(file_path, col_map):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Aggregated CSV file not found at: {file_path}")

    data = pd.read_csv(file_path)
    csv_names = col_map.keys()
    
    # Validation: Missing Columns
    missing_cols = [col for col in csv_names if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    data.rename(columns=col_map, inplace=True)
    required_cols = ['omega', 'id1', 'iq1', 'id3', 'iq3', 'T_measured']
    
    # Validation: NaNs
    if data[required_cols].isnull().values.any():
        print("Warning: NaNs found in required columns. Dropping invalid rows.")
        data.dropna(subset=required_cols, inplace=True)
        
    print(f"Loaded {len(data)} valid data points from CSV.")
    return data

class PMSMData:
    def __init__(self, T, omega, segments, isd1, isd3, isq1, isq3, k_skip=None):
        self.T = np.array(T).flatten()
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
        Ensures all input arrays are compatible with T and Omega.
        """
        nT = len(self.T)
        nOm = len(self.omega)
        
        expected_shape = (nT, nOm)
        
        if self.segments.shape != expected_shape:
             print(f"Error: Segments shape {self.segments.shape} != expected {expected_shape}")
             # TODO: never ever use this. use raise instead. this is horrible for tests, when it crashes Python instead of a desired error.
             sys.exit()
             
        if self.isd1.shape != expected_shape:
             print("Error: Current component dimensions do not match grid (T x Omega).")
             sys.exit()

        # Sanity check on values (optional but recommended)
        # TODO: should it be like this? Is negative speed ok?
        if np.any(self.omega < 0):
             print("Warning: Negative speeds detected in grid data.")

    def select_k(self, k_skip):
        # TODO: why is this here?
        if k_skip < 1: return
        i_T = np.arange(0, len(self.T), k_skip)
        i_omega = np.arange(0, len(self.omega), k_skip)
        
        self.omega = self.omega[i_omega]
        self.T = self.T[i_T]
        
        # Use np.ix_ for clean slicing of 2D arrays
        mesh = np.ix_(i_T, i_omega)
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
"""
Aggregate raw per-setpoint MATLAB measurements into a single means CSV.

Each ``.mat`` file under ``data/250416_mereni/`` holds a ``meas`` struct with
time-series fields for one operating setpoint. For every file this script
extracts the requested variables, averages each over its samples, and writes
one row per file to ``data/aggregated_file_means.csv``.

Paths are resolved relative to the repository root (via ``__file__``), so the
script runs correctly from any working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

# Repo root is the parent of this script's ``scripts/`` directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER_PATH = REPO_ROOT / "data" / "250416_mereni"
OUTPUT_FILE = REPO_ROOT / "data" / "aggregated_file_means.csv"

# uq1 was missing from the original aggregation; it is included here so the
# full fundamental + 3rd-harmonic dq voltage set (ud1, uq1, ud3, uq3) is
# available downstream.
VARIABLES_TO_PROCESS = [
    "omega", "theta", "ia", "ua",
    "id1", "id3", "iq1", "iq3",
    "ud1", "uq1", "ud3", "uq3",
    "temp1", "temp2", "torq",
]
DATA_STRUCTURE_NAME = "meas"


def recursive_extract(data):
    """Recursively unwraps a scalar or final array from nested MATLAB structures."""
    if isinstance(data, np.ndarray) and data.size == 1:
        return recursive_extract(data.item())
    elif isinstance(data, np.ndarray) and data.ndim > 1:
        return data.flatten()
    else:
        return data


def extract_and_calculate_means(mat_data, data_struct_name, variables):
    """Extracts the requested variables from the struct and returns their means."""
    file_means = {}

    if data_struct_name not in mat_data:
        print(f"  Warning: '{data_struct_name}' structure not found in file.")
        return None

    try:
        meas_struct = mat_data[data_struct_name].flat[0]

        for var in variables:
            if var in meas_struct.dtype.names:
                raw_data = recursive_extract(meas_struct[var])
                numeric_data = pd.to_numeric(raw_data, errors="coerce")
                file_means[var] = numeric_data.mean()
            else:
                file_means[var] = np.nan
    except Exception as e:
        print(f"  Error processing data structure: {e}")
        return None

    return file_means


def process_matlab_files(folder_path, variables, data_struct_name, output_file):
    """Loops through every .mat file, averages each variable, and writes the CSV."""
    file_paths = sorted(Path(folder_path).glob("*.mat"))
    if not file_paths:
        print(f"No .mat files found in: {folder_path}")
        return

    all_file_means = []

    print(f"Found {len(file_paths)} files. Starting mean calculation...")

    for i, file_path in enumerate(file_paths):
        file_name = file_path.name
        print(f"Processing ({i + 1}/{len(file_paths)}): {file_name}")

        try:
            mat_data = loadmat(file_path, squeeze_me=False)
            means = extract_and_calculate_means(mat_data, data_struct_name, variables)
            if means is not None:
                means["File_Name"] = file_name
                all_file_means.append(means)
        except Exception as e:
            print(f"  CRITICAL ERROR loading/processing {file_name}: {e}")

    if not all_file_means:
        print("No successful data aggregation. Output file not created.")
        return

    results_df = pd.DataFrame(all_file_means)
    results_df = results_df[["File_Name"] + variables]

    results_df.to_csv(output_file, index=False)

    print("\n--- Aggregation Complete ---")
    print(f"Successfully processed {len(results_df)} files.")
    print(f"Results saved to: {os.path.abspath(output_file)}")

    # Surface any column that came out (partly) NaN, so a missing field such as
    # a mistyped variable name is caught here rather than silently dropping
    # every row when the CSV is later loaded.
    nan_counts = results_df[variables].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print("\nWARNING: NaNs present in aggregated columns:")
        for col, n in nan_cols.items():
            print(f"  {col}: {int(n)}/{len(results_df)} rows NaN")


if __name__ == "__main__":
    process_matlab_files(
        DATA_FOLDER_PATH,
        VARIABLES_TO_PROCESS,
        DATA_STRUCTURE_NAME,
        OUTPUT_FILE,
    )

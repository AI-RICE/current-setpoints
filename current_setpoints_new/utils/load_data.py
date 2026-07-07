"""
CSV data loading utility for aggregated motor measurement datasets.

Functions
---------
load_aggregated_csv_data — load, validate, and rename a motor measurement CSV.
"""
from __future__ import annotations

import os

import pandas as pd


def load_aggregated_csv_data(file_path: str, col_map: dict[str, str]) -> pd.DataFrame:
    """
    Load, validate, and rename motor data from a CSV file.

    Checks file existence, ensures all required mapping columns are present,
    renames columns for internal consistency, and drops rows with NaNs in any
    mapped column.

    Parameters
    ----------
    file_path : str
        Path to the target CSV file.
    col_map : dict[str, str]
        Mapping from CSV column names to internal names
        (e.g. {"Measured_Torque": "torq_meas"}).

    Returns
    -------
    pd.DataFrame — cleaned and renamed dataset

    Raises
    ------
    FileNotFoundError if the file does not exist.
    ValueError if columns in col_map are missing from the CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Aggregated CSV file not found at: {file_path}")

    data = pd.read_csv(file_path)

    missing_cols = [col for col in col_map if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    data = data.rename(columns=col_map)
    required_cols = list(col_map.values())

    if data[required_cols].isnull().values.any():
        n_before = len(data)
        data = data.dropna(subset=required_cols)
        print(f"Warning: dropped {n_before - len(data)} rows with NaNs in mapped columns.")

    print(f"Loaded {len(data)} valid data points from CSV.")
    return data

import os
import pandas as pd
from typing import Dict


def load_aggregated_csv_data(file_path: str, col_map: Dict[str, str]) -> pd.DataFrame:
    """
    Loads, validates, and renames motor data from a CSV file.

    Checks for file existence, ensures all required mapping columns are present,
    renames columns for internal consistency, and handles missing values dynamically
    for any n-phase machine.

    Args:
        file_path: System path to the target CSV file.
        col_map: Dictionary mapping CSV column names to internal names
            (e.g., {'Measured_Torque': 'torq_meas'}).

    Returns:
        pd.DataFrame: Cleaned and renamed motor dataset.

    Raises:
        FileNotFoundError: If the file does not exist at the provided path.
        ValueError: If columns defined in col_map are missing from the CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Aggregated CSV file not found at: {file_path}")

    data: pd.DataFrame = pd.read_csv(file_path)
    csv_names = col_map.keys()

    # Validation: Missing Columns
    missing_cols = [col for col in csv_names if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    data.rename(columns=col_map, inplace=True)
    
    # Dynamically define required columns based on the provided mapping
    required_cols = list(col_map.values())

    # Validation: NaNs
    if data[required_cols].isnull().values.any():
        print("Warning: NaNs found in mapped columns. Dropping invalid rows.")
        data.dropna(subset=required_cols, inplace=True)

    print(f"Loaded {len(data)} valid data points from CSV.")
    return data
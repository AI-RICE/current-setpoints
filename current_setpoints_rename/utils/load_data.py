import os

import pandas as pd


def load_aggregated_csv_data(file_path: str, col_map: dict[str, str]) -> pd.DataFrame:
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

    missing_cols = [col for col in col_map.keys() if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    data = data.rename(columns=col_map)

    required_cols = list(col_map.values())

    if data[required_cols].isnull().values.any():
        n_before = len(data)
        data = data.dropna(subset=required_cols)
        n_dropped = n_before - len(data)
        print(f"Warning: dropped {n_dropped} rows with NaNs in mapped columns.")

    print(f"Loaded {len(data)} valid data points from CSV.")
    return data

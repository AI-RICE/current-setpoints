from __future__ import annotations

import os

import pandas as pd


def load_aggregated_csv_data(file_path: str, col_map: dict[str, str]) -> pd.DataFrame:
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

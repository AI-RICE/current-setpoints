import os
import pandas as pd


# TODO: (DONE) move somewhere else. is it going to be used for all data? if yes, move it to utils. if not, move it to ../../notebooks
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
    required_cols = ['omega', 'id1', 'iq1', 'id3', 'iq3', 'torq_meas']
    
    # Validation: NaNs
    if data[required_cols].isnull().values.any():
        print("Warning: NaNs found in required columns. Dropping invalid rows.")
        data.dropna(subset=required_cols, inplace=True)
        
    print(f"Loaded {len(data)} valid data points from CSV.")
    return data
# src/data/load_data.py

import os
import pandas as pd

def load_expense_data(file_path="data/raw/expenses.csv"):
    """
    Load historical expense data from a CSV file.

    Args:
        file_path (str): Path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded expense data as a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found at: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        raise

if __name__ == "__main__":
    # For testing purposes
    df = load_expense_data()
    print(df.head())

"""
Project: Multi-Agent Cardiac Risk Prediction System
AUTOR: Raphael da Silva RM561452
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic_data.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    # Dataset dimensions
    print("\n1. DATASET DIMENSIONS")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Data types
    print("\n2. DATA TYPES")
    print(df.dtypes)

    # Missing values
    print("\n3. MISSING VALUES")
    print(df.isnull().sum())

    # Duplicate records
    print("\n4. DUPLICATE RECORDS")
    print(df.duplicated().sum())

    # Descriptive statistics
    print("\n5. DESCRIPTIVE STATISTICS")
    print(df.describe().round(2))

    # Target class distribution
    print("\n6. TARGET CLASS DISTRIBUTION")
    print(df["risk_peak"].value_counts().sort_index())

    # Target class percentages
    print("\n7. TARGET CLASS PERCENTAGES")
    print(
        df["risk_peak"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    # Dataset validation
    print("\n8. DATASET VALIDATION")

    assert df.shape == (5000, 11), "Unexpected dataset dimensions."
    assert df["risk_peak"].isin([0, 1]).all(), "Invalid target values."
    assert not df.isnull().any().any(), "Missing values detected."

    print("All validations passed successfully.")


if __name__ == "__main__":
    main()

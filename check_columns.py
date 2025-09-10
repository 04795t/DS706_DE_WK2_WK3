"""
Quick script to check column names in each year's file.
"""

import pandas as pd

print("CHECKING COLUMN NAMES IN EACH YEAR'S FILE")
print("=" * 50)

for year in range(2015, 2020):
    try:
        df = pd.read_csv(f"Data/{year}.csv")
        print(f"\n{year}.csv columns:")
        print(f"  First 5 columns: {list(df.columns[:5])}")

        # Check for happiness-related columns.
        happiness_cols = [
            col
            for col in df.columns
            if "happiness" in col.lower()
            or "score" in col.lower()
            or "ladder" in col.lower()
        ]
        if happiness_cols:
            print(f"  Happiness columns found: {happiness_cols}")

    except Exception as e:
        print(f"\n{year}.csv: Could not load - {e}")

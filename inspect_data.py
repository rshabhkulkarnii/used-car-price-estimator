
from pathlib import Path
import pandas as pd

# Find CSV files inside the data directory
data_dir = Path("data")
csv_files = list(data_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV found inside data/. Download the dataset first."
    )

print("Available CSV files:")
for i, path in enumerate(csv_files, start=1):
    print(f"{i}. {path}")

# Select the first CSV for initial inspection.
# If multiple CSVs exist, verify that this is the correct one.
csv_path = csv_files[1]
df = pd.read_csv(csv_path)

print("\n--- DATASET OVERVIEW ---")
print("File:", csv_path)
print("Rows and columns:", df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- NUMERIC SUMMARY ---")
print(df.describe())

print("\n--- CATEGORICAL SUMMARY ---")
print(df.describe(include="object"))

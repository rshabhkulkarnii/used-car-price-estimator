
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in data/")

print("Available CSV files:")
for i, path in enumerate(csv_files):
    print(f"{i}: {path.name}")

# If multiple files exist, set this to the index of your chosen CSV.
# Index 1 means the second CSV in the displayed list.
CSV_INDEX = 1

if CSV_INDEX >= len(csv_files):
    raise ValueError("CSV_INDEX does not match an available CSV file.")

csv_path = csv_files[CSV_INDEX]
df = pd.read_csv(csv_path)

print("\nSelected file:", csv_path.name)
print("Original shape:", df.shape)

# 1. Verify the expected schema
required_columns = {
    "name", "year", "selling_price", "km_driven",
    "fuel", "seller_type", "transmission", "owner"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Unexpected dataset schema. Missing: {missing_columns}")

# 2. Inspect exact duplicates before removing anything
duplicate_mask = df.duplicated(keep=False)
duplicate_rows = df.loc[duplicate_mask].sort_values(
    by=list(df.columns)
)

print("\nRows belonging to duplicate groups:", duplicate_mask.sum())
print("Extra copies beyond the first:", df.duplicated().sum())

duplicate_rows.to_csv(
    OUTPUT_DIR / "duplicate_records_for_review.csv",
    index=False
)

# 3. Inspect numeric validity
for col in ["year", "selling_price", "km_driven"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nInvalid numeric values:")
print(df[["year", "selling_price", "km_driven"]].isna().sum())

print("\nNon-positive selling prices:", (df["selling_price"] <= 0).sum())
print("Negative kilometers:", (df["km_driven"] < 0).sum())
print("Future manufacturing years:", (df["year"] > 2026).sum())
print("Years before 1900:", (df["year"] < 1900).sum())

print("\nDistinct values in categorical columns:")
for col in ["fuel", "seller_type", "transmission", "owner"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

# 4. Remove exact duplicates for the initial experiment
clean_df = df.drop_duplicates().copy()

# Remove rows that cannot have a meaningful positive resale price
# or have invalid numeric values.
before_validation = len(clean_df)

clean_df = clean_df.dropna(
    subset=["year", "selling_price", "km_driven"]
)

clean_df = clean_df[
    (clean_df["selling_price"] > 0)
    & (clean_df["km_driven"] >= 0)
    & (clean_df["year"] >= 1900)
    & (clean_df["year"] <= 2026)
].copy()

print("\n--- CLEANING SUMMARY ---")
print("Original rows:", len(df))
print("After exact deduplication:", len(df.drop_duplicates()))
print("Rows removed by numeric validation:",
      before_validation - len(clean_df))
print("Final rows:", len(clean_df))
print("Final columns:", len(clean_df.columns))

# 5. Save a new dataset; do not overwrite the original.
output_path = OUTPUT_DIR / "used_cars_clean.csv"
clean_df.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:", output_path)

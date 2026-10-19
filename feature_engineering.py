
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/used_cars_clean.csv")
OUTPUT_PATH = Path("data/processed/used_cars_features.csv")

REFERENCE_YEAR = 2026

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Complete Step 3 first."
    )

df = pd.read_csv(INPUT_PATH)

# 1. Extract manufacturer from the first word of the car name.
# This is a baseline heuristic, not a perfect brand parser.
df["brand"] = (
    df["name"]
    .astype(str)
    .str.strip()
    .str.split()
    .str[0]
)

# 2. Create vehicle age using a fixed reference year.
df["vehicle_age"] = REFERENCE_YEAR - df["year"]

# 3. Basic checks
print("--- FEATURE ENGINEERING RESULTS ---")
print("Shape:", df.shape)

print("\nFirst 10 car names and extracted brands:")
print(df[["name", "brand"]].head(10))

print("\nVehicle age summary:")
print(df["vehicle_age"].describe())

print("\nUnique brands:", df["brand"].nunique())
print("\nMost common extracted brands:")
print(df["brand"].value_counts().head(15))

print("\nNegative vehicle ages:", (df["vehicle_age"] < 0).sum())

# 4. Save the engineered dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved engineered dataset to:", OUTPUT_PATH)

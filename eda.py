
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Locate the CSV you selected
csv_files = list(Path("data").glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in the data folder.")

print("CSV files found:")
for i, path in enumerate(csv_files, start=1):
    print(f"{i}. {path}")

# Change this index if your selected CSV is not the first file.
df = pd.read_csv(csv_files[1])

# 1. Dataset overview
print("\n--- SHAPE ---")
print(df.shape)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATES ---")
print(df.duplicated().sum())

# 2. Data types
print("\n--- DATA TYPES ---")
print(df.dtypes)

# 3. Numeric ranges
print("\n--- NUMERIC SUMMARY ---")
print(df[["year", "selling_price", "km_driven"]].describe())

# 4. Categorical distributions
for col in ["fuel", "seller_type", "transmission", "owner"]:
    print(f"\n--- {col.upper()} COUNTS ---")
    print(df[col].value_counts(dropna=False))

# 5. Price distribution
sns.histplot(data=df, x="selling_price", bins=40)
plt.title("Distribution of Used Car Selling Prices")
plt.xlabel("Selling price")
plt.tight_layout()
plt.show()

# 6. Price versus manufacturing year
sns.scatterplot(data=df, x="year", y="selling_price", alpha=0.5)
plt.title("Selling Price vs Manufacturing Year")
plt.tight_layout()
plt.show()

# 7. Price versus kilometers driven
sns.scatterplot(data=df, x="km_driven", y="selling_price", alpha=0.5)
plt.title("Selling Price vs Kilometers Driven")
plt.tight_layout()
plt.show()

# 8. Price by fuel type
sns.boxplot(data=df, x="fuel", y="selling_price")
plt.title("Selling Price by Fuel Type")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 9. Price by transmission
sns.boxplot(data=df, x="transmission", y="selling_price")
plt.title("Selling Price by Transmission")
plt.tight_layout()
plt.show()


from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/used_cars_features.csv")
MODEL_PATH = Path("models/car_price_pipeline.joblib")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = [
    "vehicle_age",
    "km_driven",
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]
target = "selling_price"

X = df[features]
y = df[target]

# Must match the split used during training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = joblib.load(MODEL_PATH)
predictions = model.predict(X_test)

# 1. Overall metrics
print("--- TEST METRICS ---")
print(f"MAE:  {mean_absolute_error(y_test, predictions):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):,.2f}")
print(f"R²:   {r2_score(y_test, predictions):.4f}")

# 2. Prediction-level error analysis
errors = pd.DataFrame({
    "actual_price": y_test.to_numpy(),
    "predicted_price": predictions,
})

errors["absolute_error"] = (
    errors["actual_price"] - errors["predicted_price"]
).abs()

errors["signed_error"] = (
    errors["predicted_price"] - errors["actual_price"]
)

errors["absolute_percentage_error"] = (
    errors["absolute_error"] / errors["actual_price"]
) * 100

errors.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

print("\n--- WORST 10 ABSOLUTE ERRORS ---")
print(
    errors.nlargest(10, "absolute_error")
    .round(2)
    .to_string(index=False)
)

# 3. Error by actual price range
errors["price_band"] = pd.cut(
    errors["actual_price"],
    bins=[0, 200_000, 500_000, 1_000_000, np.inf],
    labels=["Up to 2 lakh", "2–5 lakh", "5–10 lakh", "Above 10 lakh"],
)

print("\n--- ERROR BY PRICE BAND ---")
print(
    errors.groupby("price_band", observed=True)
    .agg(
        count=("actual_price", "size"),
        mean_actual=("actual_price", "mean"),
        MAE=("absolute_error", "mean"),
        median_absolute_error=("absolute_error", "median"),
    )
    .round(2)
    .to_string()
)

# 4. Actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(
    errors["actual_price"],
    errors["predicted_price"],
    alpha=0.5,
)
max_price = max(
    errors["actual_price"].max(),
    errors["predicted_price"].max(),
)
plt.plot([0, max_price], [0, max_price], linestyle="--")
plt.xlabel("Actual selling price")
plt.ylabel("Predicted selling price")
plt.title("Actual vs Predicted Used Car Prices")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
plt.show()

# 5. Residual distribution
plt.figure(figsize=(8, 5))
plt.hist(errors["signed_error"], bins=40)
plt.xlabel("Prediction error (predicted - actual)")
plt.ylabel("Count")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_distribution.png", dpi=150)
plt.show()

print("\nSaved prediction analysis and plots in models/.")

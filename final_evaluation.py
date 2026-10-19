
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/processed/used_cars_features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

numeric_features = ["vehicle_age", "km_driven"]
categorical_features = [
    "brand", "fuel", "seller_type", "transmission", "owner"
]
features = numeric_features + categorical_features
target = "selling_price"

X = df[features]
y = df[target]

# Reproduce the exact split used in improve_model.py.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=2026
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=2026
)

# Use training + validation data after selecting the model.
X_fit = pd.concat([X_train, X_valid])
y_fit = pd.concat([y_train, y_valid])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

final_model.fit(X_fit, y_fit)
predictions = final_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("--- FINAL HELD-OUT TEST RESULTS ---")
print("Training + validation rows:", len(X_fit))
print("Final test rows:", len(X_test))
print(f"MAE:  ₹{mae:,.2f}")
print(f"RMSE: ₹{rmse:,.2f}")
print(f"R²:   {r2:.4f}")

# Save the fitted pipeline and test results.
joblib.dump(
    final_model,
    MODEL_DIR / "car_price_pipeline.joblib"
)

pd.DataFrame({
    "actual_price": y_test.to_numpy(),
    "predicted_price": predictions,
    "absolute_error": np.abs(y_test.to_numpy() - predictions),
}).to_csv(MODEL_DIR / "final_test_predictions.csv", index=False)

pd.DataFrame([{
    "model": "Random Forest",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "training_rows": len(X_fit),
    "test_rows": len(X_test),
}]).to_csv(MODEL_DIR / "final_metrics.csv", index=False)

print("\nSaved final pipeline and evaluation results.")

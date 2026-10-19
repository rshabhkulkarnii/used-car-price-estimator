
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/processed/used_cars_features.csv")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

numeric_features = ["vehicle_age", "km_driven"]
categorical_features = [
    "brand", "fuel", "seller_type", "transmission", "owner"
]
features = numeric_features + categorical_features

X = df[features]
y = df["selling_price"]

# Fresh split: 70% training, 15% validation, 15% final test.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=2026
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=2026
)

def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("categorical", categorical, categorical_features),
    ])

def make_pipeline(regressor):
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", regressor),
    ])

models = {
    "Linear Regression": make_pipeline(LinearRegression()),

    "Random Forest": make_pipeline(
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    ),

    "Random Forest (log target)": TransformedTargetRegressor(
        regressor=make_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    ),
}

def evaluate(model, X_eval, y_eval):
    pred = model.predict(X_eval)
    return {
        "MAE": mean_absolute_error(y_eval, pred),
        "RMSE": np.sqrt(mean_squared_error(y_eval, pred)),
        "R2": r2_score(y_eval, pred),
    }

validation_results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_valid, y_valid)

    validation_results.append({
        "Model": name,
        **metrics,
    })

    print(f"\n--- {name}: VALIDATION ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:,.4f}")

results = pd.DataFrame(validation_results)
results.to_csv(OUTPUT_DIR / "validation_comparison.csv", index=False)

print("\n--- VALIDATION COMPARISON ---")
print(results.sort_values("MAE").to_string(index=False))

print(
    "\nTraining rows:", len(X_train),
    "| Validation rows:", len(X_valid),
    "| Final test rows:", len(X_test)
)

# Save the split so the final evaluation can use the same test records.
X_test.assign(selling_price=y_test).to_csv(
    OUTPUT_DIR / "final_test_set.csv", index=False
)

print("\nValidation results saved. Select a model using validation metrics.")
print("Do not evaluate or tune against the final test set yet.")

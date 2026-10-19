
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA_PATH = Path("data/processed/used_cars_features.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Feature-engineered dataset not found. Complete Step 4 first."
    )

df = pd.read_csv(DATA_PATH)

# Define target and input features
target = "selling_price"

numeric_features = [
    "vehicle_age",
    "km_driven",
]

categorical_features = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

required = [target] + numeric_features + categorical_features
missing = set(required) - set(df.columns)

if missing:
    raise ValueError(f"Required columns missing: {missing}")

# Ensure the target is valid
df = df.dropna(subset=[target]).copy()
df = df[df[target] > 0].copy()

X = df[numeric_features + categorical_features]
y = df[target]

# Split before fitting any preprocessing or model.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ]
)

print("--- DATA SPLIT ---")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training target mean:", round(y_train.mean(), 2))
print("Testing target mean:", round(y_test.mean(), 2))

print("\nTraining feature columns:", X_train.columns.tolist())
print("\nTarget column:", target)

print("\nPreprocessor created successfully.")
print("No preprocessing has been fitted yet.")

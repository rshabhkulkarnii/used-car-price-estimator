
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "used_cars_features.csv"
MODEL_PATH = BASE_DIR / "models" / "car_price_pipeline.joblib"

# Must match feature engineering during training.
REFERENCE_YEAR = 2026

st.set_page_config(
    page_title="Used Car Price Estimator",
    page_icon="🚗",
    layout="centered",
)

@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)

@st.cache_data
def load_data(path, modified_time):
    return pd.read_csv(path)

st.title("🚗 Used Car Price Estimator")
st.write(
    "Estimate the resale price of a pre-owned car using a "
    "Random Forest regression model."
)

st.caption(
    "Educational ML project. The estimate is not a guaranteed market price."
)

# Check required artifacts.
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Trained pipeline not found: {MODEL_PATH}")
    st.stop()

try:
    df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load model or dataset: {exc}")
    st.stop()

required_columns = {
    "brand",
    "year",
    "vehicle_age",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "selling_price",
}

missing = required_columns - set(df.columns)

if missing:
    st.error(f"Dataset is missing required columns: {sorted(missing)}")
    st.stop()

expected_features = [
    "vehicle_age",
    "km_driven",
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

# Verify the feature order/names against the fitted pipeline.
pipeline_features = getattr(model, "feature_names_in_", None)

if pipeline_features is not None:
    if list(pipeline_features) != expected_features:
        st.error(
            "The saved model's expected features do not match this app. "
            f"Expected by model: {list(pipeline_features)}"
        )
        st.stop()

# Dropdown options are derived from the actual engineered dataset.
def get_categories(column):
    values = df[column].dropna().astype(str).str.strip()
    return sorted(v for v in values.unique() if v)

brands = get_categories("brand")
fuels = get_categories("fuel")
seller_types = get_categories("seller_type")
transmissions = get_categories("transmission")
owners = get_categories("owner")

min_year = int(df["year"].min())
max_year = int(df["year"].max())
max_km = int(df["km_driven"].max())

st.info(
    f"The training data covers manufacturing years {min_year}–{max_year}. "
    "Predictions outside the observed range may be unreliable."
)

st.subheader("Enter vehicle details")

with st.form("car_form"):
    brand = st.selectbox("Brand", brands)

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "Manufacturing year",
            min_value=1950,
            max_value=REFERENCE_YEAR,
            value=min(max_year, REFERENCE_YEAR),
            step=1,
        )

    with col2:
        km_driven = st.number_input(
            "Kilometers driven",
            min_value=0,
            max_value=max(2_000_000, max_km),
            value=45_000,
            step=1,
        )

    fuel = st.selectbox("Fuel type", fuels)
    transmission = st.selectbox("Transmission", transmissions)
    seller_type = st.selectbox("Seller type", seller_types)
    owner = st.selectbox("Ownership history", owners)

    submitted = st.form_submit_button(
        "Estimate resale price",
        use_container_width=True,
        type="primary",
    )

if submitted:
    if year > REFERENCE_YEAR:
        st.error("Manufacturing year cannot be in the future.")
        st.stop()

    vehicle_age = REFERENCE_YEAR - int(year)

    # Warn instead of pretending out-of-range inputs are well supported.
    warnings = []

    if year < min_year or year > max_year:
        warnings.append(
            f"The selected year is outside the training range "
            f"({min_year}–{max_year})."
        )

    if vehicle_age < int(df["vehicle_age"].min()):
        warnings.append(
            "This car is newer than the vehicles represented in training."
        )

    if km_driven > max_km:
        warnings.append(
            "This mileage exceeds the maximum observed in the training data."
        )

    if km_driven == 0:
        warnings.append(
            "Zero kilometres is unusual for a used car. Verify this value."
        )

    for message in warnings:
        st.warning(message)

    input_data = pd.DataFrame([{
        "vehicle_age": vehicle_age,
        "km_driven": int(km_driven),
        "brand": brand,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }], columns=expected_features)

    try:
        prediction = float(np.asarray(model.predict(input_data)).ravel()[0])

        if not np.isfinite(prediction) or prediction < 0:
            st.error(
                "The model produced an invalid price estimate. "
                "Please review the trained model."
            )
        else:
            st.success("Estimated resale price")

            st.metric(
                label="Predicted selling price",
                value=f"₹{prediction:,.0f}",
            )

            st.caption(
                "This is a statistical estimate based on historical listings. "
                "Actual resale value depends on vehicle condition, model variant, "
                "service history, location, and current market demand."
            )

            with st.expander("View submitted vehicle details"):
                st.dataframe(input_data, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

with st.expander("About this model"):
    st.write("Algorithm: Random Forest Regressor")
    st.write("Training rows: 3,040")
    st.write("Final test rows: 537")
    st.write("Final test R²: 0.7464")
    st.write("Final test MAE: ₹151,725.48")
    st.write("Final test RMSE: ₹263,049.15")
    st.write(
        "The model was trained on the CarDekho used-car dataset. "
        "Its performance may differ on other markets or vehicle types."
    )

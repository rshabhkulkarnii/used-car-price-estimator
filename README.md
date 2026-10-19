# 🚗 Used Car Price Estimator

A Machine Learning web application that estimates the resale price of a used car based on its age, mileage, brand, fuel type, seller type, transmission, and ownership history. Built with Python, Scikit-learn, and Streamlit.

**🔴 Live Demo:** [Try the Used Car Price Estimator](https://used-car-price-estimator-rshabh.streamlit.app/)

---

## 📌 Project Overview

Estimating the resale value of a used car can be challenging because prices depend on several factors, including vehicle age, distance driven, brand, fuel type, transmission, and ownership history.

This project uses supervised machine learning to predict used car selling prices from historical vehicle data. It includes a complete workflow covering data cleaning, feature engineering, model comparison, evaluation, and deployment as an interactive web application.

### Objectives

- Clean and prepare a real-world used-car dataset.
- Engineer useful features for price prediction.
- Compare baseline and machine learning regression models.
- Evaluate the selected model on an untouched test set.
- Build an interactive interface for generating price estimates.
- Deploy the application using Streamlit Community Cloud.

## ✨ Features

- **Interactive predictions:** Enter vehicle details to estimate its selling price.
- **Feature engineering:** Derives vehicle age and extracts the car brand from its name.
- **Data preprocessing:** Uses median imputation for numerical features and most-frequent imputation for categorical features, followed by one-hot encoding.
- **Model comparison:** Evaluates a Dummy Regressor, Linear Regression, and Random Forest configurations.
- **Model evaluation:** Reports MAE, RMSE, and R².
- **Input validation:** Includes warnings for certain unusual or out-of-range inputs.
- **Online deployment:** Makes the prediction interface accessible through a web browser.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computing |
| Scikit-learn | Preprocessing, model training, and evaluation |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive web application |
| Git and GitHub | Version control and project hosting |
| Streamlit Community Cloud | Application deployment |

## 📊 Dataset

The project uses the **Car Details from Car Dekho** dataset.

The original dataset contains 4,340 records and eight columns.

| Column | Description |
|---|---|
| `name` | Car name and model |
| `year` | Manufacturing year |
| `selling_price` | Historical selling price in INR |
| `km_driven` | Distance driven in kilometres |
| `fuel` | Fuel type |
| `seller_type` | Type of seller |
| `transmission` | Manual or automatic transmission |
| `owner` | Previous ownership category |

### Data preparation

- Removed extra exact duplicate records.
- Checked for missing values and invalid numerical records.
- Extracted the vehicle brand from the car name.
- Created `vehicle_age` using the reference year 2026.
- Used `selling_price` as the prediction target.

After duplicate removal, the cleaned dataset contained **3,577 records**.

### Engineered features

The model uses seven input features:

- `vehicle_age`
- `km_driven`
- `brand`
- `fuel`
- `seller_type`
- `transmission`
- `owner`

The original `name` and `year` columns are replaced in the model input by the derived `brand` and `vehicle_age` features.

## ⚙️ Machine Learning Workflow

The project follows this workflow:

1. **Data cleaning:** Inspect the dataset and remove extra exact duplicates.
2. **Feature engineering:** Extract brand and calculate vehicle age.
3. **Data splitting:** Divide the data into training, validation, and final test sets.
4. **Preprocessing:** Impute missing values where necessary and one-hot encode categorical features.
5. **Model training:** Train and compare multiple regression models.
6. **Model selection:** Select the Random Forest configuration based on validation performance.
7. **Final evaluation:** Refit the selected model on training and validation data, then evaluate it on the untouched test set.
8. **Deployment:** Load the saved preprocessing-and-model pipeline into a Streamlit application.

### Dataset split

| Dataset | Records | Percentage |
|---|---:|---:|
| Training | 2,503 | 70% |
| Validation | 537 | 15% |
| Final test | 537 | 15% |
| **Total** | **3,577** | **100%** |

The final test set was kept separate during model selection to provide a more reliable estimate of performance on unseen data.

## 🤖 Model Selection and Results

The project compares baseline and regression approaches.

The Dummy Regressor establishes a baseline, while Linear Regression and Random Forest are evaluated for their ability to capture relationships between vehicle characteristics and selling price.

The following are the recorded final test metrics for the selected Random Forest model:

| Metric | Final test result |
|---|---:|
| Mean Absolute Error (MAE) | ₹151,725.48 |
| Root Mean Squared Error (RMSE) | ₹263,049.15 |
| R² Score | 0.7464 |

### What do these metrics mean?

- **MAE:** Predictions differ from actual selling prices by approximately ₹1.52 lakh on average in absolute terms.
- **RMSE:** Larger prediction errors receive greater weight, resulting in an RMSE of approximately ₹2.63 lakh.
- **R²:** The model explains approximately 74.64% of the variation in selling prices on the final test set, relative to predicting the test-set mean.

These metrics do **not** mean that the model is 74.64% accurate for every individual car.

### Why Random Forest?

Random Forest was selected because it performed substantially better than the Dummy Regressor and achieved stronger validation performance than the tested Linear Regression model. It can model nonlinear relationships and interactions between vehicle characteristics.

## 🏗️ Project Structure

```text
used-car-price-estimator/
│
├── app.py
├── clean_data.py
├── feature_engineering.py
├── prepare_data.py
├── train_model.py
├── improve_model.py
├── final_evaluation.py
│
├── data/
│   └── processed/
│       ├── used_cars_clean.csv
│       └── used_cars_features.csv
│
├── models/
│   ├── car_price_pipeline.joblib
│   ├── model_comparison.csv
│   └── final_metrics.csv
│
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

*Note: The exact files included in the repository may differ. The original downloaded dataset and temporary evaluation outputs may be excluded by `.gitignore`.*

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/used-car-price-estimator.git
cd used-car-price-estimator
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify required files

Make sure the repository contains:

- `models/car_price_pipeline.joblib`
- `data/processed/used_cars_features.csv`

The saved model pipeline and processed feature dataset are required by the current Streamlit application.

### 5. Launch the application

```bash
streamlit run app.py
```

Open the local URL printed in your terminal, usually `http://localhost:8501`.

## 📈 Example Usage

1. Open the application.
2. Select a car brand and its vehicle characteristics.
3. Enter the vehicle age and kilometres driven.
4. Select the fuel type, seller type, transmission, and ownership category.
5. Generate the estimated selling price.

The result is an estimated resale price in Indian rupees, not a guaranteed market valuation.

## ⚠️ Limitations and Future Improvements

### Current limitations

- The model is trained on historical data and may not reflect current market prices.
- Prediction errors are substantially higher for expensive cars.
- The dataset may not represent every brand, model, region, or market condition.
- Important variables such as vehicle condition, accident history, service records, trim level, and location are not explicitly included.
- The reference year used for calculating vehicle age is fixed at 2026 in the current implementation.

### Future improvements

- Use a current and larger dataset with more vehicle attributes.
- Add model-level features such as exact model, variant, engine capacity, and kilometres per year.
- Compare Random Forest against Gradient Boosting, Extra Trees, and other suitable regression algorithms.
- Perform systematic hyperparameter tuning using cross-validation.
- Add residual plots and error analysis by price range.
- Update the vehicle-age calculation dynamically.
- Add automated tests and input validation.
- Explore prediction intervals to communicate uncertainty.

## 👨‍💻 Author

**Rushabh Kulkarni**

Computer Engineering Student | Machine Learning and Software Development

- GitHub: [Your GitHub Profile](https://github.com/YOUR-USERNAME)
- Live Project: [Used Car Price Estimator](https://used-car-price-estimator-rshabh.streamlit.app/)

---

## 📄 Disclaimer

This project was developed for educational and demonstration purposes. Predictions are estimates based on historical data and should not be treated as professional vehicle valuations or guaranteed selling prices.

The dataset and its usage are subject to the original dataset provider's terms and licensing conditions.

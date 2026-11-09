from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "odisha_real_estate.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

TARGET = "price_inr"
NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "property_age", "parking"]
CATEGORICAL = ["location", "property_type"]
FEATURES = NUMERIC + CATEGORICAL

def main():
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df[required].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]
    if len(df) < 10:
        raise ValueError("Please provide at least 10 valid rows; more verified records are strongly recommended.")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL)
    ])

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=None, min_samples_leaf=1,
            random_state=42
        )
    }

    results = []
    fitted = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)
        results.append({"model": name, "MAE_INR": round(float(mae), 2),
                        "RMSE_INR": round(rmse, 2), "R2": round(float(r2), 4)})
        fitted[name] = pipe

    # Choose by lowest MAE on the held-out test split (small demo data: interpret cautiously).
    best_row = min(results, key=lambda row: row["MAE_INR"])
    best_name = best_row["model"]
    best_model = fitted[best_name]

    joblib.dump(best_model, ARTIFACTS / "best_model.joblib")
    metadata = {
        "model_name": best_name,
        "features": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "locations": sorted(df["location"].dropna().astype(str).unique().tolist()),
        "property_types": sorted(df["property_type"].dropna().astype(str).unique().tolist()),
        "metrics": results,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "currency": "INR",
        "dataset_notice": "Synthetic demo data; not verified market data."
    }
    (ARTIFACTS / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Save simple EDA charts.
    plt.figure()
    df[TARGET].hist(bins=12)
    plt.title("Property Price Distribution (INR)")
    plt.xlabel("Price (INR)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "price_distribution.png")
    plt.close()

    plt.figure()
    plt.scatter(df["area_sqft"], df[TARGET])
    plt.title("Area vs Property Price")
    plt.xlabel("Area (sq.ft)")
    plt.ylabel("Price (INR)")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "area_vs_price.png")
    plt.close()

    print("\nModel comparison:")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {ARTIFACTS / 'best_model.joblib'}")
    print("Note: synthetic demo data scores are not real-world accuracy.")

if __name__ == "__main__":
    main()

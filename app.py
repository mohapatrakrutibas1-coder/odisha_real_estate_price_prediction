from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"
META_PATH = ROOT / "artifacts" / "model_metadata.json"

app = Flask(__name__)

def load_assets():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python train_model.py` first."
        )
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, metadata

model, metadata = load_assets()

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        locations=metadata["locations"],
        property_types=metadata["property_types"],
        prediction=None,
        error=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        location = request.form.get("location", "").strip()
        property_type = request.form.get("property_type", "").strip()
        area = float(request.form.get("area_sqft", ""))
        bedrooms = int(request.form.get("bedrooms", ""))
        bathrooms = int(request.form.get("bathrooms", ""))
        age = int(request.form.get("property_age", ""))
        parking = int(request.form.get("parking", ""))

        if location not in metadata["locations"]:
            raise ValueError("Please select a valid location.")
        if property_type not in metadata["property_types"]:
            raise ValueError("Please select a valid property type.")
        if area <= 0 or area > 50000:
            raise ValueError("Area must be between 1 and 50,000 sq.ft.")
        if not 0 <= bedrooms <= 20:
            raise ValueError("Bedrooms must be between 0 and 20.")
        if not 0 <= bathrooms <= 20:
            raise ValueError("Bathrooms must be between 0 and 20.")
        if not 0 <= age <= 150:
            raise ValueError("Property age must be between 0 and 150 years.")
        if parking not in (0, 1, 2, 3, 4, 5):
            raise ValueError("Parking must be between 0 and 5.")

        sample = pd.DataFrame([{
            "area_sqft": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "property_age": age,
            "parking": parking,
            "location": location,
            "property_type": property_type
        }])
        predicted = float(model.predict(sample)[0])
        predicted = max(0, predicted)
        return render_template(
            "index.html",
            locations=metadata["locations"],
            property_types=metadata["property_types"],
            prediction=f"₹{predicted:,.0f}",
            error=None,
            submitted=request.form
        )
    except (ValueError, TypeError) as exc:
        return render_template(
            "index.html",
            locations=metadata["locations"],
            property_types=metadata["property_types"],
            prediction=None,
            error=str(exc),
            submitted=request.form
        )

@app.route("/health")
def health():
    return {"status": "ok", "model": metadata["model_name"]}

if __name__ == "__main__":
    app.run(debug=True)

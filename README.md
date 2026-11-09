# Odisha Real Estate Price Prediction

A complete beginner-friendly Flask + Scikit-learn project for estimating residential property prices in Odisha.

## Important dataset note
`data/odisha_real_estate.csv` contains **synthetic demo records** created only to help you run and understand the project. They are NOT verified market transactions and must not be presented as actual Odisha property prices. Replace them with reliable, legally usable, verified property data before making real-world claims.

## Features
- Data loading and validation with Pandas
- EDA charts (price distribution, area vs price, average price by location)
- Preprocessing with imputers and one-hot encoding
- Compare Linear Regression and Random Forest
- Evaluate with MAE, RMSE and R²
- Save the best fitted preprocessing + model pipeline
- Flask web app with HTML, CSS and JavaScript
- Input validation and estimated price display

## 1. Requirements
- Python 3.10 or newer
- VS Code (recommended)

## 2. Create and activate a virtual environment

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## 3. Install packages
```bash
pip install -r requirements.txt
```

## 4. Train the model
From the project root:
```bash
python train_model.py
```
This creates:
- `artifacts/best_model.joblib`
- `artifacts/model_metadata.json`
- charts inside `artifacts/`

## 5. Run the website
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## 6. How to use
Choose a location, enter area in square feet, bedrooms, bathrooms, property age, parking, and property type. Click **Predict Price**.

## 7. Replace the demo dataset
Keep the same column names:
`location, area_sqft, bedrooms, bathrooms, property_age, parking, property_type, price_inr`

Use verified, consistent data. `price_inr` must be numeric INR (not strings such as "50 lakh"). Avoid duplicates, impossible values, and mixing land prices with apartment/house prices unless property type is explicitly represented. Keep a separate test set if you have enough records.

## 8. Deploy on Render
1. Push the project to GitHub.
2. Create a new Web Service on Render and connect the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Deploy. The app loads the trained model from `artifacts/`, so train it before deployment and include the generated artifact in the repository, or add a build step that trains it during deployment.

## 9. Limitations
The demo data is small and synthetic. Its evaluation scores are for software-learning purposes only and do not indicate real predictive accuracy. Real-estate prices depend on locality, road access, building condition, floor, legal status, amenities, market date, and many other factors. Treat output as an estimate, not a valuation or financial advice.

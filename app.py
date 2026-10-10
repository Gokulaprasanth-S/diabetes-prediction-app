import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)

# This model is maintained in the separate training project.
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "model"
    / "diabetes_prediction_pipeline_v3.pkl"
)

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

DISCLAIMER = (
    "This is an academic ML prediction and NOT a medical diagnosis. "
    "Consult a qualified healthcare professional for medical advice."
)


@app.route("/")
def index():
    """Show the prediction form."""
    return render_template("index.html", disclaimer=DISCLAIMER)


@app.route("/predict", methods=["POST"])
def predict():
    """Validate submitted measurements and show the model's prediction."""
    values = {}

    for feature in FEATURES:
        raw_value = request.form.get(feature, "").strip()

        if not raw_value:
            return render_template(
                "index.html",
                error=f"Please provide a value for {feature}.",
                disclaimer=DISCLAIMER,
            )
        try:
            numeric_value = float(raw_value)
        except ValueError:
            return render_template(
                "index.html",
                error=f"Please enter a valid number for {feature}.",
                disclaimer=DISCLAIMER,
            )
        if not math.isfinite(numeric_value):
            return render_template(
                "index.html",
                error=f"Please enter a valid number for {feature}.",
                disclaimer=DISCLAIMER,
            )
        values[feature] = numeric_value

    # Construct one row with the exact names and order expected by the pipeline.
    input_data = pd.DataFrame([values], columns=FEATURES)

    try:
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        probability_percent = probability * 100
    except Exception:
        # Keep implementation details, including the model path, out of the page.
        return render_template(
            "index.html",
            error="The prediction service is unavailable. Please try again later.",
            disclaimer=DISCLAIMER,
        )

    if probability_percent < 30:
        risk_interpretation = "Low estimated risk"
    elif probability_percent < 60:
        risk_interpretation = "Moderate estimated risk"
    else:
        risk_interpretation = "Higher estimated risk"

    return render_template(
        "result.html",
        prediction=prediction,
        probability=round(probability_percent, 2),
        risk_interpretation=risk_interpretation,
        disclaimer=DISCLAIMER,
    )


if __name__ == "__main__":
    app.run()
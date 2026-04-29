"""Flask app for Heart Disease prediction demo."""

from __future__ import annotations

import json
import sys
from typing import Any
from pathlib import Path

from flask import Flask, render_template, request

# Ensure local packages can be imported when running as script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.heart_disease_predict_model import load_model, predict_patient

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)

MODEL_RESULTS_PATH = MODELS_DIR / "heart_disease_model_results.json"

PERSONAL_FIELDS = [
    ("age", "number"),
    ("sex", "select"),
]

BLOOD_TESTING_FIELDS = [
    ("chol", "number"),
    ("fbs", "select"),
]

PHYSICAL_TESTING_FIELDS = [
    ("cp", "select"),
    ("restecg", "select"),
    ("exang", "select"),
    ("trestbps", "number"),
    ("thalch", "number"),
    ("oldpeak", "number"),
]

FEATURE_FIELDS = PERSONAL_FIELDS + BLOOD_TESTING_FIELDS + PHYSICAL_TESTING_FIELDS

SELECT_OPTIONS = {
    "sex": ["Male", "Female"],
    "cp": ["asymptomatic", "atypical angina", "non-anginal", "typical angina"],
    "fbs": ["False", "True"],
    "restecg": ["lv hypertrophy", "normal", "st-t abnormality"],
    "exang": ["No", "Yes"],
}

NORMAL_RANGES = {
    "cp": "non-anginal",
    "trestbps": "~ 90-120 mm Hg",
    "chol": "< 200 mg/dL",
    "fbs": "False or <= 120 mg/dL",
    "restecg": "normal",
    "thalch": "~ 150-210 bpm",
    "exang": "False",
    "oldpeak": "< 1.0 mm",
}

FIELD_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (Systolic, mm Hg)",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalch": "Max Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression",
}

INPUT_ATTRIBUTES = {
    "age": {"step": "1", "min": "0"},
    "trestbps": {"step": "1", "min": "0"},
    "chol": {"step": "1", "min": "0"},
    "thalch": {"step": "1", "min": "0"},
    # Oldpeak is exercise-induced ST depression and commonly uses decimal values.
    "oldpeak": {"step": "0.01", "min": "0", "max": "6"},
}

def load_model_recall(path=MODEL_RESULTS_PATH):
    results = json.loads(path.read_text(encoding="utf-8"))
    return round(results["final_metrics"]["recall"] * 100, 1)


def load_model_precision(path=MODEL_RESULTS_PATH):
    results = json.loads(path.read_text(encoding="utf-8"))
    return round(results["final_metrics"]["precision"] * 100, 1)


def build_form_data(form: Any) -> dict:
    """Normalize posted form values into the model input schema."""
    form_data = {}
    for field, _ in FEATURE_FIELDS:
        value = form.get(field)
        if value is None:
            continue

        if field in ["age", "trestbps", "chol", "thalch", "oldpeak"]:
            form_data[field] = float(value)
        elif field in ["fbs", "exang"]:
            form_data[field] = value.strip().lower() in ["1", "true", "yes", "y"]
        else:
            form_data[field] = value.strip()

    return form_data


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error_message = None
    if request.method == "POST":
        try:
            form_data = build_form_data(request.form)
            model = load_model()
            result = predict_patient(form_data, model=model)
        except Exception:
            app.logger.exception("Prediction request failed.")
            error_message = (
                "Prediction could not be completed in the deployed environment. "
                "Check the Vercel function logs for the full traceback."
            )

    return render_template(
        "index.html",
        features=FEATURE_FIELDS,
        personal_fields=PERSONAL_FIELDS,
        blood_testing_fields=BLOOD_TESTING_FIELDS,
        physical_testing_fields=PHYSICAL_TESTING_FIELDS,
        field_labels=FIELD_LABELS,
        select_options=SELECT_OPTIONS,
        normal_ranges=NORMAL_RANGES,
        input_attributes=INPUT_ATTRIBUTES,
        model_recall=load_model_recall(),
        model_precision=load_model_precision(),
        result=result,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

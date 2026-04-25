"""Flask app for Heart Disease prediction demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, request, render_template, request

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
    ("dataset", "select"),
]

CLINICAL_FIELDS = [
    ("cp", "select"),
    ("trestbps", "number"),
    ("chol", "number"),
    ("fbs", "select"),
    ("restecg", "select"),
    ("thalch", "number"),
    ("exang", "select"),
    ("oldpeak", "number"),
]

FEATURE_FIELDS = PERSONAL_FIELDS + CLINICAL_FIELDS

SELECT_OPTIONS = {
    "sex": ["Male", "Female"],
    "dataset": ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"],
    "cp": ["asymptomatic", "atypical angina", "non-anginal", "typical angina"],
    "fbs": ["False", "True"],
    "restecg": ["lv hypertrophy", "normal", "st-t abnormality"],
    "exang": ["False", "True"],
}

NORMAL_RANGES = {
    "cp": "non-anginal",
    "trestbps": "~ 90-120 mm Hg",
    "chol": "< 200 mg/dL",
    "fbs": "False or <= 120 mg/dL",
    "restecg": "normal",
    "thalch": "varies by age ~ 150-210 bpm",
    "exang": "False",
    "oldpeak": "~ 0-1.5 (exercise-induced ST depression)",
}

FIELD_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "dataset": "Dataset",
    "cp": "CP",
    "trestbps": "Trestbps",
    "chol": "Chol",
    "fbs": "FBS",
    "restecg": "Restecg",
    "thalch": "Thalch",
    "exang": "Exang",
    "oldpeak": "Oldpeak",
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

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        form_data = {}
        for field, _ in FEATURE_FIELDS:
            value = request.form.get(field)
            if value is None:
                continue

            if field in ["age", "trestbps", "chol", "thalch", "oldpeak"]:
                form_data[field] = float(value)
            elif field in ["fbs", "exang"]:
                form_data[field] = value.strip().lower() in ["1", "true", "yes", "y"]
            else:
                form_data[field] = value.strip()

        model = load_model()
        result = predict_patient(form_data, model=model)

    return render_template(
        "index.html",
        features=FEATURE_FIELDS,
        personal_fields=PERSONAL_FIELDS,
        clinical_fields=CLINICAL_FIELDS,
        field_labels=FIELD_LABELS,
        select_options=SELECT_OPTIONS,
        normal_ranges=NORMAL_RANGES,
        input_attributes=INPUT_ATTRIBUTES,
        model_recall=load_model_recall(),
        model_precision=load_model_precision(),
        result=result,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

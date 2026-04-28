"""Prediction module for Heart Disease model."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

# Ensure local packages can be imported when running as script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "heart_disease_model.pkl"
MODEL_RESULTS_PATH = MODELS_DIR / "heart_disease_model_results.json"
DEFAULT_THRESHOLD = 0.30
EXPECTED_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
BOOLEAN_FEATURES = ["fbs", "exang"]


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with path.open("rb") as file:
        model = pickle.load(file)
    return model


def load_prediction(path: Path = MODEL_RESULTS_PATH) -> float:
    if not path.exists():
        return DEFAULT_THRESHOLD

    try:
        results = json.loads(path.read_text(encoding="utf-8"))
        return float(results.get("selected_threshold", DEFAULT_THRESHOLD))
    except (json.JSONDecodeError, TypeError, ValueError):
        return DEFAULT_THRESHOLD


def validate_patient_input(patient: dict) -> None:
    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in patient or patient[feature] in [None, ""]
    ]
    if missing_features:
        raise ValueError(f"Missing required patient features: {missing_features}")


def normalize_input_row(row: dict) -> dict:
    """Validate and standardize one patient row for model prediction."""
    validate_patient_input(row)
    normalized = {feature: row[feature] for feature in EXPECTED_FEATURES}

    for feature in NUMERIC_FEATURES:
        normalized[feature] = float(normalized[feature])

    if "sex" in normalized:
        sex = str(normalized["sex"]).strip().lower()
        normalized["sex"] = "Male" if sex in ["male", "m"] else "Female"

    for bool_field in BOOLEAN_FEATURES:
        value = normalized[bool_field]
        if isinstance(value, str):
            normalized[bool_field] = value.strip().lower() in ["1", "true", "t", "yes", "y"]
        else:
            normalized[bool_field] = bool(value)

    return normalized


def predict_patient(patient: dict, model=None, threshold: float | None = None) -> dict:
    if model is None:
        model = load_model()
    if threshold is None:
        threshold = load_prediction()

    patient_data = normalize_input_row(patient)
    feature_df = pd.DataFrame([patient_data], columns=EXPECTED_FEATURES)

    probability = float(model.predict_proba(feature_df)[0, 1])
    predicted_class = int(probability >= threshold)

    return {
        "prediction": predicted_class,
        "label": "Disease" if predicted_class == 1 else "No Disease",
        "probability": probability,
        "threshold": threshold,
        "risk_level": "High risk" if predicted_class == 1 else "Low risk",
        "note": "Educational decision support only; Not a medical diagnosis.",
    }


def main():
    sample = {
        "age": 58,
        "sex": "Male",
        "cp": "typical angina",
        "trestbps": 150,
        "chol": 260,
        "fbs": False,
        "restecg": "normal",
        "thalch": 150,
        "exang": False,
        "oldpeak": 1.5,
    }
    model = load_model()
    threshold = load_prediction()
    result = predict_patient(sample, threshold=threshold)
    print(result)


if __name__ == "__main__":
    main()

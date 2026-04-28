"""Model training for Heart Disease Prediction (FIXED - threshold-aware evaluation)."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ==============================
# PATH SETUP
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (
    load_clean_data,
    validate_clean_data,
    split_features_and_target,
    build_feature_preparation_pipeline,
)

MODEL_OUTPUT_PATH = Path("models/heart_disease_model.pkl")
MODEL_RESULTS_PATH = Path("models/heart_disease_model_results.json")

THRESHOLD_CANDIDATES = np.arange(0.10, 0.91, 0.01)
DEPLOYMENT_THRESHOLD = 0.23


# ==============================
# METRICS
# ==============================
def calculate_metrics(y_test, y_pred, y_proba=None):
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_test, y_pred, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if y_proba is not None and y_test.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

    return metrics


# ==============================
# THRESHOLD-AWARE EVALUATION
# ==============================
def evaluate_with_threshold(model, X_test, y_test, threshold):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return calculate_metrics(y_test, y_pred, y_proba)


# ==============================
# THRESHOLD TUNING
# ==============================
def evaluate_thresholds(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    results = []

    for threshold in THRESHOLD_CANDIDATES:
        threshold = round(float(threshold), 2)
        y_pred = (y_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_test, y_pred, y_proba)

        results.append({
            "threshold": threshold,
            "metrics": metrics,
        })

    return results


def choose_threshold(threshold_results):
    return max(
        threshold_results,
        key=lambda r: (
            r["metrics"]["f2"],
            r["metrics"]["recall"],
            r["metrics"]["f1"],
            r["metrics"]["precision"],
        ),
    )


def choose_model_and_threshold(threshold_evaluations):
    all_results = [
        {"model": m, **r}
        for m, res in threshold_evaluations.items()
        for r in res
    ]

    best = max(
        all_results,
        key=lambda r: (
            r["metrics"]["f2"],
            r["metrics"]["recall"],
            r["metrics"]["f1"],
            r["metrics"]["precision"],
        ),
    )

    return best["model"], best


def choose_best_model_at_fixed_threshold(evaluated_models):
    return max(
        evaluated_models.items(),
        key=lambda item: (
            item[1]["f2"],
            item[1]["recall"],
            item[1]["f1"],
            item[1]["precision"],
        ),
    )


# ==============================
# MAIN
# ==============================
def main():

    print("\n=== TRAINING START ===\n")

    df = load_clean_data()
    validate_clean_data(df)

    X, y = split_features_and_target(df)

    preprocessor = build_feature_preparation_pipeline()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # ==============================
    # MODELS
    # ==============================
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        ),
    }

    evaluated_models = {}

    # ==============================
    # TRAIN + THRESHOLD TUNING
    # ==============================
    for name, model in candidates.items():

        print(f"\n--- Training {name} ---")

        model.fit(X_train, y_train)

        # Evaluate every model at the same deployment threshold.
        evaluated_models[name] = evaluate_with_threshold(
            model, X_test, y_test, DEPLOYMENT_THRESHOLD
        )

        print(f"Deployment threshold: {DEPLOYMENT_THRESHOLD}")
        print(json.dumps(evaluated_models[name], indent=2))

    # ==============================
    # FINAL SELECTION
    # ==============================
    best_model_name, final_metrics = choose_best_model_at_fixed_threshold(evaluated_models)
    best_threshold = DEPLOYMENT_THRESHOLD
    best_model = candidates[best_model_name]

    print("\n=== BEST MODEL ===")
    print(best_model_name)
    print(f"Threshold: {best_threshold}")

    # ==============================
    # TRAIN FINAL PIPELINE
    # ==============================
    full_pipeline = Pipeline([
        ("preprocessor", build_feature_preparation_pipeline()),
        ("model", best_model),
    ])

    full_pipeline.fit(X, y)

    # ==============================
    # SAVE MODEL
    # ==============================
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(full_pipeline, f)

    # ==============================
    # SAVE RESULTS
    # ==============================
    results = {
        "best_model": best_model_name,
        "selected_threshold": best_threshold,
        "final_metrics": final_metrics,
        "all_models": evaluated_models,
    }

    MODEL_RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    print("\n=== FINAL METRICS ===")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()

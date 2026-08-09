"""Custom scoring entry script for the managed online endpoint.

Deployed via CodeConfiguration in src/deploy/deploy_endpoint.py, running
inside the environment defined by config/environments/score-env.yml.

Unlike a no-code MLflow deployment (which only returns the raw prediction),
this returns a per-request explanation: which features drove *this*
prediction, and by how much. That's possible cheaply and *exactly* (not an
approximation) because the deployed model is a linear LogisticRegression
behind a StandardScaler -- see compute_explanation() for the math.

Request format (POST body):
    {"data": [{"radius_mean": 17.99, "texture_mean": 10.38, ...}, ...]}
    (raw, unscaled feature values -- one dict per row; keys must match the
    feature names the model was trained on)

Response format:
    [
        {
            "prediction": "malignant",
            "probability": 0.94,
            "top_contributing_features": [
                {"feature": "concave_points_worst", "contribution": 1.31},
                ...
            ]
        },
        ...
    ]  (one entry per input row, same order)
"""
from __future__ import annotations

import json
import logging
import os

import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

TOP_N_FEATURES = 3

model: Pipeline | None = None
feature_names: list[str] | None = None


def find_mlflow_model_dir(root: str) -> str:
    """Locate the directory containing the MLmodel file under `root`.

    AZUREML_MODEL_DIR's exact nesting (whether the model files sit directly
    at the root or under a name/version subfolder) varies across AML SDK
    versions -- searching is more robust than hardcoding a guessed path.
    """
    for dirpath, _, filenames in os.walk(root):
        if "MLmodel" in filenames:
            return dirpath
    raise FileNotFoundError(f"No MLmodel file found under {root}")


def compute_explanation(pipeline: Pipeline, record: dict, names: list[str], top_n: int = TOP_N_FEATURES) -> dict:
    """Score one raw feature record and explain which features drove it.

    Exact (not approximate, unlike SHAP on non-linear models) because the
    classifier is linear: for a standardized input x, the logit is
    intercept + sum(coefficient_i * x_i), so each term is precisely that
    feature's contribution to the log-odds. Ranking by abs(contribution)
    surfaces what actually mattered for *this* prediction, not just which
    features have the largest coefficients in general (see
    src/train/train.py::log_feature_importance for the global version).
    """
    missing = [f for f in names if f not in record]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    row = pd.DataFrame([{f: record[f] for f in names}])
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    scaled = scaler.transform(row)
    probability = float(clf.predict_proba(scaled)[0][1])  # P(diagnosis == malignant)
    prediction = "malignant" if probability >= 0.5 else "benign"

    contributions = scaled[0] * clf.coef_[0]
    top_idx = sorted(range(len(contributions)), key=lambda i: abs(contributions[i]), reverse=True)[:top_n]
    top_features = [
        {"feature": names[i], "contribution": round(float(contributions[i]), 4)} for i in top_idx
    ]

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "top_contributing_features": top_features,
    }


def init() -> None:
    """Called once when the deployment container starts."""
    global model, feature_names
    model_root = os.getenv("AZUREML_MODEL_DIR")
    model_dir = find_mlflow_model_dir(model_root)
    model = mlflow.sklearn.load_model(model_dir)
    # StandardScaler fitted on a DataFrame retains the training-time column
    # names/order -- reuse it instead of hardcoding the feature list here.
    feature_names = list(model.named_steps["scaler"].feature_names_in_)
    logging.info(f"Loaded model from {model_dir} with {len(feature_names)} features")


def run(raw_data: str) -> list[dict] | dict:
    """Called for every request."""
    try:
        records = json.loads(raw_data)["data"]
        return [compute_explanation(model, record, feature_names) for record in records]
    except Exception as e:
        logging.exception("Scoring failed")
        return {"error": str(e)}

"""Training entry point.

Runnable locally for quick iteration:
    python -m src.train.train --data ./local_sample.csv

...or as the AML command job submitted by `scripts/submit_training.py`, where
`--data` is the mounted/downloaded data asset instead of a local file.

Trains a LogisticRegression on the Kaggle-format Breast Cancer Wisconsin CSV
(`diagnosis` column as target, `id`/`Unnamed: 32` dropped if present) and
logs it as an MLflow run artifact under the fixed path `model/`. This script
does NOT register the model into the workspace model registry -- that's a
separate, explicit step: run `src/register/register_model.py --job_name
<name>` afterwards (the job name is printed by `scripts/submit_training.py`).
Keeping training and registration separate means you can inspect a run's
metrics in the studio before deciding whether it's worth registering.

The scaler and classifier are bundled into a single sklearn Pipeline before
logging (not logged separately) so scaling happens automatically inside the
deployed model -- a deployment endpoint can be called with raw, unscaled
feature values instead of needing the client to replicate the exact
StandardScaler fit from training.

The data-loading/training/eval logic is split into small functions kept free
of argparse so it's directly unit-testable (see `tests/test_train.py`).
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless backend -- no display available on AML compute

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COL = "diagnosis"
# Fixed on purpose: register_model.py reads the artifact back from this exact
# path (azureml://jobs/<job_name>/outputs/artifacts/paths/model/), so it must
# match regardless of what the model gets named at registration time.
MODEL_ARTIFACT_PATH = "model"


def load_data(data_path: str) -> pd.DataFrame:
    """Load the raw CSV, dropping non-predictive Kaggle boilerplate columns."""
    df = pd.read_csv(data_path)
    cols_to_drop = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Encode the diagnosis target (M=1/B=0) and split into X, y."""
    df = df.copy()
    if df[TARGET_COL].dtype == "object":
        df[TARGET_COL] = df[TARGET_COL].map({"M": 1, "B": 0})
    y = df.pop(TARGET_COL)
    X = df
    return X, y


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000,
) -> tuple[Pipeline, dict]:
    """Fit a StandardScaler + LogisticRegression pipeline, return (pipeline, metrics).

    Scaler and classifier are bundled into one Pipeline (rather than fit and
    logged separately) so the fitted scaling is baked into the model object
    itself -- `pipeline.predict(raw_X)` scales internally, so callers (e.g. a
    deployed endpoint) never need to replicate the training-time scaling.

    class_weight="balanced" optimizes for high recall (minimizing false
    negatives), which matters more than raw accuracy for cancer detection.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C, max_iter=max_iter, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1_score": f1_score(y_test, y_pred),
    }

    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"]))
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    return pipeline, metrics


def log_feature_importance(pipeline: Pipeline, feature_names: list[str]) -> None:
    """Log ranked coefficients (global feature importance) as MLflow artifacts.

    Coefficients are directly comparable across features because the
    classifier sits after a StandardScaler in the pipeline -- every feature
    is on the same standardized scale, so abs(coefficient) ranks true
    influence on the prediction, not just raw feature magnitude. Positive
    coefficients push toward malignant (target=1); negative push toward
    benign. This is the same math score.py uses to explain individual
    predictions (see src/deploy/score.py::compute_explanation), just
    aggregated across the whole model instead of one request.
    """
    coef = pipeline.named_steps["clf"].coef_[0]
    importance = pd.Series(coef, index=feature_names).sort_values(key=abs, ascending=False)

    mlflow.log_dict(importance.round(4).to_dict(), "feature_importance.json")

    fig, ax = plt.subplots(figsize=(8, 6))
    importance.head(15).sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Standardized coefficient (log-odds toward malignant)")
    ax.set_title("Top 15 features by influence")
    fig.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.20)
    parser.add_argument("--C", required=False, default=1.0, type=float, help="Inverse regularization strength")
    parser.add_argument("--max_iter", required=False, default=1000, type=int, help="Maximum solver iterations")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    # enable autologging for params/metrics only -- log_models=False so it
    # doesn't also write to the "model/" artifact path itself; the explicit
    # mlflow.sklearn.log_model() call below is the sole writer to that path
    # (autolog + an explicit log_model to the same path causes a duplicate-
    # write "Resource Conflict" error from the AML artifact store).
    mlflow.sklearn.autolog(log_models=False)

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.data)

    df = load_data(args.data)
    X, y = prepare_features(df)

    mlflow.log_metric("num_samples", df.shape[0])
    mlflow.log_metric("num_features", X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_train_ratio, random_state=42, stratify=y
    )

    print(f"Training with data of shape {X_train.shape}")

    pipeline, metrics = train_and_evaluate(
        X_train, X_test, y_train, y_test, C=args.C, max_iter=args.max_iter
    )
    mlflow.log_metrics(metrics)

    # Log the scaler+classifier pipeline as a run artifact only -- NOT
    # registered here.
    print(f"Logging model to run artifact path '{MODEL_ARTIFACT_PATH}/'")
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path=MODEL_ARTIFACT_PATH)

    log_feature_importance(pipeline, list(X.columns))

    run_id = mlflow.active_run().info.run_id
    print(f"Done. Run id: {run_id}")
    print(
        f"To register this model, run:\n"
        f"  python -m src.register.register_model --job_name {run_id}"
    )

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()

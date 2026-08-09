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

The data-loading/training/eval logic is split into small functions kept free
of argparse so it's directly unit-testable (see `tests/test_train.py`).
"""
from __future__ import annotations

import argparse

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
) -> tuple[LogisticRegression, StandardScaler, dict]:
    """Scale features, fit a LogisticRegression, return (model, scaler, metrics).

    class_weight="balanced" optimizes for high recall (minimizing false
    negatives), which matters more than raw accuracy for cancer detection.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(
        C=C, max_iter=max_iter, class_weight="balanced", random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
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

    return clf, scaler, metrics


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

    clf, scaler, metrics = train_and_evaluate(
        X_train, X_test, y_train, y_test, C=args.C, max_iter=args.max_iter
    )
    mlflow.log_metrics(metrics)

    # Log the model as a run artifact only -- NOT registered here.
    print(f"Logging model to run artifact path '{MODEL_ARTIFACT_PATH}/'")
    mlflow.sklearn.log_model(sk_model=clf, artifact_path=MODEL_ARTIFACT_PATH)

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

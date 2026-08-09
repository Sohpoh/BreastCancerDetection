"""Offline unit tests for the training logic -- no AML/network calls, so
these run in CI without workspace credentials."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.train.train import TARGET_COL, load_data, prepare_features, train_and_evaluate


def _make_synthetic_df(n_rows: int = 200, n_features: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    # Make the label loosely dependent on the features so training isn't
    # degenerate, but keep it noisy/synthetic -- this is not real data.
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.normal(scale=0.5, size=n_rows) > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df[TARGET_COL] = np.where(y == 1, "M", "B")
    return df


def test_load_data_drops_kaggle_boilerplate_columns(tmp_path):
    df = _make_synthetic_df()
    df.insert(0, "id", range(len(df)))
    df["Unnamed: 32"] = np.nan
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_data(str(csv_path))

    assert "id" not in loaded.columns
    assert "Unnamed: 32" not in loaded.columns


def test_prepare_features_encodes_diagnosis_and_splits_target():
    df = _make_synthetic_df()

    X, y = prepare_features(df)

    assert TARGET_COL not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_train_and_evaluate_returns_fitted_model_and_expected_metrics():
    df = _make_synthetic_df()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf, scaler, metrics = train_and_evaluate(
        X_train, X_test, y_train, y_test, C=1.0, max_iter=200
    )

    assert hasattr(clf, "predict")
    assert hasattr(scaler, "transform")
    for key in ["test_accuracy", "test_precision", "test_recall", "test_f1_score"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

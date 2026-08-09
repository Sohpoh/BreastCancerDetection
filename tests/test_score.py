"""Offline unit tests for the per-prediction explanation logic in score.py --
no AML/network calls, and no AZUREML_MODEL_DIR needed (init() is not
exercised here, only the pure compute_explanation() function)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.deploy.score import compute_explanation
from src.train.train import TARGET_COL, prepare_features, train_and_evaluate


def _fitted_pipeline_and_features():
    """Reuse the real training path so the pipeline this test explains is
    structured exactly like the one score.py will actually load."""
    rng = np.random.default_rng(0)
    n_rows, n_features = 200, 4
    X = rng.normal(size=(n_rows, n_features))
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.normal(scale=0.5, size=n_rows) > 0).astype(int)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df[TARGET_COL] = np.where(y == 1, "M", "B")

    X_df, y_ser = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_ser, test_size=0.25, random_state=42, stratify=y_ser
    )
    pipeline, _ = train_and_evaluate(X_train, X_test, y_train, y_test, C=1.0, max_iter=200)
    return pipeline, feature_names, X_test


def test_compute_explanation_returns_expected_shape():
    pipeline, feature_names, X_test = _fitted_pipeline_and_features()
    record = X_test.iloc[0].to_dict()

    result = compute_explanation(pipeline, record, feature_names, top_n=2)

    assert result["prediction"] in {"malignant", "benign"}
    assert 0.0 <= result["probability"] <= 1.0
    assert len(result["top_contributing_features"]) == 2
    for entry in result["top_contributing_features"]:
        assert set(entry) == {"feature", "contribution"}
        assert entry["feature"] in feature_names


def test_compute_explanation_matches_pipeline_predict_proba():
    pipeline, feature_names, X_test = _fitted_pipeline_and_features()
    record = X_test.iloc[0].to_dict()

    result = compute_explanation(pipeline, record, feature_names)

    expected_probability = pipeline.predict_proba(X_test.iloc[[0]])[0][1]
    assert result["probability"] == pytest.approx(expected_probability, abs=1e-4)


def test_compute_explanation_contributions_sum_close_to_logit():
    """Sanity-check the actual math: sum(contributions) + intercept should
    reconstruct the same logit predict_proba is derived from."""
    pipeline, feature_names, X_test = _fitted_pipeline_and_features()
    record = X_test.iloc[0].to_dict()
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    row = pd.DataFrame([{f: record[f] for f in feature_names}])
    scaled = scaler.transform(row)
    full_contributions = scaled[0] * clf.coef_[0]
    logit = full_contributions.sum() + clf.intercept_[0]
    expected_probability = 1 / (1 + np.exp(-logit))

    result = compute_explanation(pipeline, record, feature_names, top_n=len(feature_names))
    assert result["probability"] == pytest.approx(expected_probability, abs=1e-4)


def test_compute_explanation_raises_on_missing_feature():
    pipeline, feature_names, X_test = _fitted_pipeline_and_features()
    record = X_test.iloc[0].to_dict()
    del record[feature_names[0]]

    with pytest.raises(ValueError, match="Missing required feature"):
        compute_explanation(pipeline, record, feature_names)

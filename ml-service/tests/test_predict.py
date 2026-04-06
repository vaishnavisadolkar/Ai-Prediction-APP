"""
Unit tests for ``validate_prediction_payload`` and ``run_prediction``.

``run_prediction`` uses a monkeypatched pipeline so assertions do not depend
on the conftest dummy weights.
"""
import numpy as np
import pytest

from ml_service.predict import PredictionValidationError, run_prediction, validate_prediction_payload


def test_validate_empty_payload():
    """Empty dict must raise PredictionValidationError."""
    with pytest.raises(PredictionValidationError):
        validate_prediction_payload({})


def test_validate_missing_bmi_and_status():
    with pytest.raises(PredictionValidationError):
        validate_prediction_payload({"glucose": 100})


def test_run_prediction_with_mock_model(monkeypatch):
    """Full happy path with fake predict/predict_proba on the global pipeline."""
    class FakePipeline:
        def predict(self, X):
            return np.array([1])

        def predict_proba(self, X):
            return np.array([[0.2, 0.8]])

    import ml_service.model as model_mod

    monkeypatch.setattr(model_mod, "_pipeline", FakePipeline(), raising=False)

    out = run_prediction(
        {
            "Pregnancies": 1,
            "Glucose": 150.0,
            "BloodPressure": 70.0,
            "SkinThickness": 20.0,
            "Insulin": 80.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 35,
            "BMI": 28.0,
        }
    )
    assert out["outcome"] == 1
    assert out["probability_no_diabetes"] == 0.2
    assert out["probability_diabetes"] == 0.8

"""
Prediction entry point (no HTTP).

Validates the feature dict, builds a DataFrame via preprocessing, and calls
the in-process sklearn Pipeline loaded at startup.
"""
from typing import Any

from ml_service.model import predict as model_predict
from ml_service.model import predict_proba as model_predict_proba
from ml_service.preprocessing import build_input_df_from_dict


class PredictionValidationError(ValueError):
    """Raised when the request body cannot be turned into model input (HTTP 422)."""


def validate_prediction_payload(payload: dict[str, Any]) -> None:
    """Require non-empty payload and either BMI or Nutritional_Status."""
    if not payload:
        raise PredictionValidationError("At least one feature required")
    payload_lower = {k.lower(): v for k, v in payload.items()}
    if "bmi" not in payload_lower and "nutritional_status" not in payload_lower:
        raise PredictionValidationError("Provide either BMI or Nutritional_Status")


def run_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Returns:
        Dict with outcome (0/1) and rounded class probabilities.

    Raises:
        PredictionValidationError: If validation fails.
    """
    validate_prediction_payload(payload)
    features_df = build_input_df_from_dict(payload)
    predictions = model_predict(features_df)
    probas = model_predict_proba(features_df)
    outcome = int(predictions[0])
    p0, p1 = probas[0][0], probas[0][1]
    return {
        "outcome": outcome,
        "probability_no_diabetes": round(p0, 4),
        "probability_diabetes": round(p1, 4),
    }

"""
Central configuration for paths, API settings, and feature metadata.

Loads optional ``.env`` from the ``ml-service`` root (parent of ``src/``) so
local overrides do not require shell exports. Environment variables:

- ``MODEL_VERSION``: subdirectory under ``models/`` (default ``v1``).
- ``MODEL_PATH``: full path to the joblib artifact; if unset, defaults to
  ``{BASE_DIR}/models/{MODEL_VERSION}/best_diabetes_prediction.joblib``.
- ``DATA_PATH``: CSV used by ``python -m ml_service.train``.
- ``LOG_LEVEL``, ``API_HOST``, ``API_PORT``: logging and server hints.
- ``CORS_ORIGINS``: comma-separated list; empty means allow all origins in API.
- ``OUTPUT_MODEL_PATH``: optional training output path (else ``MODEL_PATH``).
- ``MODEL_S3_URI``: optional ``s3://bucket/key/to/model.joblib``; if set, download at
  startup to ``MODEL_CACHE_DIR`` (``MODEL_PATH`` is ignored for loading).
- ``MODEL_CACHE_DIR``: folder for the downloaded file (default: temp dir ``ml_service_model``).
"""
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Resolve ml-service root: .../src/ml_service/config.py -> three parents up
_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_SERVICE_ROOT / ".env")

BASE_DIR = _SERVICE_ROOT

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1").strip() or "v1"

_default_model = BASE_DIR / "models" / MODEL_VERSION / "best_diabetes_prediction.joblib"
MODEL_PATH = os.getenv("MODEL_PATH", str(_default_model))

MODEL_S3_URI = os.getenv("MODEL_S3_URI", "").strip()
_default_cache_dir = Path(tempfile.gettempdir()) / "ml_service_model"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(_default_cache_dir))

DATA_PATH = os.getenv("DATA_PATH", str(BASE_DIR / "data" / "diabetes.csv"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


def cors_origins() -> list[str] | None:
    """
    Parse ``CORS_ORIGINS`` for FastAPI.

    Returns:
        List of allowed origin strings, or ``None`` if unset (caller may use ``*``).
    """
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if not raw:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


# Column order must match the training DataFrame passed to the sklearn pipeline.
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "DiabetesPedigreeFunction",
    "Age",
    "Nutritional_Status",
]

VALID_NUTRITIONAL_STATUS = {"Underweight", "Normal", "Overweight", "Obese"}

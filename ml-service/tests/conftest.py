"""
Pytest hooks and shared fixtures.

Writes a minimal joblib artifact and sets MODEL_PATH before importing the app or
ml_service.config, so tests never depend on a real trained model or a local
.env MODEL_PATH.
"""
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest

# Create dummy artifact before tests import ml_service.config (via api.app)
_dummy_dir = tempfile.mkdtemp()
_dummy_path = Path(_dummy_dir) / "pytest_model.joblib"


class _DummyModel:
    """Minimal sklearn-like API for load_model() during app lifespan."""

    def predict(self, X):
        return np.array([0])

    def predict_proba(self, X):
        return np.array([[0.72, 0.28]])


joblib.dump(_DummyModel(), _dummy_path)
# Force test artifact so a local .env MODEL_PATH does not break CI
os.environ["MODEL_PATH"] = str(_dummy_path)
os.environ.pop("MODEL_S3_URI", None)


@pytest.fixture
def client():
    """HTTP client against the FastAPI app (uses dummy model from env)."""
    from fastapi.testclient import TestClient
    from api.app import app

    with TestClient(app) as c:
        yield c

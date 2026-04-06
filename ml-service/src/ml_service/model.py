"""
Load and run the serialized sklearn ``Pipeline`` (preprocess + classifier).

The pipeline is loaded once at application startup (``load_model``) and
held in module state. Inference functions call ``get_model`` so the joblib
file is not re-read on every request.

If ``MODEL_S3_URI`` is set (``s3://bucket/key``), boto3 downloads the object
into ``MODEL_CACHE_DIR`` on each startup, then loads from disk. Credentials
use the default AWS chain (env vars ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``,
``AWS_DEFAULT_REGION``, or instance metadata when reachable from the process).
"""
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_service.config import (
    MODEL_CACHE_DIR,
    MODEL_PATH,
    MODEL_S3_URI,
    MODEL_VERSION,
)

_pipeline: Any = None


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"MODEL_S3_URI must start with s3://, got {uri!r}")
    rest = uri[5:].lstrip("/")
    if "/" not in rest:
        raise ValueError(f"MODEL_S3_URI must include object key: {uri!r}")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid MODEL_S3_URI: {uri!r}")
    return bucket, key


def _download_s3_to_path(uri: str, dest: Path) -> None:
    import boto3
    from botocore.exceptions import ClientError

    bucket, key = _parse_s3_uri(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    client = boto3.client("s3")
    try:
        client.download_file(bucket, key, str(tmp))
    except ClientError as e:
        raise RuntimeError(f"S3 download failed for {uri}: {e}") from e
    tmp.replace(dest)


def _resolved_model_path() -> Path:
    if MODEL_S3_URI:
        name = Path(_parse_s3_uri(MODEL_S3_URI)[1]).name or "model.joblib"
        return Path(MODEL_CACHE_DIR) / name
    return Path(MODEL_PATH)


def load_model() -> None:
    """
    Load the joblib pipeline from local ``MODEL_PATH`` or from ``MODEL_S3_URI``.

    Raises:
        FileNotFoundError: If using local path and the file is missing.
        ValueError: If ``MODEL_S3_URI`` is malformed.
        RuntimeError: If S3 download fails.
    """
    global _pipeline
    if MODEL_S3_URI:
        local = _resolved_model_path()
        _download_s3_to_path(MODEL_S3_URI, local)
        path = local
    else:
        path = Path(MODEL_PATH)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    _pipeline = joblib.load(path)


def get_model() -> Any:
    """
    Return the in-memory pipeline.

    Raises:
        RuntimeError: If :func:`load_model` has not been called successfully.
    """
    if _pipeline is None:
        raise RuntimeError("Model not loaded; call load_model() at startup")
    return _pipeline


def predict(features_df: pd.DataFrame) -> list[int]:
    """
    Predict class labels (0 = no diabetes, 1 = diabetes) per row.

    Args:
        features_df: Must match training column schema (see ``FEATURE_COLUMNS``).
    """
    pipeline = get_model()
    return pipeline.predict(features_df).tolist()


def predict_proba(features_df: pd.DataFrame) -> list[list[float]]:
    """
    Predict class probabilities ``[P(class 0), P(class 1)]`` per row.

    Args:
        features_df: Same as :func:`predict`.
    """
    pipeline = get_model()
    return pipeline.predict_proba(features_df).tolist()


def model_version_label() -> str:
    """Return ``MODEL_VERSION`` from config for structured logs."""
    return MODEL_VERSION

"""
FastAPI application: health check and diabetes risk prediction.

Run with (from ``ml-service`` after ``pip install -e .``)::

    python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

Startup loads the joblib pipeline once; routes delegate scoring to
``ml_service.predict.run_prediction``.
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictRequest, PredictResponse
from ml_service.config import LOG_LEVEL, MODEL_PATH, MODEL_S3_URI, cors_origins
from ml_service.model import load_model, model_version_label
from ml_service.predict import PredictionValidationError, run_prediction

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model before accepting traffic; log path and version on success.

    Raises:
        FileNotFoundError: Local ``MODEL_PATH`` missing.
        RuntimeError / ValueError: S3 download or URI errors.
    """
    try:
        load_model()
        if MODEL_S3_URI:
            logger.info(
                "Model loaded from S3 uri=%s version=%s",
                MODEL_S3_URI,
                model_version_label(),
            )
        else:
            logger.info(
                "Model loaded successfully path=%s version=%s",
                MODEL_PATH,
                model_version_label(),
            )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.error("%s", e)
        raise
    yield
    logger.info("Shutting down")


app = FastAPI(title="Diabetes Prediction API", lifespan=lifespan)

_origins = cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Liveness probe; returns 200 when the app process is up (model already loaded at startup)."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: Request, body: PredictRequest) -> dict[str, Any]:
    """
    Validate body, run inference, return probabilities.

    Returns:
        JSON matching ``PredictResponse``.

    Raises:
        HTTPException: 422 for validation errors, 500 for unexpected inference errors.
    """
    start = time.perf_counter()
    request_id = request.headers.get("x-request-id", "")
    try:
        payload = body.model_dump(by_alias=False, exclude_none=True)
        result = run_prediction(payload)
        elapsed = time.perf_counter() - start
        logger.info(
            "request_id=%s path=/predict outcome=%s elapsed_sec=%.4f",
            request_id,
            result["outcome"],
            elapsed,
        )
        return result
    except PredictionValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("request_id=%s path=/predict error=%s", request_id, str(e))
        raise HTTPException(status_code=500, detail="Inference failed") from e

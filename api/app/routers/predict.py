import logging
import time

from fastapi import APIRouter, HTTPException

from app.core.metrics import (
    PREDICTION_CONFIDENCE_SCORE,
    PREDICTION_LATENCY_SECONDS,
    PREDICTION_REQUESTS_TOTAL,
)
from app.models.schemas import PredictRequest, PredictResponse
from app.services.inference import predict_sentiment

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest) -> PredictResponse:
    PREDICTION_REQUESTS_TOTAL.inc()
    start = time.perf_counter()
    try:
        label, score = predict_sentiment(payload.text)
        logger.info(f"Prediction: label={label} score={score:.4f}")
        PREDICTION_CONFIDENCE_SCORE.observe(score)
        return PredictResponse(label=label, score=score)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    finally:
        PREDICTION_LATENCY_SECONDS.observe(time.perf_counter() - start)

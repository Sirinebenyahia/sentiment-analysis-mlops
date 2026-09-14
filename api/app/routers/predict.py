import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import PredictRequest, PredictResponse
from app.services.inference import predict_sentiment

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        label, score = predict_sentiment(payload.text)
        logger.info(f"Prediction: label={label} score={score:.4f}")
        return PredictResponse(label=label, score=score)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
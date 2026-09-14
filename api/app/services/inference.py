import logging

from transformers import pipeline

logger = logging.getLogger(__name__)

_classifier = None


def load_classifier() -> None:
    """Preload the model. Called once at application startup."""
    global _classifier
    if _classifier is None:
        logger.info("Loading sentiment-analysis pipeline...")
        _classifier = pipeline("sentiment-analysis")
        logger.info("Pipeline loaded successfully.")


def is_ready() -> bool:
    return _classifier is not None


def predict_sentiment(text: str) -> tuple[str, float]:
    if _classifier is None:
        raise RuntimeError("Classifier not loaded yet")
    result = _classifier(text, truncation=True, max_length=512)[0]
    return result["label"], float(result["score"])
from prometheus_client import Counter, Histogram, make_asgi_app

PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total",
    "Total number of prediction requests received",
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Time spent processing a prediction request",
)

PREDICTION_CONFIDENCE_SCORE = Histogram(
    "prediction_confidence_score",
    "Distribution of confidence scores returned by the model",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

metrics_app = make_asgi_app()

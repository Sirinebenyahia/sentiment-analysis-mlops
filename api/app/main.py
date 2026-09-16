import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.metrics import metrics_app
from app.routers import health, predict
from app.services.inference import load_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: preloading model...")
    load_classifier()
    logger.info("Startup complete — ready to serve requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-style API serving a Hugging Face sentiment classifier.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router)
app.mount("/metrics", metrics_app)

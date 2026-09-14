from fastapi import APIRouter, Response, status

from app.services.inference import is_ready

router = APIRouter()


@router.get("/health", tags=["ops"])
def health(response: Response) -> dict:
    ready = is_ready()
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if ready else "loading"}
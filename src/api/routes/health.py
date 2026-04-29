from fastapi import APIRouter
from src.api.schemas import HealthResponse
from src.api import dependencies
from src.api.logging import build_logger

router = APIRouter()
logger = build_logger("churn_api.health")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    model_loaded = "pipeline" in dependencies.MODEL_ARTIFACTS
    logger.info("Health check", extra={"model_loaded": model_loaded})

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_metadata=dependencies.MODEL_ARTIFACTS.get("metadata"),
    )
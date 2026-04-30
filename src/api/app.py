# Para rodar:
#   uvicorn src.api.app:app --reload --port 8000

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api import dependencies
from src.api.middleware import latency_middleware
from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router
from src.api.logging import build_logger

logger = build_logger("churn_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: carrega modelo. Shutdown: libera memória."""
    dependencies.load_artifacts()
    yield
    dependencies.clear_artifacts()


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predição de cancelamento via MLP PyTorch + baselines sklearn",
    version="1.0.0",
    lifespan=lifespan,
)


app.middleware("http")(latency_middleware)


app.include_router(health_router, tags=["Infra"])
app.include_router(predict_router, tags=["Inference"])


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Exceção não tratada",
        extra={"path": request.url.path, "error": str(exc)},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno do servidor", "error": str(exc)},
    )

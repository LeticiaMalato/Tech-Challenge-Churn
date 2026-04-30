# src/api/middleware.py
# Middleware de latência — registra tempo de resposta de cada request HTTP.
# Registrado em app.py via app.middleware("http").

import time
from fastapi import Request
from src.api.logging import build_logger

logger = build_logger("churn_api.middleware")


async def latency_middleware(request: Request, call_next):
    start = time.perf_counter()

    logger.info(
        "Request recebido",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
        },
    )

    response = await call_next(request)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Process-Time"] = f"{elapsed_ms}ms"

    logger.info(
        "Request processado",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
        },
    )

    return response

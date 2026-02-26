from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.config import settings
from app.routers import health, jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Quant API",
        description="Submit a HuggingFace repo to quantize and upload GGUF variants.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(jobs.router)
    return app


app = create_app()


def serve() -> None:
    """Entry point for the ``quant-api`` console script and ``py -m app``."""
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)

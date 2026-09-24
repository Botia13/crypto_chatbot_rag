from contextlib import asynccontextmanager
from collections.abc import Callable

from fastapi import FastAPI, HTTPException, Request

from RAG_model.api.schemas import QueryRequest, QueryResponse
from RAG_model.core.settings import get_settings
from RAG_model.service.rag_service import (
    ServiceBusyError,
    get_rag_service,
)


def create_fastapi_app(
    service_factory: Callable = get_rag_service,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag_service = service_factory()
        yield

    app = FastAPI(
        title="Crypto SEC Filings RAG API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready(request: Request):
        service = request.app.state.rag_service
        settings = get_settings()

        details = service.qdrant_client.get_collection(
            settings.qdrant_collection
        )

        if not details.points_count:
            raise HTTPException(
                status_code=503,
                detail="Retrieval index is empty",
            )

        return {
            "status": "ok",
            "collection": settings.qdrant_collection,
            "points": details.points_count,
        }

    @app.get("/api/v1/system-info")
    def system_info(request: Request):
        return request.app.state.rag_service.run_config

    @app.post("/api/v1/query", response_model=QueryResponse)
    def query(payload: QueryRequest, request: Request):
        service = request.app.state.rag_service
        settings = get_settings()

        history = [
            message.model_dump()
            for message in payload.history[-settings.max_history_messages :]
        ]

        try:
            result = service.query(
                question=payload.question.strip(),
                history=history,
            )
        except ServiceBusyError as error:
            raise HTTPException(
                status_code=429,
                detail="The service is busy. Try again shortly.",
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail="The RAG request could not be completed.",
            ) from error

        return QueryResponse.model_validate(result)

    return app
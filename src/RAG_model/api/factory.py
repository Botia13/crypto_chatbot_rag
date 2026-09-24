import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from RAG_model.api.schemas import QueryRequest, QueryResponse
from RAG_model.core.logging import configure_logging
from RAG_model.core.runtime_index import close_runtime_qdrant
from RAG_model.core.settings import get_settings
from RAG_model.service.rag_service import (
    InvalidRequestError,
    ServiceBusyError,
    get_rag_service,
)

logger = logging.getLogger(__name__)


def create_fastapi_app(
    service_factory: Callable = get_rag_service,
) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = service_factory()
        app.state.rag_service = service
        try:
            yield
        finally:
            close = getattr(service, "close", None)
            if close is not None:
                close()
            get_rag_service.cache_clear()
            close_runtime_qdrant()

    app = FastAPI(
        title="Crypto SEC Filings RAG API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request.state.request_id = f"req_{uuid4().hex}"
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        if request.url.path == "/api/v1/query":
            logger.warning(
                "rag_request_rejected",
                extra={
                    "request_id": request.state.request_id,
                    "status_code": 422,
                    "error_type": type(error).__name__,
                },
            )
        return await request_validation_exception_handler(request, error)

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready(request: Request):
        service = request.app.state.rag_service
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
        history = [message.model_dump() for message in payload.history]

        try:
            result = service.query(
                question=payload.question,
                history=history,
                request_id=request.state.request_id,
            )
            return QueryResponse.model_validate(result)
        except InvalidRequestError as error:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "The request is invalid.",
                    "request_id": request.state.request_id,
                },
            ) from error
        except ServiceBusyError as error:
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "The service is busy. Try again shortly.",
                    "request_id": request.state.request_id,
                },
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "The RAG request could not be completed.",
                    "request_id": request.state.request_id,
                },
            ) from error

    return app

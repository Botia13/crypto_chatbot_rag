import logging
from copy import deepcopy
from functools import lru_cache
from threading import BoundedSemaphore
from uuid import uuid4

from RAG_model.answer.answer import answer
from RAG_model.core.runtime_index import get_runtime_qdrant
from RAG_model.core.settings import get_settings
from RAG_model.ingestion.config import BASELINE_RUN_CONFIG
from RAG_model.ingestion.embedding import create_openrouter_client

logger = logging.getLogger(__name__)


class ServiceBusyError(RuntimeError):
    pass


class InvalidRequestError(ValueError):
    pass


class RAGService:
    def __init__(
        self,
        qdrant_client,
        provider_client,
        run_config: dict,
        max_concurrency: int,
        max_question_length: int,
        max_history_messages: int,
    ) -> None:
        self.qdrant_client = qdrant_client
        self.provider_client = provider_client
        self.run_config = deepcopy(run_config)
        self.capacity = BoundedSemaphore(max_concurrency)
        self.max_question_length = max_question_length
        self.max_history_messages = max_history_messages

    def close(self) -> None:
        """Close resources owned directly by this service instance."""
        self.provider_client.close()

    def query(
        self,
        question: str,
        history: list[dict] | None = None,
        request_id: str | None = None,
    ) -> dict:
        request_id = request_id or f"req_{uuid4().hex}"
        question = question.strip()
        history = history or []

        if (
            not question
            or len(question) > self.max_question_length
            or len(history) > self.max_history_messages
        ):
            logger.warning(
                "rag_request_rejected",
                extra={
                    "request_id": request_id,
                    "status_code": 422,
                    "error_type": "InvalidRequestError",
                },
            )
            raise InvalidRequestError("Invalid RAG request.")

        acquired = self.capacity.acquire(blocking=False)

        if not acquired:
            logger.warning(
                "rag_request_rejected",
                extra={
                    "request_id": request_id,
                    "status_code": 429,
                    "error_type": "ServiceBusyError",
                },
            )
            raise ServiceBusyError("Service is currently busy.")

        try:
            raw = answer(
                question=question,
                run_config=self.run_config,
                qdrant_client=self.qdrant_client,
                provider_client=self.provider_client,
                history=history,
            )

            chunks = raw["Retrieved Chunk texts"]
            chunks_by_id = {
                chunk["chunk_id"]: chunk
                for chunk in chunks
                if chunk.get("chunk_id")
            }

            citation_ids = raw["Citations"]
            citations = []

            for citation_id in citation_ids:
                chunk = chunks_by_id.get(citation_id)

                if chunk:
                    citations.append(
                        {
                            "chunk_id": citation_id,
                            "ticker": chunk.get("ticker"),
                            "filing_date": chunk.get("filing_date"),
                            "section": chunk.get("section_title"),
                            "source_url": chunk.get("source_url"),
                        }
                    )

            result = {
                "request_id": request_id,
                "answer": raw["Answer"],
                "citations": citations,
                "citation_ids": citation_ids,
                "usage": raw["Token_usage"],
                "timings_ms": raw["Latency"],
                "retrieval": {
                    "retrieved_count": len(chunks),
                    "scores": raw["Similarity Scores"],
                },
                "validation": {
                    "citations_present": bool(citation_ids),
                    "citations_resolved": all(
                        citation_id in chunks_by_id
                        for citation_id in citation_ids
                    ),
                },
                "system": {
                    "pipeline_version": self.run_config["pipeline_version"],
                    "generation_model": self.run_config["generation_model"],
                    "embedding_model": self.run_config["embedding_model"],
                    "prompt_version": self.run_config["prompt_version"],
                    "collection": self.run_config["collection_name"],
                },
                "retrieved_chunks": chunks,
            }
        except Exception as error:
            logger.error(
                "rag_request_failed",
                extra={
                    "request_id": request_id,
                    "status_code": 502,
                    "error_type": type(error).__name__,
                },
            )
            raise
        finally:
            self.capacity.release()

        logger.info(
            "rag_request_completed",
            extra={
                "request_id": result["request_id"],
                "total_latency_ms": result["timings_ms"]["total"],
                "total_tokens": result["usage"].get("rag_total_tokens"),
                "citations_resolved": result["validation"][
                    "citations_resolved"
                ],
                "pipeline_version": result["system"]["pipeline_version"],
                "status_code": 200,
            },
        )
        return result


def create_rag_service() -> RAGService:
    settings = get_settings()
    runtime = get_runtime_qdrant()

    run_config = deepcopy(BASELINE_RUN_CONFIG)
    run_config["collection_name"] = settings.qdrant_collection
    run_config["max_generation_tokens"] = settings.max_generation_tokens
    run_config["max_query_rewrite_tokens"] = (
        settings.max_query_rewrite_tokens
    )

    return RAGService(
        qdrant_client=runtime.client,
        provider_client=create_openrouter_client(
            api_key=settings.openrouter_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
            max_retries=0,
        ),
        run_config=run_config,
        max_concurrency=settings.max_concurrent_requests,
        max_question_length=settings.max_question_length,
        max_history_messages=settings.max_history_messages,
    )

@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return create_rag_service()


get_rag_service_cached = get_rag_service

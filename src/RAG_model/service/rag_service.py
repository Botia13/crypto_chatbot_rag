from copy import deepcopy
from threading import BoundedSemaphore
from uuid import uuid4

from RAG_model.answer.answer import answer
from RAG_model.core.runtime_index import get_runtime_qdrant
from RAG_model.core.settings import get_settings
from RAG_model.ingestion.config import BASELINE_RUN_CONFIG
from RAG_model.ingestion.embedding import create_openrouter_client
from functools import lru_cache


class ServiceBusyError(RuntimeError):
    pass


class RAGService:
    def __init__(
        self,
        qdrant_client,
        provider_client,
        run_config: dict,
        max_concurrency: int,
    ) -> None:
        self.qdrant_client = qdrant_client
        self.provider_client = provider_client
        self.run_config = deepcopy(run_config)
        self.capacity = BoundedSemaphore(max_concurrency)

    def close(self) -> None:
        """Close resources owned directly by this service instance."""
        self.provider_client.close()

    def query(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> dict:
        acquired = self.capacity.acquire(
            blocking=True,
            timeout=10,
        )

        if not acquired:
            raise ServiceBusyError("Service is currently busy.")

        try:
            raw = answer(
                question=question,
                run_config=self.run_config,
                qdrant_client=self.qdrant_client,
                provider_client=self.provider_client,
                history=history or [],
            )
        finally:
            self.capacity.release()

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

        return {
            "request_id": f"req_{uuid4().hex}",
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


def create_rag_service() -> RAGService:
    settings = get_settings()
    runtime = get_runtime_qdrant()

    run_config = deepcopy(BASELINE_RUN_CONFIG)
    run_config["collection_name"] = settings.qdrant_collection
    run_config["max_generation_tokens"] = settings.max_generation_tokens

    return RAGService(
        qdrant_client=runtime.client,
        provider_client=create_openrouter_client(),
        run_config=run_config,
        max_concurrency=settings.max_concurrent_requests,
    )




@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return create_rag_service()


get_rag_service_cached = get_rag_service

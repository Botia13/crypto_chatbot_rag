from functools import lru_cache

from RAG_model.service.rag_service import RAGService, create_rag_service


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return create_rag_service()


get_rag_service_cached = get_rag_service

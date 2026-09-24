"""Manually exercise one production RAG request against the packaged index."""

from __future__ import annotations

import json

from RAG_model.core.runtime_index import get_runtime_qdrant
from RAG_model.core.settings import PROJECT_ROOT, get_settings
from RAG_model.service.rag_service import create_rag_service

QUESTION = (
    "What is IBIT?"
)


def main() -> None:
    settings = get_settings()
    expected_packaged_path = (
        PROJECT_ROOT / "artifacts" / "qdrant_runtime"
    ).resolve()
    packaged_path = settings.qdrant_packaged_path.resolve()

    if packaged_path != expected_packaged_path:
        raise RuntimeError(
            "Smoke check must use the packaged production index: "
            f"expected={expected_packaged_path}, configured={packaged_path}."
        )

    runtime = get_runtime_qdrant()
    service = create_rag_service()
    qdrant_client_id = id(service.qdrant_client)

    try:
        result = service.query(QUESTION)

        if id(service.qdrant_client) != qdrant_client_id:
            raise RuntimeError("RAG service replaced its Qdrant client during query.")
        if service.qdrant_client is not runtime.client:
            raise RuntimeError("RAG service is not using the cached runtime client.")
        if not result["citation_ids"] or not result["citations"]:
            raise RuntimeError("Smoke response did not return resolved citations.")
        if not result.get("timings_ms"):
            raise RuntimeError("Smoke response did not return latency information.")
        if not result.get("usage"):
            raise RuntimeError("Smoke response did not return token usage.")

        report = {
            "packaged_index": str(packaged_path),
            "runtime_working_copy": str(runtime.directory),
            "collection": result["system"]["collection"],
            "qdrant_client_id_before_and_after": qdrant_client_id,
            "answer": result["answer"],
            "citations": result["citations"],
            "latency_ms": result["timings_ms"],
            "token_usage": result["usage"],
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        service.close()
        runtime.close()
        get_runtime_qdrant.cache_clear()


if __name__ == "__main__":
    main()

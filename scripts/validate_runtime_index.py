from pathlib import Path

from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "artifacts" / "qdrant_runtime"

COLLECTION = (
    "sec_filings__chunk-500__overlap-120__encoding-cl100k_base__"
    "embedding-openai-text-embedding-3-small"
)


def main() -> None:
    client = QdrantClient(path=str(INDEX_PATH))

    try:
        details = client.get_collection(COLLECTION)
        count = client.count(
            collection_name=COLLECTION,
            exact=True,
        ).count

        print(f"Collection: {COLLECTION}")
        print(f"Status: {details.status}")
        print(f"Points: {count}")

        if count <= 0:
            raise RuntimeError("Runtime index is empty.")

        points, _ = client.scroll(
            collection_name=COLLECTION,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        required_fields = {
            "chunk_id",
            "chunk_text",
            "ticker",
            "form_type",
            "period_end",
            "source_url",
        }

        payload = points[0].payload or {}
        missing = required_fields - payload.keys()

        if missing:
            raise RuntimeError(
                f"Required payload fields are missing: {sorted(missing)}"
            )

    finally:
        client.close()


if __name__ == "__main__":
    main()
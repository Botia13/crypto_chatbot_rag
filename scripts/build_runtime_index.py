"""Build the isolated runtime Qdrant index from the selected baseline index."""

from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient, models

from RAG_model.ingestion.config import BASELINE_RUN_CONFIG, DB_PATH_NAME, PROJECT_ROOT

RUNTIME_INDEX_PATH = PROJECT_ROOT / "artifacts" / "qdrant_runtime"
DEFAULT_BATCH_SIZE = 256


def _selected_vector_configs(collection_info):
    """Return independent copies of the required dense and BM25 configs."""
    vectors = collection_info.config.params.vectors
    sparse_vectors = collection_info.config.params.sparse_vectors

    if not isinstance(vectors, dict) or "dense" not in vectors:
        raise ValueError("Baseline collection has no named 'dense' vector.")
    if not isinstance(sparse_vectors, dict) or "bm25" not in sparse_vectors:
        raise ValueError("Baseline collection has no named 'bm25' sparse vector.")

    return (
        {"dense": vectors["dense"].model_copy(deep=True)},
        {"bm25": sparse_vectors["bm25"].model_copy(deep=True)},
    )


def build_runtime_index(
    source_path: Path,
    target_path: Path,
    collection_name: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Copy one collection into a new isolated local Qdrant index.

    The target path is reserved atomically and is deliberately retained after
    any failure so a partial build can never be mistaken for a clean rebuild.
    """
    source_path = Path(source_path).resolve()
    target_path = Path(target_path).resolve()

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source Qdrant index does not exist: {source_path}")
    if target_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing runtime index: {target_path}"
        )

    source_client = QdrantClient(path=str(source_path))
    target_client = None
    try:
        if not source_client.collection_exists(collection_name):
            raise ValueError(
                f"Selected baseline collection does not exist: {collection_name}"
            )

        collection_info = source_client.get_collection(collection_name)
        dense_config, sparse_config = _selected_vector_configs(collection_info)
        source_count = source_client.count(
            collection_name=collection_name,
            exact=True,
        ).count

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.mkdir(exist_ok=False)
        target_client = QdrantClient(path=str(target_path))
        target_client.create_collection(
            collection_name=collection_name,
            vectors_config=dense_config,
            sparse_vectors_config=sparse_config,
        )

        copied_count = 0
        offset = None
        while True:
            records, next_offset = source_client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if records:
                points = [
                    models.PointStruct(
                        id=record.id,
                        vector=record.vector,
                        payload=record.payload,
                    )
                    for record in records
                ]
                target_client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
                copied_count += len(points)
                print(f"Copied {copied_count}/{source_count} points")

            if next_offset is None:
                break
            if not records or next_offset == offset:
                raise RuntimeError("Source scroll stopped making progress.")
            offset = next_offset

        final_source_count = source_client.count(
            collection_name=collection_name,
            exact=True,
        ).count
        target_count = target_client.count(
            collection_name=collection_name,
            exact=True,
        ).count

        if final_source_count != source_count:
            raise RuntimeError(
                "Source collection changed during the copy: "
                f"started with {source_count}, ended with {final_source_count}."
            )
        if copied_count != source_count or target_count != source_count:
            raise RuntimeError(
                "Runtime index point-count mismatch: "
                f"source={source_count}, copied={copied_count}, target={target_count}."
            )

        return target_count
    finally:
        if target_client is not None:
            target_client.close()
        source_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an isolated runtime index from the baseline collection."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Points copied per batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection_name = BASELINE_RUN_CONFIG["collection_name"]
    count = build_runtime_index(
        DB_PATH_NAME,
        RUNTIME_INDEX_PATH,
        collection_name,
        batch_size=args.batch_size,
    )
    print(
        f"Built runtime index at {RUNTIME_INDEX_PATH} with "
        f"{count} points in collection '{collection_name}'."
    )


if __name__ == "__main__":
    main()

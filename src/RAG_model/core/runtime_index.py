import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient

from RAG_model.core.settings import get_settings


@dataclass
class RuntimeQdrant:
    client: QdrantClient
    directory: Path

    def close(self) -> None:
        self.client.close()
        shutil.rmtree(
            self.directory,
            ignore_errors=True,
        )


@lru_cache
def get_runtime_qdrant() -> RuntimeQdrant:
    settings = get_settings()
    packaged_path = settings.qdrant_packaged_path

    if not packaged_path.exists():
        raise RuntimeError(
            f"Packaged Qdrant index not found: {packaged_path}"
        )

    runtime_path = Path(
        tempfile.mkdtemp(prefix="crypto-rag-qdrant-")
    )

    shutil.copytree(
        packaged_path,
        runtime_path,
        dirs_exist_ok=True,
    )

    client = QdrantClient(path=str(runtime_path))

    if not client.collection_exists(settings.qdrant_collection):
        client.close()
        shutil.rmtree(runtime_path, ignore_errors=True)
        raise RuntimeError(
            f"Collection not found: {settings.qdrant_collection}"
        )

    return RuntimeQdrant(
        client=client,
        directory=runtime_path,
    )
    

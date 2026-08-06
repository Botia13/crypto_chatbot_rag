# config.py
PIPELINE_VERSION = "v1_1"

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
ENCODING_NAME = "cl100k_base"

# Embedding configuration
EMBEDDINGS_MODEL = "text-embedding-3-small"

# Vector database configuration
COLLECTION_NAME = (
    f"sec_filings_dense_{PIPELINE_VERSION}"
)
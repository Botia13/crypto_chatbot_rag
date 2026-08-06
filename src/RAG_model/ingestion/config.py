# config.py
PIPELINE_VERSION = "v1_2"

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
ENCODING_NAME = "cl100k_base"

# Embedding configuration
EMBEDDING_PROVIDER = "openrouter"
EMBEDDINGS_MODEL = ("openai/text-embedding-3-small")
EMBEDDINGS_BATCH_SIZE = 50

OPENROUTER_BASE_URL = ("https://openrouter.ai/api/v1")

# Vector database configuration
COLLECTION_NAME = (
    "sec_filings_"
    f"{EMBEDDINGS_MODEL.split('/')[0]}_"
    f"{EMBEDDINGS_MODEL.split('/')[1]}_"
    f"{PIPELINE_VERSION}"
)
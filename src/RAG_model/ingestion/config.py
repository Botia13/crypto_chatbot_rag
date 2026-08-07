# config.py
PIPELINE_VERSION = "v1_2"

# The model to use to generate the answers

GENERATION_MODEL = "openai/gpt-4o-mini"


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

# SEC html ingestion config
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
## Input name with the csv that has the SEC html links as CSV
SEC_INPUT_NAME = (PROJECT_ROOT / "data" / "url_links.csv")
## Output Name for the parquet file that has the SEC docs.
SEC_OUTPUT_NAME =  (PROJECT_ROOT / "data" / "parquet_files" / f"sec_corpus_filings_v1.parquet")
## Path for the QDRANT STORAGE
DB_PATH_NAME = (PROJECT_ROOT /"data"/ "qdrant_storage")




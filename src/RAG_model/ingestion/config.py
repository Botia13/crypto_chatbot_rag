# config.py
PIPELINE_VERSION = "v1_3"

# The model to use to generate the answers
BASELINE_RUN_CONFIG = {
    # Identity: saved with every experiment result
    "experiment_name": "baseline_v1_3",
    "pipeline_version": PIPELINE_VERSION,

    # Retrieval / generation
    "retrieval_k": 5,
    "generation_model": "openai/gpt-4o-mini",
    "temperature": 0,
    "prompt_version": "v1",

    # Embeddings / vector collection
    "embedding_provider": "openrouter",
    "embedding_model": "openai/text-embedding-3-small",
    "collection_name": (
        f"sec_filings_openai_text-embedding-3-small_{PIPELINE_VERSION}"
    ),

    # Ingestion configuration: record it for reproducibility
    "chunk_size": 500,
    "chunk_overlap": 75,
    "encoding_name": "cl100k_base",
    "embedding_batch_size": 50,
    
    # RAGAS paramaters
    "ragas_enabled": True,
    "ragas_evaluator_model": "openai/gpt-4o-mini",
    "answer_correct_threshold": 0.8,
}


OPENROUTER_BASE_URL = ("https://openrouter.ai/api/v1")

# SEC html ingestion config
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
## Input name with the csv that has the SEC html links as CSV
SEC_INPUT_NAME = (PROJECT_ROOT / "data" / "url_links.csv")

## Output Name for the parquet file that has the SEC docs.
SEC_OUTPUT_NAME = (
    PROJECT_ROOT
    / "data"
    / "parquet_files"
    / f"sec_corpus_filings_{PIPELINE_VERSION}.parquet"
)
## Path for the QDRANT STORAGE
DB_PATH_NAME = (PROJECT_ROOT /"data"/ "qdrant_storage")




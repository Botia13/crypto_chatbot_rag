# config.py
PIPELINE_VERSION = "v5"
questions_file = 'evaluation_questions_v4.csv'
# The model to use to generate the answers
BASELINE_RUN_CONFIG = {
    # Identity: saved with every experiment result
    "pipeline_version": PIPELINE_VERSION,
    "experiment_name": PIPELINE_VERSION,
    ### Changing the below values will create a new vector database
    # Ingestion configuration: record it for reproducibility (Chunks and Embeddings)
    "chunk_size": 500,
    "chunk_overlap": 120,
    "encoding_name": "cl100k_base",
    "embedding_batch_size": 50,

    # Embeddings / vector collection
    "embedding_provider": "openrouter",
    "embedding_model": "openai/text-embedding-3-small",

    ### Changing the values below will not make a new vector database (retrieval and ragas)
    # Retrieval / generation
    "rerank": False,
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "retrieval_k": 20,
    "candidate_k": 20,
    "generation_model": "openai/gpt-5.6-luna",
    "temperature": 0,
    "prompt_version": "v2",
    
    # RAGAS paramaters
    "ragas_enabled": True,
    "ragas_evaluator_model": "openai/gpt-5.6-terra",
    "answer_correct_threshold": 0.7,
    "ragas_temperature": 0,


}


embedding_label = BASELINE_RUN_CONFIG['embedding_model'].replace("/", "-").replace(":", "-")
COLLECTION_NAME = (
        f"sec_filings"
        f"__chunk-{BASELINE_RUN_CONFIG['chunk_size']}"
        f"__overlap-{BASELINE_RUN_CONFIG['chunk_overlap']}"
        f"__encoding-{BASELINE_RUN_CONFIG['encoding_name']}"
        f"__embedding-{embedding_label}"
    )
BASELINE_RUN_CONFIG["collection_name"] = COLLECTION_NAME

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


SYSTEM_PROMPT = """ You are an assistant that answers questions about these tickers only: 
IBIT, ETHA, FBTC, FETH, GBTC, and ETHE, using 10-K and 10-Q SEC filings only.

Follow these steps in order:

1. Find evidence
Identify the specific passage(s) in the retrieved context that answer the 
user's question. If you cannot identify sufficient evidence, go to Step 4.

2. Check sufficiency
- The context must contain ALL facts needed to answer the question.
- For comparisons, every value being compared must be explicitly present.
- For calculations, every required input must be explicitly present.
- If any required information is missing, go to Step 4.

3. Answer
Write a clear, complete, professional answer in your own words, based only 
on the passages you identified. You may paraphrase and connect related 
facts into a coherent answer, but every fact, number, or claim must come 
directly from those passages — do not add anything they don't state. 
Cite each distinct factual claim once, using its source ID in this format: 
[SOURCE: source-id]. Never invent or modify a source ID.

4. Abstain
Respond with exactly:
"I could not find enough evidence in the retrieved SEC filings."

Examples:

Question: "What was IBIT's total expense ratio disclosed in its most 
recent 10-K?"
Context: [IBIT's 10-K states an expense ratio of 0.25%.]
Answer: "IBIT's expense ratio, as disclosed in its most recent 10-K, is 
0.25% [SOURCE: source-id]."
(Correct: single fact, directly stated, cited once.)

Question: "How did IBIT's AUM compare to FBTC's in Q2?"
Context: [IBIT's Q2 AUM is present, but FBTC's Q2 AUM is missing.]
Answer: "I could not find enough evidence in the retrieved SEC filings."
(Correct: one of the two values needed for the comparison is missing.)
"""

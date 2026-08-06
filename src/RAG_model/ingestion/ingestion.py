import os
import pandas as pd
from pathlib import Path
from sec_filings_ingestion import filings_to_parquet
from chunking import get_chunks_from_corpus
from embedding import create_embeddings
from vector_database import store_embeddings, create_qdrant_collection
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from config import  PIPELINE_VERSION

#Constants
## Workspace of the files 
PROJECT_ROOT = Path(__file__).resolve().parents[3]
## Input name with the csv that has the SEC html links as CSV
SEC_INPUT_NAME = (PROJECT_ROOT / "data" / "url_links.csv")
## Output Name for the parquet file that has the SEC docs.
SEC_OUTPUT_NAME =  (PROJECT_ROOT / "data" / "parquet_files" / f"sec_corpus_filings_{PIPELINE_VERSION}.parquet")
## Path for the QDRANT STORAGE
DB_PATH_NAME = (PROJECT_ROOT /"data"/ "qdrant_storage")


# Ingest the SEC HTML to parquet files 
def ingest_html_sec(input_path = SEC_INPUT_NAME, output_path = SEC_OUTPUT_NAME):
    
    output_path.parent.mkdir(
    parents=True,
    exist_ok=True)
    
    filings = pd.read_csv(input_path)
    
    filings_to_parquet(
        filings,
        str(output_path),
        url_column="Link",
        user_agent = os.environ["SEC_USER_AGENT"],
        filing_date_column=None,
        ticker_column="Ticker",
        form_type_column="Type",
    )
    
def ingestion ():
    
    # Load the env variables 
    environment_path = PROJECT_ROOT / ".env"
    loaded = load_dotenv(
        dotenv_path=environment_path
    )

    if not loaded:
        raise FileNotFoundError(
            f"Could not load environment file: "
            f"{environment_path}"
        )
    # Ingest the SEC HTML files
    if SEC_OUTPUT_NAME.exists():
        print("Parquet file already exists.")
    else:
        ingest_html_sec()
        
    corpus = pd.read_parquet(SEC_OUTPUT_NAME)
    
    # Chunk the corpus
    chunks = get_chunks_from_corpus(corpus)
    #Embedd the chunks
    embeddings = create_embeddings(chunks)
    
    #Save the embeddings in Qdrant
    qdrant_client = QdrantClient(path = str(DB_PATH_NAME))
    create_qdrant_collection(qdrant_client=qdrant_client,vector_size=embeddings['embedding_dimensions'])
    store_embeddings(qdrant_client=qdrant_client, embedded_chunks=embeddings['embedded_chunks'])
    
    # Summary data for the ingestion made
    summary = {
    "filings": len(corpus),
    "chunks": len(chunks),
    "embedding_model": embeddings["embedding_model"],
    "embedding_dimensions": embeddings[
        "embedding_dimensions"
    ],
}

    print(f"Ingestion completed: {summary}")

    return summary
        
    
if __name__ == "__main__":
    ingestion()
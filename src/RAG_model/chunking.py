import pandas as pd 
from pydantic import BaseModel, Field 
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


## CONSTANTS
MAX_TOKENS = 500
OVERLAP_TOKENS = 75
ENCODING_NAME = "cl100k_base"


# Define the encoding for the text splitter
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=ENCODING_NAME,
    chunk_size=MAX_TOKENS,       # Maximum tokens per chunk
    chunk_overlap=OVERLAP_TOKENS,     # Repeated tokens between adjacent chunks
    separators=["\n\n", "\n", ". ", " ", ""],
)

def chunk_document(document):
    """
    Splits a document into chunks based on its sections and tables.
    """
    chunk_records = []

    for section in document["sections"]:
        # 1. Create narrative text chunks
        section_text = section.get("text", "").strip()

        if section_text:
            text_chunks = splitter.split_text(section_text)

            for chunk_index, chunk_text in enumerate(text_chunks):
                chunk_records.append(
                    {
                        "ticker": document["ticker"],
                        "company_name": document["company_name"],
                        "form_type": document["form_type"],
                        "filing_date": document["filing_date"],
                        "period_end": document["period_end"],
                        "source_url": document["source_url"],
                        "accession_number": document["accession_number"],
                        "section_id": section["id"],
                        "chunk_title": section["title"],
                        "chunk_type": "text",
                        "table_id": None,
                        "chunk_index": chunk_index,
                        "chunk_id": (
                            f"{document['accession_number']}"
                            f"*section-{section['id']}"
                            f"*text-{chunk_index:04d}"
                        ),
                        "chunk_text": chunk_text,
                    }
                )

        # 2. Create one separate chunk for each table
        for table in section.get("tables", []):
            table_id = table["table_id"]
            table_text = table["markdown"]

            chunk_records.append(
                {
                    "ticker": document["ticker"],
                    "company_name": document["company_name"],
                    "form_type": document["form_type"],
                    "filing_date": document["filing_date"],
                    "period_end": document["period_end"],
                    "source_url": document["source_url"],
                    "accession_number": document["accession_number"],
                    "section_id": section["id"],
                    "chunk_title": section["title"],
                    "chunk_type": "table",
                    "table_id": table_id,
                    "chunk_index": 0,
                    "chunk_id": (
                        f"{document['accession_number']}"
                        f"*section-{section['id']}"
                        f"*{table_id}"
                    ),
                    "chunk_text": table_text,
                }
            )

    return chunk_records
    
    
def get_chunks_from_corpus(df_corpus):
    """
    Splits all documents in the corpus into chunks and returns a list of chunk records.
    """
    chunks_final = []
    for _, row in df_corpus.iterrows():
        document_chunks = chunk_document(row)
        chunks_final.extend(document_chunks)
    return chunks_final
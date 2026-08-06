from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import  CHUNK_OVERLAP, CHUNK_SIZE, ENCODING_NAME, PIPELINE_VERSION



# Define the encoding for the text splitter
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=ENCODING_NAME,
    chunk_size=CHUNK_SIZE,       # Maximum tokens per chunk
    chunk_overlap=CHUNK_OVERLAP,     # Repeated tokens between adjacent chunks
    separators=["\n\n", "\n", ". ", " ", ""],
)

def chunk_document(document):
    """
    Splits a document into chunks based on its sections and tables.
    Follow these rules to make sure that the chunks are created correctly:
    1. Never cross filling boundaries.
    2. Never cross section boundaries.
    3. Prefer Sentence or line boundaries. 
    4. Include the section heading in every chunk.
    5. Apply overlap between adjacent chunks to preserve context.
    6. Create one separate chunk for each table, even if the table is small.
    7. Refer the table id in the chunk id for table chunks.

    """
    chunk_records = []

    for section in document["sections"]:
        # 1. Create narrative text chunks
        section_text = section.get("text", "").strip()

        if section_text:
            text_chunks = splitter.split_text(section_text)

            for chunk_index, chunk_text in enumerate(text_chunks):
                chunk_text_with_title = (
                    f"Section: {section['title']}\n\n"
                    f"{chunk_text}"
                                    )
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
                        "chunk_version": PIPELINE_VERSION,
                        "table_id": None,
                        "chunk_index": chunk_index,
                        "chunk_id": (
                            f"{document['accession_number']}"
                            f"*section-{section['id']}"
                            f"*text-{chunk_index:04d}"
                        ),
                        "chunk_text": chunk_text_with_title,
                    }
                )

        # 2. Create one separate chunk for each table
        for table in section.get("tables", []):
            table_id = table["table_id"]
            table_text = (
                f"Section: {section['title']}\n"
                f"Table ID: {table_id}\n\n"
                f"{table['markdown']}"
            )

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
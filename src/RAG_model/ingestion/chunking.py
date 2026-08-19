from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import BASELINE_RUN_CONFIG, PIPELINE_VERSION


# Define the encoding for the text splitter
def create_splitter(run_config: dict):
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=run_config["encoding_name"],
        chunk_size=run_config["chunk_size"],
        chunk_overlap=run_config["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def chunk_document(
    document,
    splitter=None,
    pipeline_version: str = PIPELINE_VERSION,
):
    
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
    if splitter is None:
        splitter = create_splitter(BASELINE_RUN_CONFIG)

    chunk_records = []

    for section in document["sections"]:
        section_part = section.get("part")
        if section_part:
            section_key = (
                f"{section_part.lower().replace(' ', '-')}"
                f"-item-{section['id'].lower()}"
            )
        elif section.get("section_type", "item") == "item":
            section_key = f"item-{section['id'].lower()}"
        else:
            section_key = section["id"].lower().replace("_", "-")

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
                        "section_part": section_part,
                        "section_key": section_key,
                        "section_type": section.get("section_type", "item"),
                        "chunk_title": section["title"],
                        "chunk_type": "text",
                        "chunk_version": pipeline_version,
                        "table_id": None,
                        "table_type": None,
                        "table_title": None,
                        "chunk_index": chunk_index,
                        "chunk_id": (
                            f"{document['accession_number']}"
                            f"*{section_key}"
                            f"*text-{chunk_index:04d}"
                        ),
                        "chunk_text": chunk_text_with_title,
                    }
                )

        # 2. Create one separate chunk for each table
        for table in section.get("tables", []):
            table_id = table["table_id"]
            table_title = table.get("table_title") or "Untitled table"
            periods = ", ".join(table.get("periods", [])) or "Not detected"
            units = table.get("units") or "Not stated"
            table_text = (
                f"Section: {section['title']}\n"
                f"Table ID: {table_id}\n\n"
                f"Table title: {table_title}\n"
                f"Table type: {table.get('table_type', 'data_table')}\n"
                f"Periods: {periods}\n"
                f"Units: {units}\n\n"
                f"{table.get('embedding_text') or table['markdown']}"
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
                    "section_part": section_part,
                    "section_key": section_key,
                    "section_type": section.get("section_type", "item"),
                    "chunk_title": section["title"],
                    "chunk_type": "table",
                    "table_type": table.get("table_type", "data_table"),
                    "table_title": table_title,
                    "table_id": table_id,
                    "chunk_index": 0,
                    "chunk_version": pipeline_version,
                    "chunk_id": (
                        f"{document['accession_number']}"
                        f"*{section_key}"
                        f"*{table_id}"
                    ),
                    "chunk_text": table_text,
                }
            )

    return chunk_records
    
    
def get_chunks_from_corpus(df_corpus, run_config: dict):
    splitter = create_splitter(run_config)

    chunks_final = []
    for _, row in df_corpus.iterrows():
        document_chunks = chunk_document(
            row,
            splitter=splitter,
            pipeline_version=run_config["pipeline_version"],
        )
        chunks_final.extend(document_chunks)

    return chunks_final

import os 
from openai import OpenAI
from .config import OPENROUTER_BASE_URL


EMBEDDING_METADATA_FIELDS = (
    ("Ticker", "ticker"),
    ("Company", "company_name"),
    ("Form type", "form_type"),
    ("Filing date", "filing_date"),
    ("Period end", "period_end"),
    ("Accession number", "accession_number"),
    ("Section", "chunk_title"),
    ("Section key", "section_key"),
    ("Chunk type", "chunk_type"),
    ("Table title", "table_title"),
)
EMBEDDING_INPUT_VERSION = "metadata-v1"

# Create a small client factory 
def create_openrouter_client():
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("There is no Open Router Key configured.")
    
    return OpenAI(api_key = api_key,
                  base_url= OPENROUTER_BASE_URL,
                  max_retries=8,
                  timeout=60.0)


def embedding_text(chunk: dict) -> str:
    """Return searchable metadata plus content for a chunk embedding."""
    metadata_lines = [
        f"{label}: {chunk.get(field)}"
        for label, field in EMBEDDING_METADATA_FIELDS
        if chunk.get(field) not in (None, "")
    ]
    chunk_text = str(chunk.get("chunk_text", "")).strip()
    return "Metadata:\n" + "\n".join(metadata_lines) + f"\n\nContent:\n{chunk_text}"
    


def create_embeddings(chunks, model: str, batch_size: int, pipeline_version: str, embedding_client=None):    
   
    """Create embeddings for a list of text chunks using the specified model and batch size."""
    
    embedded_chunks = []
    
    #Check that the chunks are not empty 
    if not chunks:
        raise ValueError("The input 'chunks' list is empty. Please provide valid chunk data.")
    
    if embedding_client is None:
        embedding_client = create_openrouter_client()
    
    # Divide the chunks into batches to avoid hitting API limits 
    for start in range(0, len(chunks), batch_size):
        
        end =  start + batch_size
        batch = chunks[start:end]

        raw_texts = [str(chunk.get("chunk_text", "")).strip() for chunk in batch]
        if any(not text for text in raw_texts):
            raise ValueError("One or more chunks have empty text. Please check the input data.")

        texts = [embedding_text(chunk) for chunk in batch]

        # Create embeddings for the batch    
        response = embedding_client.embeddings.create(
            model = model,
            input = texts
        )   
        
        # Preserve the order of the embedings to match the order of the input chunks
        response_ordered = sorted(response.data, key=lambda x: x.index)
        
        
        # Extract the chunk and the corresponding embedding 
        if len(response_ordered) != len(batch):
            raise RuntimeError("The embeddings API returned a different number of embeddings than the number of input chunks.")
        
        # Extract the chunk and the corresponding embedding withb Strist=True to raise an error if the lengths is diff. 
        for chunk, item in zip(batch, response_ordered, strict=True):
            embedded_chunks.append({
                **chunk,
                "embedding_input_version": EMBEDDING_INPUT_VERSION,
                "embedding" : item.embedding
            })
        
        
    
    vectors_data = {
        "embedding_model": model,
        "chunking_version": pipeline_version,
        "embedding_dimensions": len(embedded_chunks[0]['embedding']),
        "total_chunks": len(chunks),
        "embedded_chunks": embedded_chunks
        
    }
    return vectors_data 
    

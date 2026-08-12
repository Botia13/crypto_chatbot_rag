import os 
from openai import OpenAI
from .config import OPENROUTER_BASE_URL

# Create a small client factory 
def create_openrouter_client():
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("There is no Open Router Key configured.")
    
    return OpenAI(api_key = api_key,
                  base_url= OPENROUTER_BASE_URL,
                  max_retries=8,
                  timeout=60.0)
    


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
        
        texts = [chunk['chunk_text'].strip() for chunk in batch]
        
        # Check for empty text in the batch
        if any(not text for text in texts):
            raise ValueError("One or more chunks have empty text. Please check the input data.")

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
    

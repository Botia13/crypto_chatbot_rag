from openai import OpenAI
from config import EMBEDDINGS_MODEL, PIPELINE_VERSION




def create_embeddings(chunks, model = EMBEDDINGS_MODEL, batch_size = 100, openai_client=None):
    
    """Create embeddings for a list of text chunks using the specified model and batch size."""
    
    embedded_chunks = []
    
    #Check that the chunks are not empty 
    if not chunks:
        raise ValueError("The input 'chunks' list is empty. Please provide valid chunk data.")
    
    if openai_client is None:
        openai_client = OpenAI()
    
    # Divide the chunks into batches to avoid hitting API limits 
    for start in range(0, len(chunks), batch_size):
        
        end =  start + batch_size
        batch = chunks[start:end]
        
        texts = [chunk['chunk_text'].strip() for chunk in batch]
        
        # Check for empty text in the batch
        if any(not text for text in texts):
            raise ValueError("One or more chunks have empty text. Please check the input data.")

        # Create embeddings for the batch    
        response = openai_client.embeddings.create(
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
        "chunking_version" : PIPELINE_VERSION,
        "embedding_dimensions": len(embedded_chunks[0]['embedding']),
        "total_chunks": len(chunks),
        "embedded_chunks": embedded_chunks
        
    }
    return vectors_data 
    

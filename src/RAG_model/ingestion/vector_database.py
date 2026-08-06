from qdrant_client.models import Distance, VectorParams, PointStruct
from uuid import uuid5, NAMESPACE_URL
from config import COLLECTION_NAME



# Create the Server and client for Qdrant
def create_qdrant_collection(qdrant_client,vector_size):

    
    # Check if the collection already exists and if the vector size match the collection dimensions
    if qdrant_client.collection_exists(COLLECTION_NAME):
        collection_info = qdrant_client.get_collection(
            collection_name=COLLECTION_NAME
        )

        existing_vector_size = (
            collection_info.config.params.vectors.size
        )

        if existing_vector_size != vector_size:
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' expects vectors "
                f"with {existing_vector_size} dimensions, but the "
                f"current embedding model returned {vector_size}."
            )

        print(
            f"Collection already exists: {COLLECTION_NAME}"
        )
        return
    
    # Create the collection and define the vector parameters (size and distance metric)
    qdrant_client.create_collection(
        collection_name = COLLECTION_NAME,
        vectors_config = VectorParams(
            size = vector_size,
            distance = Distance.COSINE
        )
    )
    
    print(f"Created collection: {COLLECTION_NAME}")

# Generate a point format for each embedded chunk 
def create_point(embedded_chunk):
        
    # Generate a UUID based on the chunk_id
    # We do this to ensure that the same chunk_id always produces the same UUID, which is important for idempotency.
    chunk_id = embedded_chunk['chunk_id']
    point_id = uuid5(NAMESPACE_URL, chunk_id)
    
    
    # Generate the payload for the point 
    payload = {
        "chunk_id": chunk_id,
        "chunk_text": embedded_chunk["chunk_text"],
        "ticker": embedded_chunk["ticker"],
        "company_name": embedded_chunk["company_name"],
        "form_type": embedded_chunk["form_type"],
        "filing_date": str(embedded_chunk["filing_date"]),
        "period_end": str(embedded_chunk["period_end"]),
        "source_url": embedded_chunk["source_url"],
        "accession_number": embedded_chunk["accession_number"],
        "section_id": embedded_chunk["section_id"],
        "chunk_title": embedded_chunk["chunk_title"],
        "chunk_type": embedded_chunk["chunk_type"],
    }
    if embedded_chunk['table_id'] is not None:
        payload["table_id"] = embedded_chunk["table_id"]
        
    return PointStruct(
        id = point_id,
        vector = embedded_chunk['embedding'],
        payload = payload
    )

# Store the embeddings in Qdrant
def store_embeddings (qdrant_client, embedded_chunks, batch_size = 100):
    
    if not embedded_chunks:
        raise ValueError(
            "There are no embedded chunks to store."
        )
    
    for start in range(0, len(embedded_chunks), batch_size):
        end = start + batch_size
        batch = embedded_chunks[start:end]
        
        # Create a point for each embedded chunk
        points = [create_point(embedded_chunk) for embedded_chunk in batch]
        
        # Add the points to the collection 
        qdrant_client.upsert(
            collection_name  =  COLLECTION_NAME,
            points = points,
            wait = True
        )
        
        print(
            f"Stored {min(start + batch_size, len(embedded_chunks))}"
            f"/{len(embedded_chunks)} chunks"
        )
        
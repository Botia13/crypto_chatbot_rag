from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from uuid import uuid5, NAMESPACE_URL


# Checks if the collection already exists
def index_is_ready(
    qdrant_client,
    collection_name: str,
    embedding_input_version: str | None = None,
) -> bool:
    if not qdrant_client.collection_exists(collection_name):
        return False

    info = qdrant_client.get_collection(collection_name)
    if info.points_count == 0:
        return False
    if embedding_input_version is None:
        return True

    mismatched_points, _ = qdrant_client.scroll(
        collection_name=collection_name,
        limit=1,
        scroll_filter=Filter(
            must_not=[
                FieldCondition(
                    key="embedding_input_version",
                    match=MatchValue(value=embedding_input_version),
                )
            ]
        ),
        with_payload=["embedding_input_version"],
        with_vectors=False,
    )
    return not mismatched_points


# Create the Server and client for Qdrant
def create_qdrant_collection(qdrant_client, collection_name: str, vector_size: int):
    
    # Check if the collection already exists and if the vector size match the collection dimensions
    if qdrant_client.collection_exists(collection_name):
        collection_info = qdrant_client.get_collection(
            collection_name=collection_name
        )
        existing_vector_size = (
            collection_info.config.params.vectors.size
        )
        if existing_vector_size != vector_size:
            raise ValueError(
                f"Collection '{collection_name}' expects vectors "
                f"with {existing_vector_size} dimensions, but the "
                f"current embedding model returned {vector_size}."
            )

        print(
            f"Collection already exists: {collection_name}"
        )
        return
    
    # Create the collection and define the vector parameters (size and distance metric)
    qdrant_client.create_collection(
        collection_name = collection_name,
        vectors_config = VectorParams(
            size = vector_size,
            distance = Distance.COSINE
        )
    )
    
    print(f"Created collection: {collection_name}")

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
        "section_type": embedded_chunk["section_type"],
        "section_part": embedded_chunk["section_part"],
        "section_key": embedded_chunk["section_key"],
        "table_type": embedded_chunk["table_type"],
        "table_title": embedded_chunk["table_title"],
        "chunk_version": embedded_chunk["chunk_version"],
        "section_id": embedded_chunk["section_id"],
        "chunk_title": embedded_chunk["chunk_title"],
        "chunk_type": embedded_chunk["chunk_type"],
        "embedding_input_version": embedded_chunk["embedding_input_version"],
    }
    if embedded_chunk['table_id'] is not None:
        payload["table_id"] = embedded_chunk["table_id"]
        
    return PointStruct(
        id = point_id,
        vector = embedded_chunk['embedding'],
        payload = payload
    )

# Store the embeddings in Qdrant
def store_embeddings(qdrant_client,collection_name: str,embedded_chunks,batch_size: int = 100):
    
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
            collection_name  =  collection_name,
            points = points,
            wait = True
        )
        
        print(
            f"Stored {min(start + batch_size, len(embedded_chunks))}"
            f"/{len(embedded_chunks)} chunks"
        )

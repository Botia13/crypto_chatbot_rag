from RAG_model.ingestion.config import EMBEDDINGS_MODEL, COLLECTION_NAME, DB_PATH_NAME, GENERATION_MODEL, RETRIEVAL_K
from RAG_model.ingestion.embedding import create_openrouter_client
from qdrant_client import QdrantClient
from tenacity import retry, wait_exponential, stop_after_attempt
from time import perf_counter
from pathlib import Path
from dotenv import load_dotenv

wait = wait_exponential(multiplier=1, min=10, max=240)

qdrant_client = QdrantClient(path = str(DB_PATH_NAME))
print("Qdrant client Open.")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")
client = create_openrouter_client()

def fetch_context(question, retrieval_k = RETRIEVAL_K):
    """Embed a question and retrieve its top-k most similar chunks."""
    
    # Embedd the question with the same embedding model
    query = client.embeddings.create(model = EMBEDDINGS_MODEL, input=[question]).data[0].embedding
    
    # Return results from the database based on the question
    results = qdrant_client.query_points(
        collection_name= COLLECTION_NAME,
        query = query,
        limit= retrieval_k,
        with_payload= True)
        
    # Convert the payload to the right format 
    
    chunks = []
    
    for point in results.points:
        payload = point.payload or {}

        chunks.append(
            {
                "chunk_id": payload["chunk_id"],
                "chunk_text": payload["chunk_text"],
                "ticker": payload["ticker"],
                "filing_date": payload["filing_date"],
                "section_title": payload["chunk_title"],
                "source_url": payload["source_url"],
                "score": point.score,
            }
        )

    return chunks

def make_rag_messages(question, history, chunks):
    """Build chat messages for the RAG answer step: system (with context) + history + user question."""
    context = "\n\n".join(
        f"-Extract from:\n \
           [SOURCE:{chunk['chunk_id']:}]\n \
            - TICKER: {chunk['ticker']}\n \
            - Section:{chunk['section_title']}\n \
            - Filing Date:{chunk['filing_date']}\n \
            - SEC URL{chunk['source_url']}] \n \
            - Content: \n {chunk['chunk_text']}  " for chunk in chunks)
    
    system_prompt = f"""
    You are a knowledgeable, friendly assistant that helps with question regard some specific companies extracting only the information available from the SEC 10-K and 10-Q forms those companies submmit.
    You are chatting with a user about Answer questions about these SEC filing entities: IBIT, ETHA, FBTC, FETH, GBTC, ETHE. If the question is outside this corpus, explain that the corpus does
    not contain evidence to answer it.
    Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
    Answer using only the supplied context.

    If the context does not contain enough information, say:
    "I could not find enough evidence in the retrieved SEC filings."

    Cite the supplied source IDs for every factual claim using:
    [SOURCE: chunk_id]

    Never invent a source ID or cite a source that was not supplied.
    
    Do not infer, calculate, or compare values unless the retrieved context
    contains all evidence needed. Otherwise say that there is not enough evidence.
    
    For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
    {context}

    With this context, please answer the user's question. Be accurate, relevant and complete.
    """
    
    return (
        [{"role": "system",
          "content": system_prompt}]
        + history
        + [{"role": "user",
            "content": question}]
        )
   
   
def format_answer(answer):
    
    question, answer_text, chunks, latency_ms = answer
    
    scores = [ {"chunk_id": chunk["chunk_id"],
                "scores": chunk["score"]} for chunk in chunks]
    
    return {
        "User Question": question,
        "Answer": answer_text,
        "Similarity Scores": scores,
        "Retrieved Chunk texts": chunks,
        "Citations": answer_text.split("SOURCE")[1][:-2],
        "Latency": latency_ms
    }
    
       
   
    
@retry(wait=wait,
       stop=stop_after_attempt(4))
def answer(question: str, history:list[dict] | None = None):
    
    if history is None:
        history = []
    
    total_start = perf_counter()
    
    
    retrieval_start = perf_counter()
    chunks = fetch_context(question)
    retrieval_end = perf_counter()
    
    prompt_start = perf_counter()
    messages = make_rag_messages(question, history, chunks)
    prompt_end = perf_counter()
    
    generation_start = perf_counter()
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
    )
    generation_end = perf_counter()
    
    answer_text = response.choices[0].message.content
    
    latency_ms = {
        "retrieval": (retrieval_end - retrieval_start) * 1000,
        "prompt_building": (prompt_end - prompt_start) * 1000,
        "generation": (generation_end - generation_start) * 1000,
        "total": (generation_end - total_start) * 1000,
    }
    
    answer = question, answer_text, chunks, latency_ms
    
    final_answer = format_answer(answer)
    
    return final_answer
    
client.close()
qdrant_client.close()
print("Qdrant client closed.")

print("Done")

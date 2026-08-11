import re
from RAG_model.ingestion.config import EMBEDDINGS_MODEL, COLLECTION_NAME, DB_PATH_NAME, GENERATION_MODEL, RETRIEVAL_K
from RAG_model.ingestion.embedding import create_openrouter_client
from qdrant_client import QdrantClient
from time import perf_counter
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


qdrant_client = QdrantClient(path=str(DB_PATH_NAME))
print("Qdrant client Open.")

client = create_openrouter_client()

def fetch_context(question: str, retrieval_k: int = RETRIEVAL_K) -> list[dict]:
    query = client.embeddings.create(model=EMBEDDINGS_MODEL, input=[question]).data[0].embedding
    results = qdrant_client.query_points(collection_name=COLLECTION_NAME, query=query, limit=retrieval_k, with_payload=True)
    chunks = []
    for point in results.points:
        payload = point.payload or {}
        chunks.append({
            "chunk_id": payload.get("chunk_id"),
            "chunk_text": payload.get("chunk_text"),
            "ticker": payload.get("ticker"),
            "filing_date": payload.get("filing_date"),
            "section_title": payload.get("chunk_title"),
            "source_url": payload.get("source_url"),
            "score": point.score,
            "section_type": payload.get("section_type", "item"),
            "section_part": payload.get("section_part"),
            "section_key": payload.get("section_key"),
            "table_type": payload.get("table_type"),
            "table_title": payload.get("table_title")
        })
    return chunks


def make_rag_messages(question, history, chunks):
    """Build chat messages for the RAG answer step: system (with context) + history + user question."""
    context = "\n\n".join(
        f"""[SOURCE: {chunk['chunk_id']}]
    Ticker: {chunk['ticker']}
    Section: {chunk['section_title']}
    Section key: {chunk.get('section_key', 'N/A')}
    Table: {chunk.get('table_title') or 'N/A'}
    SEC URL: {chunk['source_url']}

    Content:
    {chunk['chunk_text']}"""
        for chunk in chunks)
    
    system_prompt = f"""You answer only with information from supplied SEC 10-K and 10-Q filing extracts.
        The corpus covers IBIT, ETHA, FBTC, FETH, GBTC, and ETHE. If the corpus does not contain enough information, say: "I could not find enough evidence in the retrieved SEC filings."
        Answer using only the supplied context. Cite every factual claim using supplied IDs in this format: [SOURCE: source-id]. Never invent a source ID. Do not infer, calculate, or compare values unless the retrieved context contains all evidence needed.

        Context:
        {context}"""
    
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
        "Citations": re.findall(r"\[SOURCE:\s*([^\]]+)\]", answer_text),
        "Latency": latency_ms
    }
    
       
   

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
    
    result = question, answer_text, chunks, latency_ms
    
    final_answer = format_answer(result)
    
    return final_answer

def main() -> None:
    question = input("Question: ").strip()
    if not question:
        raise SystemExit("A question is required.")
    result = answer(question)
    print("\nAnswer:\n", result["Answer"])

    
if __name__ == "__main__":
    try:
        main()
    finally:
        client.close()
        qdrant_client.close()




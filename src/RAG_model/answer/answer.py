import re
from RAG_model.ingestion.embedding import create_openrouter_client
from qdrant_client import QdrantClient
from time import perf_counter
from pathlib import Path
from dotenv import load_dotenv
from RAG_model.ingestion.config import DB_PATH_NAME, BASELINE_RUN_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


qdrant_client = QdrantClient(path=str(DB_PATH_NAME))
print("Qdrant client Open.")

client = create_openrouter_client()

def fetch_context(question: str,retrieval_k: int,embedding_model: str,collection_name: str):
    
    embedding_response = client.embeddings.create(model=embedding_model,input=[question])
    query = embedding_response.data[0].embedding

    results = qdrant_client.query_points(collection_name=collection_name,query=query,limit=retrieval_k,with_payload=True)
    
    chunks = []
    for point in results.points:
        payload = point.payload or {}
        chunks.append({
            "chunk_id": payload.get("chunk_id"),
            "accession_number": payload.get("accession_number"),
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
    
    embedding_usage = {"input_tokens": embedding_response.usage.prompt_tokens,
                         "total_tokens": embedding_response.usage.total_tokens}
    
    return chunks,embedding_usage


def make_rag_messages(question, history, chunks, prompt_version: str):
    """Build chat messages for the RAG answer step: system (with context) + history + user question."""
    
    if prompt_version != "v1":
     raise ValueError(f"Unknown prompt version: {prompt_version}")
    
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
    
    question, answer_text, chunks, latency_ms, token_usage = answer
    
    scores = [ {"chunk_id": chunk["chunk_id"],
                "scores": chunk["score"]} for chunk in chunks]
    
    return {
        "User Question": question,
        "Answer": answer_text,
        "Similarity Scores": scores,
        "Retrieved Chunk texts": chunks,
        "Citations": re.findall(r"\[SOURCE:\s*([^\]]+)\]", answer_text),
        "Latency": latency_ms,
        "Token_usage": token_usage
    }
    
       
   

def answer(question: str, run_config:dict, history:list[dict] | None = None):
    
    if history is None:
        history = []
    
    total_start = perf_counter()
    
    
    retrieval_start = perf_counter()
    chunks, embedding_usage = fetch_context(question=question, retrieval_k=run_config["retrieval_k"],embedding_model=run_config["embedding_model"],collection_name=run_config["collection_name"])
    retrieval_end = perf_counter()
    
    prompt_start = perf_counter()
    messages = make_rag_messages(question,history,chunks,prompt_version=run_config["prompt_version"])
    prompt_end = perf_counter()
    
    generation_start = perf_counter()
    response = client.chat.completions.create(
        model = run_config["generation_model"],
        messages=messages,
    )
    generation_end = perf_counter()
    generation_usage = {"input_tokens":response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "total_tokens":response.usage.total_tokens}
    
    answer_text = response.choices[0].message.content
    
    latency_ms = {
        "retrieval": (retrieval_end - retrieval_start) * 1000,
        "prompt_building": (prompt_end - prompt_start) * 1000,
        "generation": (generation_end - generation_start) * 1000,
        "total": (generation_end - total_start) * 1000
    }
    
    token_usage = {
    "embedding_input_tokens": embedding_usage["input_tokens"],
    "embedding_total_tokens": embedding_usage["total_tokens"],
    "generation_input_tokens": generation_usage["input_tokens"],
    "generation_output_tokens": generation_usage["output_tokens"],
    "generation_total_tokens": generation_usage["total_tokens"],
    "rag_total_tokens": (
        embedding_usage["total_tokens"]
        + generation_usage["total_tokens"]
    )
}
    
    result = question, answer_text, chunks, latency_ms, token_usage
    
    final_answer = format_answer(result)
    
    return final_answer

def main() -> None:
    question = input("Question: ").strip()
    if not question:
        raise SystemExit("A question is required.")
    result = answer(question, BASELINE_RUN_CONFIG)
    print("\nAnswer:\n", result["Answer"])

    
if __name__ == "__main__":
    try:
        main()
    finally:
        client.close()
        qdrant_client.close()




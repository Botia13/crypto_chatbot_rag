import re
from RAG_model.ingestion.embedding import create_openrouter_client
from qdrant_client import QdrantClient
from time import perf_counter
from pathlib import Path
from dotenv import load_dotenv
from RAG_model.ingestion.config import DB_PATH_NAME, BASELINE_RUN_CONFIG, SYSTEM_PROMPT
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from sentence_transformers import CrossEncoder
from qdrant_client import models          
from fastembed import SparseTextEmbedding  


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")



client = create_openrouter_client()
_bm25 = SparseTextEmbedding(model_name="Qdrant/bm25") 


class EmptyGenerationResponseError(RuntimeError):
    """Raised when a provider returns no usable chat-completion text."""


@retry(
    retry=retry_if_exception_type(EmptyGenerationResponseError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)

def generate_response(messages, run_config: dict):
    """Generate an answer and retry a transient empty provider response."""
    response = client.chat.completions.create(
        model=run_config["generation_model"],
        messages=messages,
        temperature=run_config["temperature"],
        max_tokens=1000
    )

    choices = response.choices or []
    if not choices:
        raise EmptyGenerationResponseError(
            "Generation model returned no choices."
        )

    choice = choices[0]
    if choice.message is None or not choice.message.content:
        raise EmptyGenerationResponseError(
            "Generation model returned no text. "
            f"finish_reason={choice.finish_reason}"
        )

    return response

def fetch_context(question: str,candidate_k: int,embedding_model: str,collection_name: str, qdrant_client):
    
    embedding_response = client.embeddings.create(model=embedding_model,input=[question])
    dense_q = embedding_response.data[0].embedding
    
    sparse_q = next(_bm25.embed(question))

    results = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=dense_q, using="dense", limit=candidate_k),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_q.indices.tolist(),
                    values=sparse_q.values.tolist(),
                ),
                using="bm25",
                limit=candidate_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=candidate_k,
        with_payload=True,
    )
    
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


reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1500)
def rerank(question,chunks,top_k):
    
    scorable = [chunk for chunk in chunks if chunk.get("chunk_text")]
    if not scorable:
        return chunks[:top_k]

    pairs = [(question, chunk["chunk_text"]) for chunk in scorable]
    scores  = reranker.predict(pairs)
    
    for chunk, score in zip(scorable,scores):
        chunk['rerank_scores'] = float(score)
        
    ranked = sorted(scorable, key=lambda chunk: chunk['rerank_scores'], reverse=True)
    
    return ranked[:top_k]

def retrieve(question: str, run_config: dict, qdrant_client):
    """Retrieve chunks without building a prompt or calling a generation model."""
    retrieval_start = perf_counter()
    
    chunks, embedding_usage = fetch_context(
        question=question,
        candidate_k=run_config["candidate_k"],
        embedding_model=run_config["embedding_model"],
        collection_name=run_config["collection_name"],
        qdrant_client=qdrant_client,
    )
    rank = run_config['rerank']
    if rank:
        reranked_chunks = rerank(question,chunks,run_config['retrieval_k'])
    else:
        reranked_chunks = chunks[: run_config['retrieval_k']]
    
    retrieval_latency_ms = (perf_counter() - retrieval_start) * 1000

    
    return {
        "User Question": question,
        "Retrieved Chunk texts": reranked_chunks,
        "Similarity Scores": [
            {
                "chunk_id": chunk["chunk_id"],
                "dense_score": chunk["score"],
                "rerank_score": chunk.get("rerank_scores"),
            }
            for chunk in reranked_chunks
        ],
        "Latency": {
            "retrieval": retrieval_latency_ms,
            "prompt_building": 0.0,
            "generation": 0.0,
            "total": retrieval_latency_ms,
        },
        "Token_usage": {
            "embedding_input_tokens": embedding_usage.get("input_tokens"),
            "embedding_total_tokens": embedding_usage.get("total_tokens"),
            "generation_input_tokens": 0,
            "generation_output_tokens": 0,
            "generation_total_tokens": 0,
            "rag_total_tokens": embedding_usage.get("total_tokens"),
        },
    }



def make_rag_messages(question, history, chunks, system_prompt: str):
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
    
    system_prompt = f"""

    {system_prompt}

    Context:
    {context}

    """

    
    return (
        [{"role": "system",
          "content": system_prompt}]
        + history
        + [{"role": "user",
            "content": question}]
        )
   
   
def format_answer(question,chunks,generation_result,embedding_usage, retrieval_latency_ms):
    
    similarity_scores = [
        {
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
        }
        for chunk in chunks
    ]

    prompt_latency_ms = generation_result["prompt_latency_ms"]
    generation_latency_ms = generation_result["generation_latency_ms"]

    latency_ms = {
        "retrieval": retrieval_latency_ms,
        "prompt_building": prompt_latency_ms,
        "generation": generation_latency_ms,
        "total": (
            retrieval_latency_ms
            + prompt_latency_ms
            + generation_latency_ms
        ),
    }

    embedding_tokens = embedding_usage.get("total_tokens")
    generation_tokens = generation_result.get(
        "generation_total_tokens"
    )

    if embedding_tokens is not None and generation_tokens is not None:
        rag_total_tokens = embedding_tokens + generation_tokens
    else:
        rag_total_tokens = None

    token_usage = {
        "embedding_input_tokens": embedding_usage.get("input_tokens"),
        "embedding_total_tokens": embedding_tokens,
        "generation_input_tokens": generation_result.get(
            "generation_input_tokens"
        ),
        "generation_output_tokens": generation_result.get(
            "generation_output_tokens"
        ),
        "generation_total_tokens": generation_tokens,
        "rag_total_tokens": rag_total_tokens,
    }

    return {
        "User Question": question,
        "Answer": generation_result["answer"],
        "Similarity Scores": similarity_scores,
        "Retrieved Chunk texts": chunks,
        "Citations": generation_result["citations"],
        "Latency": latency_ms,
        "Token_usage": token_usage,
    }
    

def answer_from_chunks(
    question: str,
    chunks,
    run_config: dict,
    system_prompt: str = SYSTEM_PROMPT,
    history: list[dict] | None = None,
):
    
    if history is None:
        history = []

    prompt_start = perf_counter()
    messages = make_rag_messages(question,history,chunks,system_prompt=system_prompt)
    prompt_end = perf_counter()
    
    generation_start = perf_counter()
    response = generate_response(messages, run_config)
    generation_end = perf_counter()
    
    usage = response.usage
    answer_text = response.choices[0].message.content
    
    return {
        "answer": answer_text,
        "citations": re.findall(
            r"\[SOURCE:\s*([^\]]+)\]",
            answer_text or "",
        ),
        "generation_latency_ms": (
            generation_end - generation_start
        ) * 1000,
        "prompt_latency_ms": (
            prompt_end - prompt_start
        ) * 1000,
        "generation_input_tokens": (
            usage.prompt_tokens if usage else None
        ),
        "generation_output_tokens": (
            usage.completion_tokens if usage else None
        ),
        "generation_total_tokens": (
            usage.total_tokens if usage else None
        )
    }
    

def answer(
    question,
    run_config,
    system_prompt: str = SYSTEM_PROMPT,
    qdrant_client=None,
    history=None,
):
    if qdrant_client is None:
        raise ValueError("A Qdrant client is required to answer a question.")

    retrieval_result = retrieve(question, run_config, qdrant_client)
    chunks = retrieval_result["Retrieved Chunk texts"]

    generation_result = answer_from_chunks(
        question=question,
        chunks=chunks,
        run_config=run_config,
        history=history,
        system_prompt=system_prompt
    )

    return format_answer(
        question=question,
        chunks=chunks,
        generation_result=generation_result,
        embedding_usage={
            "input_tokens": retrieval_result["Token_usage"]["embedding_input_tokens"],
            "total_tokens": retrieval_result["Token_usage"]["embedding_total_tokens"],
        },
        retrieval_latency_ms=retrieval_result["Latency"]["retrieval"],
    )


def main(system_prompt: str = SYSTEM_PROMPT) -> None:
    question = input("Question: ").strip()
    if not question:
        raise SystemExit("A question is required.")
    qdrant_client = QdrantClient(path=str(DB_PATH_NAME))
    try:
        result = answer(question, BASELINE_RUN_CONFIG,system_prompt,qdrant_client)
        print("\nAnswer:\n", result["Answer"])
    finally:
        qdrant_client.close()
    
if __name__ == "__main__":
    try:
        main(SYSTEM_PROMPT)
    finally:
        client.close()


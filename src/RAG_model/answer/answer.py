import re
from math import ceil
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from RAG_model.ingestion.config import BASELINE_RUN_CONFIG, DB_PATH_NAME, SYSTEM_PROMPT
from RAG_model.ingestion.embedding import create_openrouter_client
from RAG_model.retrieval.full_benchmark import (
    question_constraints,
    resolve_searches,
)
from RAG_model.retrieval.retrieval_benchmark import (
    assert_scope,
    deduplicate,
    make_filter,
    round_robin,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")



_bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")

SOURCE_MARKER_PATTERN = re.compile(
    r"\[SOURCE:\s*([^\]\}\r\n]+?)\s*[\]\}]",
    re.IGNORECASE,
)

QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite conversational follow-up questions
into standalone search questions for SEC filings.

Use the conversation only to resolve omitted tickers, entities, metrics,
reporting periods, and comparison targets. Preserve the current user's intent.
Do not answer the question. Return only the standalone question, with no label,
explanation, quotation marks, or citation markers.
"""


class EmptyGenerationResponseError(RuntimeError):
    """Raised when a provider returns no usable chat-completion text."""


@retry(
    retry=retry_if_exception_type(EmptyGenerationResponseError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)

def generate_response(
    messages,
    run_config: dict,
    provider_client,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
):
    """Generate an answer and retry a transient empty provider response."""
    response = provider_client.chat.completions.create(
        model=run_config["generation_model"],
        messages=messages,
        temperature=(
            run_config["temperature"]
            if temperature is None
            else temperature
        ),
        max_tokens=(
            run_config.get("max_generation_tokens", 1000)
            if max_tokens is None
            else max_tokens
        ),
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


def rewrite_followup_question(
    question: str,
    history: list[dict] | None,
    run_config: dict,
    provider_client,
) -> dict:
    """Resolve conversational references before retrieval."""
    if not history:
        return {
            "question": question,
            "latency_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    messages = (
        [{"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": question}]
    )

    rewrite_start = perf_counter()
    response = generate_response(
        messages,
        run_config,
        provider_client,
        max_tokens=run_config.get("max_query_rewrite_tokens", 128),
        temperature=0,
    )
    rewrite_latency_ms = (perf_counter() - rewrite_start) * 1000

    rewritten_question = response.choices[0].message.content.strip()
    if (
        len(rewritten_question) >= 2
        and rewritten_question[0] == rewritten_question[-1]
        and rewritten_question[0] in {'"', "'"}
    ):
        rewritten_question = rewritten_question[1:-1].strip()

    if not rewritten_question:
        raise EmptyGenerationResponseError(
            "Question rewrite returned no usable text."
        )

    usage = response.usage
    return {
        "question": rewritten_question,
        "latency_ms": rewrite_latency_ms,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
    }


def extract_citations_and_clean_answer(
    answer_text: str | None,
) -> tuple[str, list[str]]:
    """Remove machine-readable source markers after extracting their IDs."""
    answer_text = answer_text or ""
    citations = []
    for match in SOURCE_MARKER_PATTERN.finditer(answer_text):
        citation_id = match.group(1).strip()
        if citation_id and citation_id not in citations:
            citations.append(citation_id)
    clean_answer = SOURCE_MARKER_PATTERN.sub("", answer_text)
    clean_answer = re.sub(r"[ \t]+(?=[,.;:!?])", "", clean_answer)
    clean_answer = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", clean_answer)
    clean_answer = re.sub(r"(?m)[ \t]+$", "", clean_answer)
    clean_answer = re.sub(r"\n{3,}", "\n\n", clean_answer).strip()
    return clean_answer, citations

def load_filing_catalog(qdrant_client, collection_name: str) -> list[dict]:
    """Return one metadata record per filing stored in the collection.

    Users describe filings with phrases such as "2024 annual filing", while
    retrieval filters need exact values such as ``10-K`` and ``2024-12-31``.
    The catalog provides that mapping without loading vectors or chunk text.
    """
    fields = ["ticker", "form_type", "period_end", "accession_number"]
    filings = {}
    offset = None

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=fields,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            if not all(payload.get(field) for field in fields):
                raise ValueError("Indexed filing is missing required scope metadata.")

            filing = {field: payload[field] for field in fields}
            accession = filing["accession_number"]
            previous = filings.setdefault(accession, filing)
            if previous != filing:
                raise ValueError(
                    f"Conflicting scope metadata for accession {accession}."
                )

        if offset is None:
            break

    return list(filings.values())


def fetch_context(question: str,candidate_k: int,embedding_model: str,collection_name: str, qdrant_client,provider_client):
    """Retrieve a balanced, metadata-filtered hybrid candidate pool.

    The question is resolved to concrete filing scopes before one dense and
    BM25 query is reused across ticker-specific searches. Results retain the
    raw hybrid order, are interleaved across tickers, deduplicated, and capped
    at ``candidate_k`` without unrelated backfill.
    """
    constraints = question_constraints(question)
    catalog = (
        []
        if constraints["mode"] == "explicit_scope"
        else load_filing_catalog(qdrant_client, collection_name)
    )
    search_entries = resolve_searches(constraints, catalog)

    if not any(not entry["empty"] for entry in search_entries):
        return [], {"input_tokens": 0, "total_tokens": 0}

    embedding_response = provider_client.embeddings.create(model=embedding_model,input=[question])
    dense_q = embedding_response.data[0].embedding
    sparse_q = next(_bm25.embed(question))
    per_search_limit = ceil(candidate_k / len(search_entries))

    groups = []
    for entry in search_entries:
        if entry["empty"]:
            groups.append([])
            continue

        # Filtering both prefetches prevents irrelevant points from consuming
        # their limits; the final filter also protects the fused RRF output.
        query_filter = make_filter(entry["spec"])
        results = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_q,
                    using="dense",
                    limit=per_search_limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_q.indices.tolist(),
                        values=sparse_q.values.tolist(),
                    ),
                    using="bm25",
                    limit=per_search_limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=per_search_limit,
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
                "form_type": payload.get("form_type"),
                "filing_date": payload.get("filing_date"),
                "period_end": payload.get("period_end"),
                "section_title": payload.get("section_title") or payload.get("chunk_title"),
                "source_url": payload.get("source_url"),
                "score": float(point.score),
                "section_type": payload.get("section_type", "item"),
                "section_part": payload.get("section_part"),
                "section_key": payload.get("section_key"),
                "table_type": payload.get("table_type"),
                "table_title": payload.get("table_title")
            })

        assert_scope(chunks, entry["spec"])
        groups.append(chunks)

    chunks = deduplicate(round_robin(groups), candidate_k)
    embedding_usage = {"input_tokens": embedding_response.usage.prompt_tokens,
                         "total_tokens": embedding_response.usage.total_tokens}

    return chunks,embedding_usage


def rerank(question,chunks,top_k,run_config):
    """Optionally reorder retrieved chunks with the configured cross-encoder."""
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(run_config["reranker_model"], max_length=1500)

    scorable = [chunk for chunk in chunks if chunk.get("chunk_text")]
    if not scorable:
        return chunks[:top_k]

    pairs = [(question, chunk["chunk_text"]) for chunk in scorable]
    scores  = reranker.predict(pairs)
    
    for chunk, score in zip(scorable,scores):
        chunk['rerank_scores'] = float(score)
        
    ranked = sorted(scorable, key=lambda chunk: chunk['rerank_scores'], reverse=True)
    
    return ranked[:top_k]

def retrieve(question: str, run_config: dict, qdrant_client, provider_client):
    """Retrieve and optionally rerank context without generating an answer."""
    retrieval_start = perf_counter()
    
    chunks, embedding_usage = fetch_context(
        question=question,
        candidate_k=run_config["candidate_k"],
        embedding_model=run_config["embedding_model"],
        collection_name=run_config["collection_name"],
        qdrant_client=qdrant_client,
        provider_client=provider_client
    )
    if run_config['rerank']:
        selected_chunks = rerank(question,chunks,run_config['retrieval_k'],run_config)
    else:
        selected_chunks = chunks[: run_config['retrieval_k']]
    
    retrieval_latency_ms = (perf_counter() - retrieval_start) * 1000

    
    return {
        "User Question": question,
        "Retrieved Chunk texts": selected_chunks,
        "Similarity Scores": [
            {
                "chunk_id": chunk["chunk_id"],
                "hybrid_score": chunk["score"],
                "rerank_score": chunk.get("rerank_scores"),
            }
            for chunk in selected_chunks
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



from RAG_model.answer.prompt import make_rag_messages
   
   
def format_answer(
    question,
    chunks,
    generation_result,
    embedding_usage,
    retrieval_latency_ms,
    rewrite_result,
):
    
    similarity_scores = [
        {
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
        }
        for chunk in chunks
    ]

    prompt_latency_ms = generation_result["prompt_latency_ms"]
    generation_latency_ms = generation_result["generation_latency_ms"]
    rewrite_latency_ms = rewrite_result["latency_ms"]

    latency_ms = {
        "query_rewrite": rewrite_latency_ms,
        "retrieval": retrieval_latency_ms,
        "prompt_building": prompt_latency_ms,
        "generation": generation_latency_ms,
        "total": (
            rewrite_latency_ms
            + retrieval_latency_ms
            + prompt_latency_ms
            + generation_latency_ms
        ),
    }

    embedding_tokens = embedding_usage.get("total_tokens")
    rewrite_tokens = rewrite_result.get("total_tokens")
    generation_tokens = generation_result.get(
        "generation_total_tokens"
    )

    if all(
        value is not None
        for value in (rewrite_tokens, embedding_tokens, generation_tokens)
    ):
        rag_total_tokens = (
            rewrite_tokens + embedding_tokens + generation_tokens
        )
    else:
        rag_total_tokens = None

    token_usage = {
        "rewrite_input_tokens": rewrite_result.get("input_tokens"),
        "rewrite_output_tokens": rewrite_result.get("output_tokens"),
        "rewrite_total_tokens": rewrite_tokens,
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
    provider_client,
    system_prompt: str = SYSTEM_PROMPT,
    history: list[dict] | None = None,
):
    
    if history is None:
        history = []

    prompt_start = perf_counter()
    messages = make_rag_messages(question,history,chunks,system_prompt=system_prompt)
    prompt_end = perf_counter()
    
    generation_start = perf_counter()
    response = generate_response(messages, run_config, provider_client)
    generation_end = perf_counter()
    
    usage = response.usage
    answer_text = response.choices[0].message.content
    clean_answer, citations = extract_citations_and_clean_answer(answer_text)
    
    return {
        "answer": clean_answer,
        "citations": citations,
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
    provider_client,
    system_prompt: str = SYSTEM_PROMPT,
    qdrant_client=None,
    history=None,
):
    if qdrant_client is None:
        raise ValueError("A Qdrant client is required to answer a question.")

    history = history or []
    rewrite_result = rewrite_followup_question(
        question=question,
        history=history,
        run_config=run_config,
        provider_client=provider_client,
    )

    retrieval_result = retrieve(
        rewrite_result["question"],
        run_config,
        qdrant_client,
        provider_client,
    )
    chunks = retrieval_result["Retrieved Chunk texts"]

    generation_result = answer_from_chunks(
        question=question,
        chunks=chunks,
        run_config=run_config,
        provider_client=provider_client,
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
        rewrite_result=rewrite_result,
    )


def main(system_prompt: str = SYSTEM_PROMPT) -> None:
    question = input("Question: ").strip()
    if not question:
        raise SystemExit("A question is required.")
    qdrant_client = QdrantClient(path=str(DB_PATH_NAME))
    provider_client = create_openrouter_client()
    try:
        result = answer(
            question,
            BASELINE_RUN_CONFIG,
            provider_client=provider_client,
            system_prompt=system_prompt,
            qdrant_client=qdrant_client,
        )
        print("\nAnswer:\n", result["Answer"])
    finally:
        qdrant_client.close()
        provider_client.close()
    
if __name__ == "__main__":
    main(SYSTEM_PROMPT)

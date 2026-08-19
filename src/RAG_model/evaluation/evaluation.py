import os
from collections.abc import Mapping
import pandas as pd
from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness,FactualCorrectness,LLMContextPrecisionWithReference
from RAG_model.answer.answer import answer, retrieve
from RAG_model.ingestion.config import BASELINE_RUN_CONFIG, OPENROUTER_BASE_URL, DB_PATH_NAME
from RAG_model.ingestion.ingestion import ingestion
from pathlib import Path
from tqdm.auto import tqdm
from qdrant_client import QdrantClient


_PREPARED_COLLECTIONS: set[tuple] = set()


# Read the questions file 
def load_evaluation_questions():
    root = Path.cwd()
    while not (root / 'data' / 'rag_evaluation' / 'evaluation_questions_v2.csv').exists() and root.parent != root:
        root = root.parent

    csv_path = root / 'data' / 'rag_evaluation' / 'evaluation_questions_v2.csv'

    questions = pd.read_csv(csv_path)
    
    return questions

# Create evaluator LLM
def create_ragas_metrics(run_config):
    ragas_client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL, timeout=60.0 , max_retries = 5 )
    ragas_llm = llm_factory(run_config['ragas_evaluator_model'],provider="openai", client = ragas_client, max_tokens = 2048, temperature=run_config["ragas_temperature"])

    # Create metric Objects
    metrics = {
        "faithfulness_metric": Faithfulness(llm=ragas_llm),
        "factual_correctness_metric": FactualCorrectness(llm=ragas_llm,mode="f1"),
        "context_precision_metric":LLMContextPrecisionWithReference(llm=ragas_llm)}
    
    return ragas_client, metrics
        
    
# Helper Functions 

def reciprocal_rank (retrieved_accession_numbers, expected_accession_number):
    """ 
    Return 1/rank for the first relevant retrieved result.
    Rank starts at 1, return 0 if it was not retrieved
    """
    
    for rank, accesion_number in enumerate(retrieved_accession_numbers,start=1):
        if accesion_number==expected_accession_number:
            return 1/rank

    return 0.0


def mean_reciprocal_rank(
    retrieved_accession_numbers: list[str],
    expected_accession_numbers: list[str],
) -> float:
    """Average reciprocal rank across every document required by a question."""
    if not expected_accession_numbers:
        return 0.0
    return sum(
        reciprocal_rank(retrieved_accession_numbers, expected)
        for expected in expected_accession_numbers
    ) / len(expected_accession_numbers)


def normalize_accession_number(value) -> str:
    """Normalize IDs read from CSV/Qdrant before metric comparison."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def parse_expected_documents(value) -> list[str]:
    """Parse one or more semicolon-delimited expected accession numbers."""
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        if value is None or pd.isna(value):
            return []
        raw_values = str(value).split(";")
    return [
        normalized
        for item in raw_values
        if (normalized := normalize_accession_number(item))
    ]


def normalize_run_config(run_config) -> dict:
    """Accept dictionaries and notebook/Pydantic configuration objects."""
    if isinstance(run_config, Mapping):
        return dict(run_config)
    if hasattr(run_config, "get_dictionary"):
        return dict(run_config.get_dictionary())
    if hasattr(run_config, "model_dump"):
        return dict(run_config.model_dump())
    raise TypeError(
        "run_config must be a mapping or expose get_dictionary()/model_dump()."
    )


def unique_ranked_documents(retrieved_accession_numbers) -> list[str]:
    """Collapse repeated chunks while preserving document retrieval order."""
    return list(dict.fromkeys(
        normalized
        for value in retrieved_accession_numbers
        if (normalized := normalize_accession_number(value))
    ))


def is_answerable(value) -> bool:
    """Convert CSV and Python boolean representations without truthy strings."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
        raise ValueError(f"Unknown answerable value: {value!r}")
    if value is None or pd.isna(value):
        return False
    return bool(value)


def prepare_vector_index(run_config: dict) -> None:
    """Prepare each vector collection once per evaluation process."""
    index_key = (
        str(DB_PATH_NAME.resolve()),
        run_config["collection_name"],
        run_config.get("pipeline_version"),
        run_config.get("chunk_size"),
        run_config.get("chunk_overlap"),
        run_config.get("encoding_name"),
        run_config.get("embedding_model"),
    )
    if index_key not in _PREPARED_COLLECTIONS:
        ingestion(run_config)
        _PREPARED_COLLECTIONS.add(index_key)


def evaluate_question_deterministic(
    question_record: dict,
    run_config: dict,
    qdrant_client,
    retrieval_only: bool = False,
):
    
    # Answer the question using the current run_config
    if retrieval_only:
        result = retrieve(
            question_record["question"], run_config, qdrant_client
        )
    else:
        result = answer(
            question_record["question"],
            run_config=run_config,
            qdrant_client=qdrant_client,
            history=None,
        )
    
    # Get the important values for each chunk
    chunks =  result['Retrieved Chunk texts']
    answer_text = result.get('Answer')
    cited_chunk_ids = result.get('Citations', [])
    
    retrieved_accession_numbers = [
        normalize_accession_number(chunk.get("accession_number"))
        for chunk in chunks
    ]
    ranked_documents = unique_ranked_documents(
        retrieved_accession_numbers
    )
    retrieved_chunk_ids = [chunk['chunk_id'] for chunk in chunks if chunk.get("chunk_id")]
    
    expected_documents = parse_expected_documents(
        question_record['expected_document']
    )
    expected_accession_number = "; ".join(expected_documents)
    
    answerable = is_answerable(question_record['answerable'])
    if answerable:
         # Retrieval: expected SEC filing appears anywhere in top K
        retrieved_document_set = set(ranked_documents)
        expected_document_set = set(expected_documents)
        matched_documents = expected_document_set & retrieved_document_set
        retrieval = expected_document_set.issubset(retrieved_document_set)
        document_recall = (
            len(matched_documents) / len(expected_document_set)
            if expected_document_set
            else 0.0
        )
        mrr = mean_reciprocal_rank(ranked_documents, expected_documents)
    else:
        retrieval = None
        document_recall = None
        mrr = None
        
    # Citation validity: Each citation was actually retrieved 
    invalid_citation  = [ citation for citation in cited_chunk_ids if citation not in retrieved_chunk_ids]
    if retrieval_only:
        citation_validity = None
    elif answerable:
        citation_validity = (len(cited_chunk_ids) > 0 and len(invalid_citation) == 0)
    else:
        citation_validity = len(invalid_citation) == 0
    
    # Abstention: only for question not answerable
    if not retrieval_only and not answerable:
        abstention_correct = (result["Answer"].strip() == "I could not find enough evidence in the retrieved SEC filings.")
    else:
        abstention_correct = None
    
    
    
    values = {
        # Evaluation question
        "question_id": question_record["question_id"],
        "question": question_record["question"],
        "answerable": answerable,
        "expected_document": expected_accession_number,
        "expected_documents": expected_documents,
        "reference_answer": question_record["reference_answer"],
        "category": question_record["category"],
        "difficulty": question_record["difficulty"],
        "evaluation_focus": question_record["evaluation_focus"],

        # Experiment identity
        "experiment_name": run_config["experiment_name"],
        "pipeline_version": run_config["pipeline_version"],
        "retrieval_k": run_config["retrieval_k"],
        "embedding_model": run_config["embedding_model"],
        "generation_model": run_config.get("generation_model"),
        "prompt_version": run_config.get("prompt_version"),

        # Deterministic retrieval metrics
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_chunks" : chunks,
        "retrieved_accession_numbers": retrieved_accession_numbers,
        "retrieved_ranked_documents": ranked_documents,
        "similarity_scores": [
            chunk["score"] for chunk in chunks
        ],
        "document_hit_at_k": retrieval,
        "document_recall_at_k": document_recall,
        "reciprocal_rank": mrr,

        # Deterministic citation metrics
        "citations": cited_chunk_ids,
        "citations_valid": citation_validity,
        "invalid_citations": invalid_citation,

        # Deterministic abstention metric
        "abstention_correct": abstention_correct,

        # Output and operational data
        "answer": answer_text,
        "latency_ms": result["Latency"],
        
        #Token Usage
        "token_usage": result['Token_usage']
        
    }
    
    return values
    
async def evaluate_question_ragas(deterministic_result, metrics, return_values):

    if return_values:
        if not deterministic_result['answerable']:
            return {
                    **deterministic_result,
                    "ragas_faithfulness" : None,
                    "ragas_factual_correctness": None,
                    "ragas_context_precision": None
                }

        sample  = SingleTurnSample(user_input=deterministic_result['question'],
                                response=deterministic_result['answer'],
                                reference=deterministic_result['reference_answer'],
                                retrieved_contexts=[chunk['chunk_text'] for chunk in deterministic_result['retrieved_chunks']])


        faithfulness = await metrics["faithfulness_metric"].single_turn_ascore(sample)
        factual_correctness = await metrics["factual_correctness_metric"].single_turn_ascore(sample)
        context_precision = await metrics["context_precision_metric"].single_turn_ascore(sample)

        return {
            **deterministic_result,
            "ragas_faithfulness" : faithfulness,
            "ragas_factual_correctness": factual_correctness,
            "ragas_context_precision": context_precision
        }

    else:
        return {
                **deterministic_result,
                "ragas_faithfulness" : None,
                "ragas_factual_correctness": None,
                "ragas_context_precision": None
                }

async def evaluate_all_questions(questions: pd.DataFrame,run_config, metrics, qdrant_client,return_values, retrieval_only=False):
    question_results = []
    
    # Add a dict for each question with the respective results
    for _ ,row in tqdm(questions.iterrows(),total = len(questions), desc= 'Evaluating Questions'):
        question_dict = row.to_dict()
        
        try:
            deterministic_answer = evaluate_question_deterministic(
                question_dict, run_config, qdrant_client, retrieval_only
            )
            ragas_result = await evaluate_question_ragas(
                deterministic_answer, metrics, return_values and not retrieval_only
            )
        except Exception as error:
            raise RuntimeError(
                f"Evaluation failed for question {row['question_id']}: {error}"
            ) from error
        
        if retrieval_only:
            answer_correct = None
        elif ragas_result['answerable']:
            factual_correctness = ragas_result['ragas_factual_correctness']
            answer_correct = (
                factual_correctness >= run_config['answer_correct_threshold']
                if factual_correctness is not None
                else None
            )
        else:
            answer_correct=ragas_result['abstention_correct']
        
        question_results.append({
            "question_id": ragas_result["question_id"],
            "category": ragas_result["category"],
            "difficulty": ragas_result["difficulty"],
            "answerable": ragas_result["answerable"],
            "expected_documents": ragas_result["expected_documents"],

            "document_hit_at_k": ragas_result["document_hit_at_k"],
            "document_recall_at_k": ragas_result["document_recall_at_k"],
            "reciprocal_rank": ragas_result["reciprocal_rank"],
            "retrieved_chunk_ids": ragas_result["retrieved_chunk_ids"],
            "retrieved_accession_numbers": ragas_result[
                "retrieved_accession_numbers"
            ],
            "retrieved_ranked_documents": ragas_result[
                "retrieved_ranked_documents"
            ],
            "similarity_scores": ragas_result["similarity_scores"],
            "citations_valid": ragas_result["citations_valid"],
            "abstention_correct": ragas_result["abstention_correct"],

            "retrieval_latency_ms": ragas_result["latency_ms"]["retrieval"],
            "generation_latency_ms": ragas_result["latency_ms"]["generation"],
            "total_latency_ms": ragas_result["latency_ms"]["total"],
                        
            "ragas_answer_correct":answer_correct,
            "ragas_faithfulness": ragas_result["ragas_faithfulness"],
            "ragas_factual_correctness": (
                ragas_result["ragas_factual_correctness"]
            ),
            "ragas_context_precision": (
                ragas_result["ragas_context_precision"]
            ),
            
            "embedding_tokens": ragas_result["token_usage"]["embedding_total_tokens"],
            "generation_input_tokens": (ragas_result["token_usage"]["generation_input_tokens"]),
            "generation_output_tokens": (ragas_result["token_usage"]["generation_output_tokens"]),
            "rag_total_tokens": ragas_result["token_usage"]["rag_total_tokens"],
        })
        
   
    return {
    "metadata": {
        "experiment_name": run_config["experiment_name"],
        "pipeline_version": run_config["pipeline_version"],
        "retrieval_k": run_config["retrieval_k"],
        "embedding_model": run_config["embedding_model"],
        "generation_model": run_config.get("generation_model"),
        "prompt_version": run_config.get("prompt_version"),
    },
   
    "question_results": question_results
    }
    
def summarize_evaluation(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return one summary row for all questions and one row per difficulty."""

    def mean_or_none(series):
        series = series.dropna()
        return float(series.mean()) if not series.empty else None

    def sum_booleans(series):
        return int(series.astype("boolean").fillna(False).sum())

    def build_summary_row(scope: str, group: pd.DataFrame) -> dict:
        answerable = group[group["answerable"]]
        unanswerable = group[~group["answerable"]]

        return {
            "scope": scope,

            "total_questions": len(group),
            "answerable_questions": len(answerable),
            "unanswerable_questions": len(unanswerable),

            # Overall outcome
            "correct_answers": sum_booleans(group["ragas_answer_correct"]),
            "overall_answer_accuracy": mean_or_none(
                group["ragas_answer_correct"]
            ),

            # Retrieval / answer quality: answerable questions only
            "document_hit_rate_at_k": mean_or_none(
                answerable["document_hit_at_k"]
            ),
            "mean_document_recall_at_k": mean_or_none(
                answerable["document_recall_at_k"]
            ),
            "mrr": mean_or_none(answerable["reciprocal_rank"]),
            "answerable_correct_answers": sum_booleans(
                answerable["ragas_answer_correct"]
            ),
            "answerable_answer_accuracy": mean_or_none(
                answerable["ragas_answer_correct"]
            ),
            "citation_validity_rate": mean_or_none(
                group["citations_valid"]
            ),
            "mean_ragas_faithfulness": mean_or_none(
                answerable["ragas_faithfulness"]
            ),
            "mean_ragas_factual_correctness": mean_or_none(
                answerable["ragas_factual_correctness"]
            ),
            "mean_ragas_context_precision": mean_or_none(
                answerable["ragas_context_precision"]
            ),

            # Unanswerable questions only
            "abstention_accuracy": mean_or_none(
                unanswerable["abstention_correct"]
            ),

            # Efficiency: all questions
            "mean_total_latency_ms": mean_or_none(
                group["total_latency_ms"]
            ),
            "p95_total_latency_ms": (
                float(group["total_latency_ms"].quantile(0.95))
                if not group.empty
                else None
            ),
            "mean_rag_total_tokens": mean_or_none(
                group["rag_total_tokens"]
            ),
            "total_rag_tokens": int(group["rag_total_tokens"].sum()),
        }

    summary_rows = [
        build_summary_row("All questions", results_df),
    ]

    for difficulty in ["hard", "medium"]:
        group = results_df[
            results_df["difficulty"].str.lower() == difficulty
        ]

        summary_rows.append(
            build_summary_row(difficulty.title(), group)
        )

    return pd.DataFrame(summary_rows)


async def evaluate(df_questions, run_config,metrics=None,return_values=True, retrieval_only=False):
    run_config = normalize_run_config(run_config)
    prepare_vector_index(run_config)
    qdrant_client = QdrantClient(path=str(DB_PATH_NAME))
    try:
        evaluated_questions = await evaluate_all_questions(
            df_questions, run_config, metrics, qdrant_client,
            return_values, retrieval_only
        )
    finally:
        qdrant_client.close()
    formatted_questions = pd.DataFrame(evaluated_questions['question_results'])
    summary = summarize_evaluation(formatted_questions)
    
    return formatted_questions, summary


async def main(run_config: dict,return_ragas_values=True, retrieval_only=False):
    run_config = normalize_run_config(run_config)
    questions = load_evaluation_questions()

    ragas_client = None
    metrics = None
    if not retrieval_only and return_ragas_values:
        ragas_client, metrics = create_ragas_metrics(run_config)

    try:
        detailed_questions, summary = await evaluate(
            questions, run_config, metrics, return_ragas_values, retrieval_only
        )

        project_root = Path(__file__).resolve().parents[3]

        output_directory = (
            project_root
            / "data"
            / "rag_evaluation"
            / "results"
            / run_config["experiment_name"]
        )

        if retrieval_only:
            output_directory = output_directory / "retrieval_only"

        output_directory.mkdir(parents=True, exist_ok=True)

        detailed_questions.to_csv(output_directory / "question_results.csv",index=False)

        summary.to_csv(output_directory / "summary.csv",index=False,)

        metadata = pd.DataFrame([{
            "experiment_name": run_config["experiment_name"],
            "pipeline_version": run_config["pipeline_version"],
            "retrieval_k": run_config["retrieval_k"],
            "embedding_model": run_config["embedding_model"],
            "generation_model": run_config.get("generation_model"),
            "temperature": run_config.get("temperature"),
            "prompt_version": run_config.get("prompt_version"),
            "ragas_evaluator_model": run_config.get("ragas_evaluator_model"),
            "ragas_temperature": run_config.get("ragas_temperature"),
            "evaluation_mode": (
                "retrieval_only" if retrieval_only else "full_rag"
            ),
        }])

        metadata.to_csv(output_directory / "metadata.csv",index=False)

        return detailed_questions, summary

    finally:
        if ragas_client is not None:
            await ragas_client.close()
        
        
if __name__ == "__main__":
    import asyncio
    asyncio.run(main(BASELINE_RUN_CONFIG))

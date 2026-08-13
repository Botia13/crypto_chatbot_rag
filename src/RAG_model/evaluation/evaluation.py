import os
import pandas as pd
from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness,FactualCorrectness,LLMContextPrecisionWithReference
from RAG_model.answer.answer import answer
from RAG_model.ingestion.config import BASELINE_RUN_CONFIG, OPENROUTER_BASE_URL
from pathlib import Path
from tqdm.auto import tqdm

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


def evaluate_question_deterministic(question_record:dict, run_config:dict):
    
    # Answer the question using the current run_config
    result  = answer(question_record['question'], run_config=run_config ,history=None)
    
    # Get the important values for each chunk
    chunks =  result['Retrieved Chunk texts']
    answer_text = result['Answer']
    cited_chunk_ids = result['Citations']
    
    retrieved_accession_numbers  = [chunk['accession_number'] for chunk in chunks if chunk.get("accession_number")]
    retrieved_chunk_ids = [chunk['chunk_id'] for chunk in chunks if chunk.get("chunk_id")]
    
    expected_accession_number = str(question_record['expected_document']).strip()
    
    if question_record['answerable']:
         # Retrieval: expected SEC filing appears anywhere in top K
        retrieval  = expected_accession_number in retrieved_accession_numbers
        #MRR: Find the rank of the first expected retrieve accession #
        mrr = reciprocal_rank(retrieved_accession_numbers,expected_accession_number)
    else:
        retrieval =  None 
        mrr = None
        
    # Citation validity: Each citation was actually retrieved 
    invalid_citation  = [ citation for citation in cited_chunk_ids if citation not in retrieved_chunk_ids]
    if question_record["answerable"]:
        citation_validity = (len(cited_chunk_ids) > 0 and len(invalid_citation) == 0)
    else:
        citation_validity = len(invalid_citation) == 0
    
    # Abstention: only for question not answerable
    if not question_record["answerable"]:
        abstention_correct = (result["Answer"].strip() == "I could not find enough evidence in the retrieved SEC filings.")
    else:
        abstention_correct = None
    
    
    
    values = {
        # Evaluation question
        "question_id": question_record["question_id"],
        "question": question_record["question"],
        "answerable": question_record["answerable"],
        "expected_document": expected_accession_number,
        "reference_answer": question_record["reference_answer"],
        "category": question_record["category"],
        "difficulty": question_record["difficulty"],
        "evaluation_focus": question_record["evaluation_focus"],

        # Experiment identity
        "experiment_name": run_config["experiment_name"],
        "pipeline_version": run_config["pipeline_version"],
        "retrieval_k": run_config["retrieval_k"],
        "embedding_model": run_config["embedding_model"],
        "generation_model": run_config["generation_model"],
        "prompt_version": run_config["prompt_version"],

        # Deterministic retrieval metrics
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_chunks" : chunks,
        "retrieved_accession_numbers": retrieved_accession_numbers,
        "similarity_scores": [
            chunk["score"] for chunk in chunks
        ],
        "document_hit_at_k": retrieval,
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
    
async def evaluate_question_ragas(deterministic_result, metrics):

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

async def evaluate_all_questions(questions: pd.DataFrame,run_config, metrics):
    question_results = []
    
    # Add a dict for each question with the respective results
    for _ ,row in tqdm(questions.iterrows(),total = len(questions), desc= 'Evaluating Questions'):
        question_dict = row.to_dict()
        
        deterministic_answer = evaluate_question_deterministic(question_dict,run_config)
        try:
            ragas_result = await evaluate_question_ragas(deterministic_answer, metrics)
        except Exception as error:
            print(f"Error in Question:{row['question_id']}, with error {str(error)} ")
            continue 
        
        if ragas_result['answerable']:
            answer_correct = (
                ragas_result['ragas_factual_correctness']>= run_config['answer_correct_threshold']
            )
        else:
            answer_correct=ragas_result['abstention_correct']
        
        question_results.append({
            "question_id": ragas_result["question_id"],
            "category": ragas_result["category"],
            "difficulty": ragas_result["difficulty"],
            "answerable": ragas_result["answerable"],

            "document_hit_at_k": ragas_result["document_hit_at_k"],
            "reciprocal_rank": ragas_result["reciprocal_rank"],
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
        "generation_model": run_config["generation_model"],
        "prompt_version": run_config["prompt_version"],
    },
   
    "question_results": question_results
    }
    
def summarize_evaluation(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return one summary row for all questions and one row per difficulty."""

    def mean_or_none(series):
        series = series.dropna()
        return float(series.mean()) if not series.empty else None

    def sum_booleans(series):
        return int(series.fillna(False).sum())

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


async def evaluate(df_questions, run_config,metrics):
    
    evaluated_questions = await evaluate_all_questions(df_questions,run_config,metrics)
    formatted_questions = pd.DataFrame(evaluated_questions['question_results'])
    summary = summarize_evaluation(formatted_questions)
    
    return formatted_questions, summary


async def main(run_config: dict):
    questions = load_evaluation_questions()

    ragas_client, metrics = create_ragas_metrics(run_config)

    try:
        detailed_questions, summary = await evaluate(questions,run_config,metrics)

        project_root = Path(__file__).resolve().parents[3]

        output_directory = (
            project_root
            / "data"
            / "rag_evaluation"
            / "results"
            / run_config["experiment_name"]
        )

        output_directory.mkdir(parents=True, exist_ok=True)

        detailed_questions.to_csv(output_directory / "question_results.csv",index=False)

        summary.to_csv(output_directory / "summary.csv",index=False,)

        metadata = pd.DataFrame([{
            "experiment_name": run_config["experiment_name"],
            "pipeline_version": run_config["pipeline_version"],
            "retrieval_k": run_config["retrieval_k"],
            "embedding_model": run_config["embedding_model"],
            "generation_model": run_config["generation_model"],
            "temperature": run_config["temperature"],
            "prompt_version": run_config["prompt_version"],
            "ragas_evaluator_model": run_config["ragas_evaluator_model"],
            "ragas_temperature": run_config["ragas_temperature"],
        }])

        metadata.to_csv(output_directory / "metadata.csv",index=False)

        return detailed_questions, summary

    finally:
        await ragas_client.close()
        
        
if __name__ == "__main__":
    import asyncio
    asyncio.run(main(BASELINE_RUN_CONFIG))
        
    
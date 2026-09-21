"""Lazy RAGAS construction shared by evaluation entrypoints."""
import os
from RAG_model.ingestion.config import OPENROUTER_BASE_URL

def create_ragas_metrics(run_config):
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextPrecisionWithReference
    ragas_client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL, timeout=60.0 , max_retries = 5 )
    ragas_llm = llm_factory(run_config['ragas_evaluator_model'],provider="openai", client = ragas_client, max_tokens = 8192, temperature=run_config["ragas_temperature"])

    # Create metric Objects
    metrics = {
        "faithfulness_metric": Faithfulness(llm=ragas_llm),
        "factual_correctness_metric": FactualCorrectness(llm=ragas_llm,mode="recall",atomicity='high',coverage='high'),
        "context_precision_metric":LLMContextPrecisionWithReference(llm=ragas_llm)}
    
    return ragas_client, metrics

from pydantic import BaseModel




class run_config(BaseModel):
    
    pipeline_version: str 
    prompt_version: str = "v2"
    retrieval_k: int = 5,
    candidate_k: int = 30,
    generation_model: str= 'openai/gpt-4o-mini'
    temperature: int = 0
    embedding_model:str   = "openai/text-embedding-3-small"
    chunk_size: int  = 500
    chunk_overlap: int = 75
    encoding_name:str = "cl100k_base"
    embedding_batch_size:int = 50
    ragas_evaluator_model:str = "openai/gpt-4o-mini"
    
    
    
    def get_dictionary(self):
        
        generation_label = self.generation_model.replace("/", "-").replace(":", "-")
        embedding_label = self.embedding_model.replace("/", "-").replace(":", "-")
        
        return {
        # Identity: saved with every experiment result
        "experiment_name": (
            f"chunk-{self.chunk_size}"
            f"__overlap-{self.chunk_overlap}"
            f"__embedding-{embedding_label}"
            f"__k-{self.retrieval_k}"
            f"__generator-{generation_label}"
            f"__prompt-{self.prompt_version}"
        ),
        "pipeline_version": self.pipeline_version,

        # Retrieval / generation
        "retrieval_k": self.retrieval_k,
        "candidate_k": self.candidate_k,
        "generation_model": self.generation_model,
        "temperature": self.temperature,
        "prompt_version": self.prompt_version,

        # Embeddings / vector collection
        "embedding_provider": "openrouter",
        "embedding_model": self.embedding_model,
        "embedding_label": embedding_label,
        "collection_name" : (f"sec_filings"
                f"__chunk-{self.chunk_size}"
                f"__overlap-{self.chunk_overlap}"
                f"__encoding-{self.encoding_name}"
                f"__embedding-{embedding_label}"
            ),

        # Ingestion configuration: record it for reproducibility
        "chunk_size": self.chunk_size,
        "chunk_overlap": self.chunk_overlap,
        "encoding_name": self.encoding_name,
        "embedding_batch_size": self.embedding_batch_size,
        
        # RAGAS paramaters
        "ragas_enabled": True,
        "ragas_evaluator_model": self.ragas_evaluator_model,
        "answer_correct_threshold": 0.8,
        "ragas_temperature": 0
    }   
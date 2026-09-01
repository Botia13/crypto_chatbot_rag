import pandas as pd
from RAG_model.evaluation.evaluation import main as evaluate
from pydantic import BaseModel
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from time import perf_counter
from tqdm import tqdm
import numpy as np




class run_config(BaseModel):
    
    pipeline_version: str 
    prompt_version: str = "v2"
    retrieval_k: int = 5
    candidate_k: int = 30
    generation_model: str= 'openai/gpt-4o-mini'
    temperature: int = 0
    embedding_model:str   = "openai/text-embedding-3-small"
    chunk_size: int  = 500
    chunk_overlap: int = 75
    encoding_name:str = "cl100k_base"
    embedding_batch_size:int = 50
    ragas_evaluator_model:str = "openai/gpt-4o-mini"
    rerank:bool = False
    rerank_model:str = "BAAI/bge-reranker-v2-m3"
    
    
    
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
            f"__rerank-{self.rerank}__{self.pipeline_version}"
        ),
        "pipeline_version": self.pipeline_version,

        # Retrieval / generation
        "retrieval_k": self.retrieval_k,
        "rerank":self.rerank,
        "rerank_model":self.rerank_model,
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
        
        
def plot_retrieval_metrics(
    df: pd.DataFrame,
    x_axis: str
) -> None:
    """Plot Document Hit Rate ,MRR and Recall for All questions only."""

    metrics = {
        "document_hit_rate_at_k": "Document Hit Rate@k",
        "mrr": "MRR",
        "mean_document_recall_at_k": "Document Recall" 
    }

    plot_df = df.loc[
        df["scope"].eq("All questions"),
        [x_axis, *metrics.keys()],
    ].copy()

    # Ensures 3, 5, 8, 10 instead of alphabetical ordering.
    plot_df[x_axis] = pd.to_numeric(
        plot_df[x_axis],
        errors="raise",
    )

    plot_df = plot_df.sort_values(x_axis)

    # Prevent accidentally averaging multiple experiments together.
    if plot_df[x_axis].duplicated().any():
        raise ValueError(
            f"There is more than one 'All questions' row per {x_axis}. "
            "Filter the dataframe to one index configuration first."
        )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 9),
        sharex=True,
    )

    for ax, (metric, label) in zip(axes, metrics.items()):
        sns.lineplot(
            data=plot_df,
            x=x_axis,
            y=metric,
            marker="o",
            markersize=9,
            linewidth=2.5,
            color="#1f77b4",
            ax=ax,
        )

        for _, row in plot_df.iterrows():
            ax.annotate(
                f"{row[metric]:.2f}",
                xy=(row[x_axis], row[metric]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )

        ax.set_title(label, fontsize=14)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Retrieval k", fontsize=11)
    axes[-1].set_xticks(plot_df[x_axis].tolist())

    fig.suptitle(
        f"Retrieval Performance by {x_axis} — All Questions",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()
    plt.show()
    
    
    
def plot_retrieval_by_config(
    df: pd.DataFrame,
    group_by: list[str],
) -> None:
    """Plot Document Hit Rate, MRR and Recall across arbitrary configuration
    columns, for 'All questions' scope only. Assumes df is already filtered
    to a single value for every other varying parameter (e.g. retrieval_k)
    so exactly one row per combination of `group_by` columns remains.

    Args:
        group_by: columns whose combination defines the x-axis categories,
            e.g. ["chunk_size", "chunk_overlap"], ["embedding_model"],
            ["generation_model", "prompt_version"], etc.
    """

    if not group_by:
        raise ValueError("group_by must contain at least one column.")

    metrics = {
        "document_hit_rate_at_k": "Document Hit Rate@k",
        "mrr": "MRR",
        "mean_document_recall_at_k": "Document Recall",
    }

    missing = [col for col in group_by if col not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in df: {missing}")

    cols = [*group_by, *metrics.keys()]
    plot_df = df.loc[df["scope"].eq("All questions"), cols].copy()

    # Build a categorical x-axis label from the requested columns,
    # e.g. "chunk_size=500, chunk_overlap=75"
    plot_df["config"] = plot_df[group_by].astype(str).agg(
        lambda row: ", ".join(f"{col}={val}" for col, val in zip(group_by, row)),
        axis=1,
    )

    # Sort configs by the underlying columns (numeric-aware where possible)
    # so the x-axis reads sensibly instead of alphabetically.
    unique_configs = plot_df[[*group_by, "config"]].drop_duplicates().copy()
    for col in group_by:
        try:
            unique_configs[col] = pd.to_numeric(unique_configs[col])
        except (ValueError, TypeError):
            pass  # leave as string, sort_values will fall back to lexical order

    config_order = unique_configs.sort_values(group_by)["config"].tolist()
    plot_df["config"] = pd.Categorical(
        plot_df["config"], categories=config_order, ordered=True
    )
    plot_df = plot_df.sort_values("config")

    if plot_df["config"].duplicated().any():
        raise ValueError(
            f"There is more than one 'All questions' row per {group_by} "
            "combination. Filter df to a single value for every other "
            "varying parameter (e.g. retrieval_k) first."
        )

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

    for ax, (metric, label) in zip(axes, metrics.items()):
        sns.lineplot(
            data=plot_df,
            x="config",
            y=metric,
            marker="o",
            markersize=9,
            linewidth=2.5,
            color="#1f77b4",
            ax=ax,
        )

        for _, row in plot_df.iterrows():
            ax.annotate(
                f"{row[metric]:.2f}",
                xy=(row["config"], row[metric]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )

        ax.set_title(label, fontsize=14)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel(", ".join(group_by), fontsize=11)
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")

    fig.suptitle(
        f"Retrieval Performance by {', '.join(group_by)} — All Questions",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()
    plt.show()
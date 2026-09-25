import json
from collections.abc import Callable
from functools import partial
from html import escape
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from RAG_model.core.settings import PROJECT_ROOT
from RAG_model.service.rag_service import get_rag_service

EVALUATION_REPORT = (
    PROJECT_ROOT / "artifacts" / "evaluation" / "summary.json"
)

EMPTY_CITATIONS = "### Citations\n\n*No citations are available yet.*"
EMPTY_STATUS = "*Submit a question to see request details.*"

TOKEN_METRICS = (
    ("Rewrite input", "rewrite_input_tokens"),
    ("Rewrite output", "rewrite_output_tokens"),
    ("Rewrite total", "rewrite_total_tokens"),
    ("Embedding input", "embedding_input_tokens"),
    ("Embedding total", "embedding_total_tokens"),
    ("Generation input", "generation_input_tokens"),
    ("Generation output", "generation_output_tokens"),
    ("Generation total", "generation_total_tokens"),
    ("RAG total", "rag_total_tokens"),
)

LATENCY_METRICS = (
    ("Query rewrite", "query_rewrite"),
    ("Retrieval", "retrieval"),
    ("Prompt building", "prompt_building"),
    ("Generation", "generation"),
    ("Total", "total"),
)

SUGGESTED_QUESTIONS = (
    (
        "In IBIT’s Q1 2025 filing, where do the shares trade, which benchmark "
        "is used to value bitcoin, when is NAV calculated, and when is it "
        "normally released?"
    ),
    (
        "How does FBTC issue or redeem shares, and how is a Basket tied to "
        "the Trust's bitcoin holdings?"
    ),
    (
        "Summarize the main investor-facing differences between GBTC and ETHE "
        "using underlying asset, ticker, and trading venue."
    ),
)

EVALUATION_TITLE = "Evaluation Results of the model with dataset"
EVALUATION_DESCRIPTION = (
    "We tested the model using 40 questions to evaluate its performance and "
    "iteratively improve the system. This evaluation set was used throughout "
    "the development process to identify weaknesses and guide improvements to "
    "the RAG pipeline.\n\n"
    "The dataset contains questions classified by difficulty as either "
    "**medium** or **hard**. Each question can also belong to one or more "
    "question categories, such as **factual**, **synthesis**, or "
    "**comparison**.\n\n"
    "After completing the evaluation and iterative improvement process, we "
    "obtained the results shown below."
)

RUN_CONFIG_FIELDS = (
    ("Chunk size", "chunk_size"),
    ("Chunk overlap", "chunk_overlap"),
    ("Encoding", "encoding_name"),
    ("Embedding model", "embedding_model"),
    ("Retrieval K", "retrieval_k"),
    ("Reranking enabled", "rerank"),
    ("Generation model", "generation_model"),
    ("Temperature", "temperature"),
    ("RAGAS evaluator model", "ragas_evaluator_model"),
    ("Prompt version", "prompt_version"),
)

EVALUATION_SCOPES = ("All questions", "Hard", "Medium")
EVALUATION_COLORS = {
    "All questions": "#2563EB",
    "Hard": "#DC2626",
    "Medium": "#16A34A",
}
EVALUATION_METRICS = (
    {
        "title": "Overall Answer Accuracy",
        "field": "overall_answer_accuracy",
        "unit": "percent",
        "y_title": "Score (%)",
        "y_lim": [0, 100],
        "description": (
            "The percentage of all evaluation questions answered correctly, "
            "including questions where abstaining is the correct response."
        ),
    },
    {
        "title": "MRR",
        "field": "mrr",
        "unit": "percent",
        "y_title": "Score (%)",
        "y_lim": [0, 100],
        "description": (
            "Mean Reciprocal Rank measures how highly the first relevant "
            "retrieved result appears. Higher values mean relevant evidence "
            "is found earlier."
        ),
    },
    {
        "title": "Faithfulness",
        "field": "mean_ragas_faithfulness",
        "unit": "percent",
        "y_title": "Score (%)",
        "y_lim": [0, 100],
        "description": (
            "RAGAS Faithfulness measures whether claims in the answer are "
            "supported by the retrieved SEC filing context."
        ),
    },
    {
        "title": "Correctness",
        "field": "mean_ragas_factual_correctness",
        "unit": "percent",
        "y_title": "Score (%)",
        "y_lim": [0, 100],
        "description": (
            "RAGAS Factual Correctness measures how closely the generated "
            "answer's facts match the reference answer."
        ),
    },
    {
        "title": "Context Precision",
        "field": "mean_ragas_context_precision",
        "unit": "percent",
        "y_title": "Score (%)",
        "y_lim": [0, 100],
        "description": (
            "RAGAS Context Precision measures whether relevant retrieved "
            "chunks are ranked ahead of irrelevant chunks."
        ),
    },
    {
        "title": "Mean Latency",
        "field": "mean_total_latency_ms",
        "unit": "seconds",
        "y_title": "Seconds",
        "y_lim": [0, None],
        "description": (
            "The average end-to-end time required to process one evaluation "
            "question and return an answer."
        ),
    },
    {
        "title": "Mean Tokens",
        "field": "mean_rag_total_tokens",
        "unit": "count",
        "y_title": "Tokens",
        "y_lim": [0, None],
        "description": (
            "The average total tokens used per evaluation question across "
            "query rewriting, embedding, and answer generation."
        ),
    },
)


def load_evaluation_report(path: Path = EVALUATION_REPORT) -> dict:
    if not path.exists():
        return {
            "error": "Evaluation report not found.",
        }

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": "Evaluation report could not be read."}


def format_run_config_rows(run_config: dict) -> list[list[str]]:
    """Select and format the configuration shown in the dashboard."""
    rows = []
    for label, key in RUN_CONFIG_FIELDS:
        if key not in run_config:
            raise ValueError(f"Evaluation run_config is missing {key}.")

        value = run_config[key]
        if isinstance(value, bool):
            formatted = "Yes" if value else "No"
        else:
            formatted = str(value)
        rows.append([label, formatted])

    return rows


def format_run_config_table(rows: list[list[str]]) -> str:
    """Render configuration as an HTML table with a prompt anchor link."""
    body = []
    for label, value in rows:
        safe_label = escape(label)
        safe_value = escape(value)
        if label == "Prompt version":
            safe_value = (
                '<a href="#evaluation-prompt">'
                f"{safe_value} — View prompt"
                "</a>"
            )
        body.append(
            "<tr>"
            f'<td style="padding:8px 12px;border:1px solid #d1d5db">'
            f"{safe_label}</td>"
            f'<td style="padding:8px 12px;border:1px solid #d1d5db">'
            f"{safe_value}</td>"
            "</tr>"
        )

    return (
        '<table style="width:100%;border-collapse:collapse">'
        "<thead><tr>"
        '<th style="text-align:left;padding:8px 12px;border:1px solid #d1d5db">'
        "Parameter</th>"
        '<th style="text-align:left;padding:8px 12px;border:1px solid #d1d5db">'
        "Value</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def format_prompt_viewer(prompt_version: str, system_prompt: str) -> str:
    """Render the evaluated system prompt as a readable anchored section."""
    return (
        '<section id="evaluation-prompt" style="scroll-margin-top:16px">'
        f"<h3>System prompt {escape(prompt_version)}</h3>"
        "<p>This is the system-prompt snapshot used by the evaluation. "
        "Retrieved context and conversation messages are appended at runtime."
        "</p>"
        '<pre style="white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;'
        'border:1px solid #d1d5db;border-radius:8px;line-height:1.5">'
        f"<code>{escape(system_prompt.strip())}</code></pre>"
        "</section>"
    )


def transform_metric_value(value: object, unit: str) -> float:
    """Convert report values into dashboard-friendly units."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Evaluation metric values must be numeric.")
    if unit == "percent":
        return round(float(value) * 100, 2)
    if unit == "seconds":
        return round(float(value) / 1000, 2)
    return round(float(value), 2)


def prepare_evaluation_dashboard(report: dict) -> dict:
    """Validate and transform the exported evaluation report for Gradio."""
    if report.get("error"):
        return {"error": report["error"]}

    try:
        run_config_rows = format_run_config_rows(report["run_config"])
        summary_by_scope = {
            row["scope"]: row for row in report["summary"]
        }
        missing_scopes = [
            scope for scope in EVALUATION_SCOPES if scope not in summary_by_scope
        ]
        if missing_scopes:
            raise ValueError("Evaluation report is missing required scopes.")

        charts = []
        for metric in EVALUATION_METRICS:
            values = [
                transform_metric_value(
                    summary_by_scope[scope][metric["field"]],
                    metric["unit"],
                )
                for scope in EVALUATION_SCOPES
            ]
            charts.append(
                {
                    "definition": metric,
                    "data": pd.DataFrame(
                        {
                            "Scope": EVALUATION_SCOPES,
                            "Value": values,
                        }
                    ),
                }
            )
    except (KeyError, TypeError, ValueError):
        return {"error": "Evaluation report is missing required data."}

    return {
        "run_config_rows": run_config_rows,
        "charts": charts,
        "prompt_version": str(report["run_config"]["prompt_version"]),
        "system_prompt": report.get("system_prompt"),
    }


def build_metric_figure(chart: dict) -> go.Figure:
    """Build a bar chart whose values remain visible above every bar."""
    metric = chart["definition"]
    scopes = chart["data"]["Scope"].tolist()
    values = chart["data"]["Value"].tolist()

    if metric["unit"] == "percent":
        labels = [f"{value:.2f}%" for value in values]
        y_range = [0, 110]
        hover_template = "%{x}: %{y:.2f}%<extra></extra>"
    elif metric["unit"] == "seconds":
        labels = [f"{value:.2f} s" for value in values]
        y_range = [0, max(values, default=0) * 1.2 or 1]
        hover_template = "%{x}: %{y:.2f} s<extra></extra>"
    else:
        labels = [f"{value:,.2f}" for value in values]
        y_range = [0, max(values, default=0) * 1.2 or 1]
        hover_template = "%{x}: %{y:,.2f}<extra></extra>"

    figure = go.Figure(
        go.Bar(
            x=scopes,
            y=values,
            marker_color=[EVALUATION_COLORS[scope] for scope in scopes],
            text=labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=hover_template,
        )
    )
    figure.update_layout(
        title=metric["title"],
        xaxis_title="Question difficulty",
        yaxis_title=metric["y_title"],
        yaxis_range=y_range,
        height=340,
        margin={"t": 65, "r": 20, "b": 55, "l": 60},
        showlegend=False,
        template="plotly_white",
        annotations=[
            {
                "x": 1,
                "y": 1.16,
                "xref": "paper",
                "yref": "paper",
                "text": "ⓘ",
                "showarrow": False,
                "font": {"size": 18, "color": "#475569"},
                "hovertext": metric["description"],
                "hoverlabel": {
                    "bgcolor": "#0f172a",
                    "font": {"color": "white", "size": 12},
                },
                "captureevents": True,
            }
        ],
    )
    return figure


def format_citations(citations: list[dict]) -> str:
    """Format structured citations for the chat-side Markdown panel."""
    if not citations:
        return "### Citations\n\n*No citations were returned for this answer.*"

    lines = ["### Citations", ""]
    for citation in citations:
        ticker = citation.get("ticker") or "Unknown ticker"
        section = citation.get("section") or "Section unavailable"
        filing_date = citation.get("filing_date")
        source_url = citation.get("source_url")

        detail = f"**{ticker} — {section}**"
        if filing_date:
            detail += f" · Filed {filing_date}"
        if source_url:
            detail += f" · [Open SEC filing]({source_url})"

        lines.append(f"- {detail}")

    return "\n".join(lines)


def format_metric_rows(
    values: dict,
    metrics: tuple[tuple[str, str], ...],
) -> list[list]:
    """Return table rows in a stable, user-friendly order."""
    return [[label, values.get(key)] for label, key in metrics]


def format_request_status(result: dict) -> str:
    """Format request identity and citation validation as one compact line."""
    resolved = result["validation"].get("citations_resolved", False)
    resolved_label = "Yes" if resolved else "No"
    return (
        f"**Request ID:** `{result['request_id']}` · "
        f"**Citations resolved:** {resolved_label}"
    )


def build_chat_outputs(
    question: str,
    history: list[dict],
    result: dict,
) -> tuple:
    """Build all dynamic Chat-tab outputs from one service result."""
    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": result["answer"]},
    ]
    return (
        "",
        updated_history,
        format_citations(result["citations"]),
        format_metric_rows(result["usage"], TOKEN_METRICS),
        format_metric_rows(result["timings_ms"], LATENCY_METRICS),
        format_request_status(result),
    )


def create_gradio_app(
    service_factory: Callable = get_rag_service,
    evaluation_report_path: Path = EVALUATION_REPORT,
) -> gr.Blocks:
    def ask(question: str, history: list[dict] | None):
        question = question.strip()
        history = history or []

        if not question:
            return "", history, EMPTY_CITATIONS, [], [], EMPTY_STATUS

        result = service_factory().query(
            question=question,
            history=history,
        )

        return build_chat_outputs(question, history, result)

    evaluation = prepare_evaluation_dashboard(
        load_evaluation_report(evaluation_report_path)
    )

    with gr.Blocks(title="Crypto SEC Filings RAG") as demo:
        with gr.Tab("Chat"):
            gr.Markdown(
                "# Chat Crypto Tickers\n\n"
                "Ask questions about IBIT, ETHA, FBTC, FETH, GBTC, and ETHE "
                "using their SEC 10-K and 10-Q filings from 2023 through "
                "Q1 2026."
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                    )
                    question = gr.Textbox(
                        label="Question",
                        placeholder="Ask about a crypto ticker's SEC filings",
                        lines=2,
                    )
                    gr.Markdown("#### Suggested questions")
                    suggestion_buttons = [
                        gr.Button(suggestion, variant="secondary")
                        for suggestion in SUGGESTED_QUESTIONS
                    ]
                    ask_button = gr.Button("Ask", variant="primary")

                with gr.Column(scale=2):
                    citations = gr.Markdown(
                        value=EMPTY_CITATIONS,
                        label="Citations",
                        show_label=True,
                        container=True,
                        height=500,
                    )

            with gr.Row():
                token_table = gr.Dataframe(
                    value=[],
                    headers=["Token metric", "Tokens"],
                    datatype=["str", "number"],
                    type="array",
                    label="Token usage",
                    interactive=False,
                    wrap=True,
                )
                latency_table = gr.Dataframe(
                    value=[],
                    headers=["Pipeline stage", "Milliseconds"],
                    datatype=["str", "number"],
                    type="array",
                    label="Latency",
                    interactive=False,
                    wrap=True,
                )

            request_status = gr.Markdown(EMPTY_STATUS)

            submit_inputs = [question, chatbot]
            submit_outputs = [
                question,
                chatbot,
                citations,
                token_table,
                latency_table,
                request_status,
            ]
            ask_button.click(
                ask,
                inputs=submit_inputs,
                outputs=submit_outputs,
            )
            question.submit(
                ask,
                inputs=submit_inputs,
                outputs=submit_outputs,
            )
            for suggestion_button, suggestion in zip(
                suggestion_buttons,
                SUGGESTED_QUESTIONS,
                strict=True,
            ):
                suggestion_button.click(
                    partial(ask, suggestion),
                    inputs=[chatbot],
                    outputs=submit_outputs,
                )

        with gr.Tab("Evaluation"):
            gr.Markdown(
                f"# {EVALUATION_TITLE}\n\n{EVALUATION_DESCRIPTION}"
            )

            if evaluation.get("error"):
                gr.Markdown(
                    "### Evaluation report unavailable\n\n"
                    f"{evaluation['error']}"
                )
            else:
                gr.HTML(
                    value=format_run_config_table(
                        evaluation["run_config_rows"]
                    ),
                    label="Run configuration",
                    show_label=True,
                    container=True,
                )

                charts = evaluation["charts"]
                for start in range(0, len(charts), 3):
                    chart_group = charts[start : start + 3]
                    with gr.Row():
                        for chart in chart_group:
                            with gr.Column(scale=1):
                                gr.Plot(value=build_metric_figure(chart))
                        for _ in range(3 - len(chart_group)):
                            with gr.Column(scale=1):
                                gr.Markdown("")

                if evaluation["system_prompt"]:
                    gr.HTML(
                        value=format_prompt_viewer(
                            evaluation["prompt_version"],
                            evaluation["system_prompt"],
                        )
                    )
                else:
                    gr.Markdown(
                        "### System prompt unavailable\n\n"
                        "This report does not contain a prompt snapshot."
                    )

        with gr.Tab("About"):
            gr.Markdown(
                "This system answers questions using SEC filing evidence.\n\n"
                "[View the project on GitHub]("
                "https://github.com/Botia13/crypto_chatbot_rag)"
            )

    return demo

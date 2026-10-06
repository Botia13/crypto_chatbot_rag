import json
from collections.abc import Callable
from functools import partial
from html import escape
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from RAG_model.core.settings import PROJECT_ROOT, get_settings
from RAG_model.service.rag_service import get_rag_service
from RAG_model.ui.presentation import (
    BORDER,
    CHAT_PLACEHOLDER,
    COPPER,
    INK,
    IVORY,
    MUTED,
    RESEARCH_HEADER,
    SLATE,
    TEAL,
)

EVALUATION_REPORT = PROJECT_ROOT / "artifacts" / "evaluation" / "summary.json"

EMPTY_CITATIONS = (
    "**Your evidence starts here.**\n\n"
    "Ask a question to see the supporting filing sections and SEC source links."
)
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

EVALUATION_TITLE = "Evidence behind the answers"
EVALUATION_DESCRIPTION = (
    "A saved evaluation of 40 research questions, used during development "
    "to assess and improve retrieval and answer quality.\n\n"
    "Compare **all questions**, **hard**, and **medium** across factual, "
    "synthesis, and comparison tasks. Scores describe this evaluation set; "
    "they do not guarantee the correctness of an individual answer. "
    "Hover over ⓘ for each metric's definition."
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
    "All questions": TEAL,
    "Hard": COPPER,
    "Medium": "#95AFBE",
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
            safe_value = f'<a href="#evaluation-prompt">{safe_value} — View prompt</a>'
        body.append(f"<tr><td>{safe_label}</td><td>{safe_value}</td></tr>")

    return (
        '<table class="config-table">'
        "<thead><tr>"
        '<th scope="col">Parameter</th>'
        '<th scope="col">Value</th>'
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def format_prompt_viewer(prompt_version: str, system_prompt: str) -> str:
    """Render the evaluated system prompt as a readable anchored section."""
    return (
        '<section id="evaluation-prompt">'
        f"<h3>System prompt {escape(prompt_version)}</h3>"
        "<p>This is the system-prompt snapshot used by the evaluation. "
        "Retrieved context and conversation messages are appended at runtime."
        "</p>"
        "<pre>"
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
        summary_by_scope = {row["scope"]: row for row in report["summary"]}
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
        title={"text": metric["title"], "font": {"size": 17}},
        xaxis_title="Question difficulty",
        yaxis_title=metric["y_title"],
        yaxis_range=y_range,
        height=340,
        margin={"t": 65, "r": 20, "b": 55, "l": 60},
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor=SLATE,
        plot_bgcolor=SLATE,
        font={"family": "Segoe UI, system-ui, sans-serif", "color": IVORY, "size": 12},
        bargap=0.48,
        hoverlabel={"bgcolor": INK, "font": {"color": IVORY}},
        annotations=[
            {
                "x": 1,
                "y": 1.16,
                "xref": "paper",
                "yref": "paper",
                "text": "ⓘ",
                "showarrow": False,
                "font": {"size": 18, "color": MUTED},
                "hovertext": metric["description"],
                "hoverlabel": {
                    "bgcolor": INK,
                    "font": {"color": IVORY, "size": 12},
                },
                "captureevents": True,
            }
        ],
    )
    figure.update_xaxes(showgrid=False, tickfont={"color": MUTED}, automargin=True)
    figure.update_yaxes(
        gridcolor=BORDER, zerolinecolor=BORDER, tickfont={"color": MUTED}
    )
    return figure


def format_citations(citations: list[dict]) -> str:
    """Format structured citations for the chat-side Markdown panel."""
    if not citations:
        return "*No citations were returned for this answer.*"

    lines = []
    for citation in citations:
        ticker = (citation.get("ticker") or "").strip() or "Unknown ticker"
        section = (citation.get("section") or "").strip() or "Section unavailable"
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


def clear_chat() -> tuple:
    """Reset every chat output without invoking the research service."""
    return "", [], EMPTY_CITATIONS, [], [], EMPTY_STATUS


def create_gradio_app(
    service_factory: Callable = get_rag_service,
    evaluation_report_path: Path = EVALUATION_REPORT,
) -> gr.Blocks:
    def ask(question: str, history: list[dict] | None):
        question = question.strip()
        history = history or []

        if not question:
            return "", history, EMPTY_CITATIONS, [], [], EMPTY_STATUS

        service = service_factory()
        history_limit = getattr(
            service, "max_history_messages", get_settings().max_history_messages
        )
        # Keep the complete conversation on screen, but bound the model context.
        request_history = history[-history_limit:] if history_limit else []
        result = service.query(
            question=question,
            history=request_history,
        )

        return build_chat_outputs(question, history, result)

    evaluation = prepare_evaluation_dashboard(
        load_evaluation_report(evaluation_report_path)
    )

    # Blocks' elem_id is not rendered by Gradio 6; scope CSS to a real node.
    with (
        gr.Blocks(title="Crypto Filing Research", fill_width=True) as demo,
        gr.Column(elem_id="research-app"),
    ):
        gr.HTML(RESEARCH_HEADER, container=False, elem_classes="unframed-content")
        with gr.Tabs(elem_id="research-tabs"):
            _build_research_tabs(ask, evaluation)
        gr.Markdown(
            "Research & education only · Verify important claims in the linked "
            "SEC filings · Not investment advice",
            elem_classes="research-footer",
        )

    return demo


def _build_research_tabs(ask: Callable, evaluation: dict) -> None:
    """Compose the workspace while keeping all six chat outputs stable."""
    with gr.Tab("Chat"):
        with gr.Row(elem_id="chat-workspace"):
            with gr.Column(scale=3, min_width=480):
                gr.HTML(
                    '<div class="workspace-heading"><div>'
                    '<span class="eyebrow">01 / ASK & EXPLORE</span>'
                    "<h2>Your research workspace</h2></div></div>",
                    container=False,
                    elem_classes="unframed-content",
                )
                chatbot = gr.Chatbot(
                    label="Conversation",
                    show_label=False,
                    height=440,
                    placeholder=CHAT_PLACEHOLDER,
                    buttons=["copy"],
                    elem_id="conversation",
                )
                with gr.Row(elem_id="question-composer"):
                    question = gr.Textbox(
                        label="Research question",
                        placeholder="What would you like to understand about these funds?",
                        lines=1,
                        max_lines=4,
                        scale=5,
                        min_width=220,
                        elem_id="question-input",
                    )
                    ask_button = gr.Button(
                        "Ask question ↗",
                        variant="primary",
                        scale=1,
                        min_width=140,
                        elem_id="ask-question",
                    )
                gr.Markdown(
                    "**Suggested questions**\n\nSelect a question to ask it.",
                    elem_classes="evidence-note",
                )
                with gr.Column(elem_id="example-prompts"):
                    suggestion_buttons = [
                        gr.Button(
                            suggestion,
                            variant="secondary",
                            min_width=0,
                            elem_classes="example-card",
                            elem_id=f"example-{index}",
                        )
                        for index, suggestion in enumerate(SUGGESTED_QUESTIONS)
                    ]
                clear_button = gr.Button(
                    "Clear chat",
                    size="sm",
                    elem_id="clear-chat",
                )

            with gr.Column(scale=1, min_width=280, elem_id="evidence-panel"):
                gr.HTML(
                    '<div class="workspace-heading"><div>'
                    '<span class="eyebrow">02 / FOLLOW THE EVIDENCE</span>'
                    "<h2>Sources & citations</h2>"
                    "<p>Supporting evidence for the latest answer.</p></div></div>",
                    container=False,
                    elem_classes="unframed-content",
                )
                citations = gr.Markdown(
                    value=EMPTY_CITATIONS,
                    container=False,
                    height=500,
                    elem_id="citations",
                )
                gr.Markdown(
                    "**Read the original.** Each source links to a public SEC filing. "
                    "A resolved citation confirms that its identifier belongs to "
                    "the retrieved evidence; it does not establish factual correctness.",
                    elem_classes="evidence-note",
                )

        with gr.Accordion("Request details", open=False, elem_id="request-details"):
            request_status = gr.Markdown(EMPTY_STATUS, elem_id="request-status")
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

        submit_inputs = [question, chatbot]
        submit_outputs = [
            question,
            chatbot,
            citations,
            token_table,
            latency_table,
            request_status,
        ]
        submit_events = [
            ask_button.click(ask, inputs=submit_inputs, outputs=submit_outputs),
            question.submit(ask, inputs=submit_inputs, outputs=submit_outputs),
        ]
        for suggestion_button, suggestion in zip(
            suggestion_buttons,
            SUGGESTED_QUESTIONS,
            strict=True,
        ):
            submit_events.append(
                suggestion_button.click(
                    partial(ask, suggestion),
                    inputs=[chatbot],
                    outputs=submit_outputs,
                )
            )
        clear_button.click(
            clear_chat,
            outputs=submit_outputs,
            cancels=submit_events,
            queue=False,
        )
        # Gradio also exposes a clear control inside the conversation itself.
        chatbot.clear(
            clear_chat,
            outputs=submit_outputs,
            cancels=submit_events,
            queue=False,
        )

    with gr.Tab("Evaluation"):
        gr.Markdown(
            f"## {EVALUATION_TITLE}\n\n{EVALUATION_DESCRIPTION}",
            elem_classes="section-intro",
        )
        if evaluation.get("error"):
            gr.Markdown(
                f"### Evaluation report unavailable\n\n{evaluation['error']}",
            )
        else:
            charts = evaluation["charts"]
            for start in range(0, len(charts), 3):
                chart_group = charts[start : start + 3]
                with gr.Row(elem_classes="chart-row"):
                    for chart in chart_group:
                        with gr.Column(scale=1, min_width=280):
                            gr.Plot(
                                value=build_metric_figure(chart),
                                show_label=False,
                                elem_classes="metric-chart",
                            )
                    for _ in range(3 - len(chart_group)):
                        with gr.Column(scale=1, min_width=280):
                            gr.Markdown("")

            with gr.Accordion("Evaluation run configuration", open=False):
                gr.HTML(
                    value=format_run_config_table(evaluation["run_config_rows"]),
                    container=False,
                )

            if evaluation["system_prompt"]:
                gr.HTML(
                    value=format_prompt_viewer(
                        evaluation["prompt_version"],
                        evaluation["system_prompt"],
                    ),
                )
            else:
                gr.Markdown(
                    "### System prompt unavailable\n\n"
                    "This report does not contain a prompt snapshot.",
                )

    with gr.Tab("About"):
        gr.Markdown(
            "## Research with a source trail\n\n"
            "Crypto fund disclosures are dense. This workspace helps you ask "
            "focused questions and return to the SEC evidence behind an answer.",
            elem_classes="section-intro",
        )
        with gr.Row():
            gr.Markdown(
                "### A focused filing library\n\n"
                "Selected **10-K and 10-Q filings from 2023 through Q1 2026** "
                "for six crypto funds:\n\n"
                "- **IBIT / ETHA** — iShares Bitcoin and Ethereum trusts\n"
                "- **FBTC / FETH** — Fidelity Bitcoin and Ethereum funds\n"
                "- **GBTC / ETHE** — Grayscale Bitcoin and Ethereum trusts\n\n"
                "Explore fund mechanics, reporting details, and comparisons "
                "supported by this library.",
                container=True,
                elem_classes="about-card",
            )
            gr.Markdown(
                "### From question to evidence\n\n"
                "The system retrieves relevant filing sections, generates "
                "an answer from that context, and returns supporting citations.\n\n"
                "Open the source links to inspect the original disclosures. "
                "Citation resolution checks identifiers against retrieved "
                "evidence; it does not prove every claim is correct.\n\n"
                "The Evaluation tab shows a saved benchmark and the system "
                "prompt used for that run.",
                container=True,
                elem_classes="about-card",
            )
        gr.Markdown(
            "### Know the boundaries\n\n"
            "This is a fixed, selected corpus, not a live market feed or a "
            "complete SEC archive. New filings require an index update. Answers "
            "can be incomplete or incorrect, and the system may abstain when "
            "retrieved evidence is insufficient. Conversation history is "
            "limited and is not persisted.\n\n"
            "Use this tool for research and education, not investment advice. "
            "Verify important information in the original filing.\n\n"
            "[View the project on GitHub]("
            "https://github.com/Botia13/crypto_chatbot_rag)",
            container=True,
            elem_classes="about-card",
        )

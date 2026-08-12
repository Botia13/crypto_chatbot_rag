import gradio as gr

from RAG_model.answer.answer import answer
from RAG_model.ingestion.config import BASELINE_RUN_CONFIG

def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "*No chunks were retrieved.*"

    blocks = ["## Retrieved SEC filing context"]
    for rank, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"### {rank}. `{chunk['chunk_id']}`\n"
            f"- Similarity score: `{chunk['score']:.4f}`\n"
            f"- Ticker: `{chunk['ticker']}`\n"
            f"- Filing date: `{chunk['filing_date']}`\n"
            f"- Section: {chunk['section_title']}\n"
            f"- [Open SEC filing]({chunk['source_url']})\n\n"
            f"{chunk['chunk_text']}"
            f"- Section key: `{chunk.get('section_key', 'N/A')}`\n"
            f"- Table: {chunk.get('table_title') or 'N/A'}\n"
            f"- Table type: {chunk.get('table_type') or 'text'}\n"
        )
    return "\n\n---\n\n".join(blocks)


def ask_question(question: str, history: list[dict] | None):
    history = history or []
    question = question.strip()

    if not question:
        return "", history, "*Enter a question first.*", {}

    try:
        result = answer(question, run_config=BASELINE_RUN_CONFIG,history=history)
    except Exception as error:
        error_message = "I could not complete the request. Check the terminal for details."
        updated_history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": error_message},
        ]
        return "", updated_history, f"## Error\n\n`{type(error).__name__}: {error}`", {}

    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": result["Answer"]},
    ]
    diagnostics = {
        "citations": result["Citations"],
        "similarity_scores": result["Similarity Scores"],
        "latency_ms": result["Latency"],
    }

    return "", updated_history, format_context(result["Retrieved Chunk texts"]), diagnostics


def clear_chat():
    return [], "*Retrieved context will appear here.*", {}


def main() -> None:
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Crypto SEC Filings RAG", theme=theme) as app:
        gr.Markdown(
            "# Crypto SEC Filings RAG\n"
            "Ask questions about IBIT, ETHA, FBTC, FETH, GBTC, and ETHE. "
            "Answers are based only on retrieved SEC filing extracts."
        )

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                    buttons=["copy", "copy_all"],
                )
                question = gr.Textbox(
                    label="Question",
                    placeholder="For example: What is the ticker symbol for Fidelity Ethereum Fund?",
                    lines=2,
                )
                with gr.Row():
                    ask_button = gr.Button("Ask", variant="primary")
                    clear_button = gr.Button("Clear")

            with gr.Column(scale=1):
                context = gr.Markdown(
                    value="*Retrieved context will appear here.*",
                    height=600,
                )
                diagnostics = gr.JSON(label="Citations, similarity scores, and latency")

        submit_inputs = [question, chatbot]
        submit_outputs = [question, chatbot, context, diagnostics]
        question.submit(ask_question, inputs=submit_inputs, outputs=submit_outputs)
        ask_button.click(ask_question, inputs=submit_inputs, outputs=submit_outputs)
        clear_button.click(clear_chat, outputs=[chatbot, context, diagnostics])

    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()

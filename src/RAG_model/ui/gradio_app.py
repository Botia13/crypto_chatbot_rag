from collections.abc import Callable

import gradio as gr

from RAG_model.service.rag_service import get_rag_service


def create_gradio_app(
    service_factory: Callable = get_rag_service,
) -> gr.Blocks:
    def ask(question: str, history: list[dict] | None):
        question = question.strip()
        history = history or []

        if not question:
            return "", history, [], {}

        result = service_factory().query(
            question=question,
            history=history,
        )

        updated_history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result["answer"]},
        ]

        metadata = {
            "request_id": result["request_id"],
            "usage": result["usage"],
            "timings_ms": result["timings_ms"],
            "validation": result["validation"],
        }

        return (
            "",
            updated_history,
            result["retrieved_chunks"],
            metadata,
        )

    with gr.Blocks(title="Crypto SEC Filings RAG") as demo:
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Conversation")
            question = gr.Textbox(label="Question")
            ask_button = gr.Button("Ask", variant="primary")
            evidence = gr.JSON(label="Retrieved evidence")
            metadata = gr.JSON(label="Request metadata")

            ask_button.click(
                ask,
                inputs=[question, chatbot],
                outputs=[
                    question,
                    chatbot,
                    evidence,
                    metadata,
                ],
            )

        with gr.Tab("Evaluation"):
            gr.Markdown(
                "Evaluation results will be loaded from "
                "`artifacts/evaluation/summary.json`."
            )

        with gr.Tab("About"):
            gr.Markdown(
                "This system answers questions using SEC filing evidence."
            )

    return demo
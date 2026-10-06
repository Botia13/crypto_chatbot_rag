import gradio as gr

from RAG_model.api.factory import create_fastapi_app
from RAG_model.ui.gradio_app import create_gradio_app
from RAG_model.ui.presentation import RESEARCH_CSS, create_research_theme

api = create_fastapi_app()
ui = create_gradio_app()

app = gr.mount_gradio_app(
    api,
    ui,
    path="/",
    theme=create_research_theme(),
    css=RESEARCH_CSS,
)

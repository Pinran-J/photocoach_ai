import gradio as gr
from ui.gradio_app import create_gradio_app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()
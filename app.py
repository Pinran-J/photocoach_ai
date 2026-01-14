from ui.gradio_app import create_gradio_app
import gradio as gr

app = create_gradio_app()

app.queue().launch(
    theme=gr.themes.Soft(primary_hue="blue", radius_size="lg"),
)

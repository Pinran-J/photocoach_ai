from ui.gradio_app import create_gradio_app
import gradio as gr
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # suppress DecompressionBombWarning for large photos

app = create_gradio_app()

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
)

app.queue().launch(
    theme=theme,
)

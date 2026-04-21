import gradio as gr
from core.chat_interface import chat_interface

chat = chat_interface()


def create_gradio_app():
    with gr.Blocks(title="PhotoCoach AI") as demo:
        gr.Markdown(
            """
            # 📸 PhotoCoach AI
            **Agentic photography coaching** — composition, aesthetics, EXIF analysis & curated tips, no multimodal LLM required.
            """
        )
        gr.HTML("""
        <button onclick="
            document.body.classList.toggle('dark');
            this.textContent = document.body.classList.contains('dark') ? '☀️' : '🌙';
        " title="Toggle dark mode" style="
            position: fixed; top: 14px; right: 18px; z-index: 9999;
            width: 36px; height: 36px; border-radius: 8px;
            border: none; background: #2563eb; color: white;
            font-size: 16px; cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 1px 6px rgba(0,0,0,0.15);
        ">🌙</button>
        """)

        image_state = gr.State(value=None)

        with gr.Row():
            # ── Left column: tabs (upload+original | heatmap) + tips ──
            with gr.Column(scale=1):
                with gr.Tabs() as photo_tabs:
                    with gr.Tab("📷 Upload / Original", id="original"):
                        image_input = gr.Image(
                            type="filepath",
                            label="Upload photo",
                        )
                        image_status = gr.Markdown()
                    with gr.Tab("🌡️ Aesthetic Heatmap", id="heatmap"):
                        gradcam_output = gr.Image(
                            interactive=False,
                            show_label=False,
                        )

                gr.Markdown(
                    """
**Tips**
- Street / portrait / landscape photos work best
- JPG or PNG recommended
- EXIF data improves technical feedback
- Heatmap highlights regions driving the aesthetic score
                    """
                )

            # ── Right column: unified chat interface ──
            with gr.Column(scale=2):
                gr.ChatInterface(
                    fn=chat.async_rag_chat,
                    chatbot=gr.Chatbot(height=550, label="PhotoCoach AI"),
                    additional_inputs=[image_state],
                    additional_outputs=[gradcam_output, photo_tabs],
                )

        # On upload: store path, reset to Original tab, clear stale heatmap
        image_input.change(
            fn=chat.set_image,
            inputs=image_input,
            outputs=[image_state, image_status],
        ).then(
            fn=lambda: (gr.update(selected="original"), None),
            outputs=[photo_tabs, gradcam_output],
        )

    return demo

import gradio as gr
from core.chat_interface import chat_interface

chat = chat_interface()

def create_gradio_app():
    with gr.Blocks(title="PhotoCoach AI") as demo:
        gr.Markdown(
        """
        # 📸 PhotoCoach AI  
        **Multimodal photography critique powered by vision-enabled LLMs**
        """
    )
        gr.Markdown(
            "Upload a photo, then ask for feedback on composition, lighting, or aesthetics."
        )

        # 🔹 Shared state
        image_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="filepath",
                    label="Upload photo",
                )
                image_status = gr.Markdown()
                
                gr.Markdown(
                    """
                    **Tips**
                    - Street / portrait / landscape photos work best
                    - JPG or PNG recommended
                    - EXIF data improves technical feedback
                    """
                    )

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=720,
                    label="PhotoCoach AI"
                )
                gr.ChatInterface(
                    fn=chat.async_rag_chat,
                    additional_inputs=[image_state],
                    chatbot=chatbot,
                )
        # 🔹 Image upload → state update
        image_input.change(
            fn=chat.set_image,
            inputs=image_input,
            outputs=[image_state, image_status],
        )



    return demo

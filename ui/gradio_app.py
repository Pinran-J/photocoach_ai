import gradio as gr
from core.chat_interface import chat_interface

def create_gradio_app():
    chat_bot = chat_interface()
    demo = gr.ChatInterface(
    fn=chat_bot.rag_chat, 
    examples=[
        {"text": "No files", "files": []}
    ], 
    multimodal=True,
    textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image"], sources=["upload"])
    )
    
    return demo

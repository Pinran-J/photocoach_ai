from dotenv import load_dotenv
from tools.models_utils import score_aesthetic
from tools.exif_tool import fetch_exif
from tools.captioning_tool import caption_image
from rag.retriever_fetch_tool import retrieve_photography_tips
from agent.agent_state import ToolCalls
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from agent.graph import build_graph

load_dotenv()

class chat_interface:
    def __init__(self):
        tools = [
            retrieve_photography_tips,
            score_aesthetic,
            fetch_exif,
            caption_image,
        ]

        response_model = init_chat_model("gpt-5-nano", temperature=0)

        tool_decider_model = create_agent(
            model="gpt-5-nano",
            response_format=ToolCalls,
        )

        self.graph = build_graph(tool_decider_model, response_model)
        self.image = None

    # 🔹 NEW: image updater (NO LLM CALLS)
    def set_image(self, image_path):
        """
        Called when the user uploads an image.
        Just updates shared UI state.
        """
        if image_path is None:
            return None, "No image uploaded."

        return image_path, "✅ Image uploaded and ready"

    # 🔹 Chat-only logic
    def rag_chat(self, message, history, image_path):
        initial_state = {
            "user_query": message,
            "image_path": image_path,
            "messages": history,
        }

        print(initial_state)
        result = self.graph.invoke(initial_state) 
        return result["messages"][-1].content
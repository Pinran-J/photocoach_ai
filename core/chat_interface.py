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
        retrieve_photography_tips_tool = retrieve_photography_tips
        score_aesthetic_tool = score_aesthetic
        fetch_exif_tool = fetch_exif
        captioning_tool = caption_image
        tools = [retrieve_photography_tips_tool, score_aesthetic_tool, fetch_exif_tool, captioning_tool]
        
        response_model = init_chat_model("gpt-5-nano", temperature=0)

        tool_decider_model = create_agent(
            model="gpt-5-nano",
            tools=tools,
            response_format=ToolCalls
        )

        self.graph = build_graph(tool_decider_model, response_model)


    def rag_chat(self, message, history):
        print(message)
        initial_state = {
            "user_query": message["text"],
            "image_path": message["files"][0] if message["files"] else None,
            "messages": history,
        }
        print(initial_state)
        result = self.graph.invoke(initial_state) 
        return result["messages"][-1].content
                
        # any_yielded = False
        # for chunk in self.graph.stream(initial_state, stream_mode="values"):
        #     if "messages" in chunk and chunk["messages"]:
        #         any_yielded = True
        #         yield chunk["messages"][-1].content

        # # If nothing was yielded, yield a fallback
        # if not any_yielded:
        #     yield "Sorry, could not process your input."


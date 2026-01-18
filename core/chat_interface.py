from dotenv import load_dotenv
from tools.models_utils import score_aesthetic
from tools.exif_tool import fetch_exif
from tools.captioning_tool import caption_image
from rag.retriever_fetch_tool import retrieve_photography_tips
from agent.agent_state import ToolCalls
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from agent.graph import build_graph

import gradio as gr

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
    
    
    
    async def async_rag_chat(self, message, history, image_path):
        try:
            initial_state = {
                "user_query": message,
                "image_path": image_path,
                "messages": history,
            }
            print(initial_state)
            emitted_tools = set()
            emitted_tool_plan = False
            results = []
            async for stream_mode, chunk in self.graph.astream(initial_state, 
                                                               stream_mode=["values", "messages"]):
                print("test")
                print("history:", history)
                print("chunk:", chunk)
                if stream_mode == "values":
                    final_state = chunk
                    if "tool_plan" in final_state:
                        if any(final_state["tool_plan"].values()) and not emitted_tool_plan:
                            results.append(gr.ChatMessage(role="assistant", content=f"Plan: {final_state['tool_plan']}", metadata={"title": f"🛠️ Used tools"}))
                            emitted_tool_plan = True
                            yield results
                            
                        elif any(final_state["tool_plan"].values()) and emitted_tool_plan:
                            if "caption" in final_state and not "caption" in emitted_tools:
                                results.append(gr.ChatMessage(role="assistant", content=final_state["caption"], metadata={"title": f"🛠️ Used tool Captioner"}))
                                emitted_tools.add("caption")
                                yield results
                            if "aesthetic_score" in final_state and not "aesthetic_score" in emitted_tools:
                                results.append(gr.ChatMessage(role="assistant", content=f"{final_state['aesthetic_score']}", metadata={"title": f"🛠️ Used tool Aesthetic Scorer"} ))
                                emitted_tools.add("aesthetic_score")
                                yield results
                            if "exif" in final_state and not "exif" in emitted_tools:
                                results.append(gr.ChatMessage(role="assistant", content=f"{final_state['exif']}", metadata={"title": f"🛠️ Used tool EXIF Extractor"}))
                                emitted_tools.add("exif")
                                yield results
                            if "retrieved_docs" in final_state and not "retrieved_docs" in emitted_tools:
                                results.append(gr.ChatMessage(role="assistant", content="\n\n".join(final_state["retrieved_docs"])[:300], metadata={"title": f"🛠️ Used tool Photography Tips Retriever"}))
                                emitted_tools.add("retrieved_docs")
                                yield results
                elif stream_mode == "messages":
                    msg, metadata = chunk
                    print("output: ", msg, metadata)
                    if metadata['langgraph_node'] == "final" and msg.content:
                        results.append(gr.ChatMessage(role="assistant", content=msg.content))
                        yield results
                

        except Exception as e:
            user_error_message = "There was an error processing your request. Please try again."
            print(e)
            yield [gr.ChatMessage(role="assistant", content=user_error_message + str(e))]
            

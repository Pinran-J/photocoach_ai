from agent.agent_state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from tools.models_utils import score_aesthetic
from tools.exif_tool import fetch_exif
from tools.captioning_tool import caption_image
from rag.retriever_fetch_tool import retrieve_photography_tips

def planner_node(state: AgentState, tool_deciding_llm):
    """A simple planner node that decides which tools to call based on the user query."""
    prompt = f"""
    You are a photography coach designed to help users improve their photos. 
    Based on the user's query and the conversation history, decide which tools are required to assist the user, usually the EXIF data and captioning tools are very helpful.
    Do not need to call any tools if the user query is unrelated to photography or image critique. (Such as a simple greeting.)
    
    Conversation history:
    {state['messages']}

    User query:
    {state['user_query']}

    Image present:
    {'Yes' if state['image_path'] else 'No'}
    
    Available tools:
    - caption_image (describes the content of the image, call this if the user does not describe the image, REQUIRES an image)
    - aesthetic_score (provides an aesthetic quality score for the image, call this if the user asks about image quality, REQUIRES an image)
    - extract_exif (extracts EXIF metadata from the image, call this if the user asks about camera settings, REQUIRES an image)
    - retrieve_photography_tips (provides photography tips based on the user's query, call this for general photography questions)

    Return a JSON object with boolean values for each tool ONLY, indicating whether to call it or not. Usually it is better to call more tools to gather more information about the image.
    """
    sys_msg = SystemMessage(content=prompt)
    human_msg = HumanMessage(content=state["user_query"])

    response = tool_deciding_llm.invoke({"messages": [sys_msg] + [human_msg]})
    print("STATE:", response)
    print("PLANNER RESPONSE:", response["structured_response"])
    return {"tool_plan": response["structured_response"]} # The old state is compounded on top.

def route_after_planner(state: AgentState):
    print(state)
    if not any(state["tool_plan"].values()):
        return "final"
    return "tool_executor"


def tool_node(state: AgentState):
    """Perform tool calls based on the planner's decision."""
    plan = state["tool_plan"]
    updates = {}
    print("TOOL PLAN:", plan)
    print(f"state {state}")

    if plan["caption_image"] and state["image_path"]:
        updates["caption"] = caption_image.invoke({"image_path": state["image_path"]})

    if plan["aesthetic_score"] and state["image_path"]:
        updates["aesthetic_dist"], updates["aesthetic_score"] = score_aesthetic.invoke({"image_path": state["image_path"]})

    if plan["extract_exif"] and state["image_path"]:
        updates["exif"] = fetch_exif.invoke({"image_path": state["image_path"]})

    if plan["retrieve_photography_tips"]:
        updates["retrieved_docs"] = retrieve_photography_tips.invoke({"query": state["user_query"]})

    return updates

def final_answer_node(state: AgentState, llm):
    prompt = f"""
    You are a professional photography coach.

    User question:
    {state["user_query"]}

    Image caption:
    {state.get("caption")}
    
    Image present:
    {'Yes' if state['image_path'] else 'No'}

    Aesthetic score:
    {state.get("aesthetic_score")}
    
    Aesthetic distribution:
    {state.get("aesthetic_dist")}

    EXIF data:
    {state.get("exif")}

    Photography knowledge:
    {state.get("retrieved_docs")}

    Give clear, practical and specific advice based on the following information fetched about the image's aesthetic quality, EXIF data, and caption.
    Answer normally if no tools were called and non-image related question was asked.
    """
    print(prompt)
    response = llm.invoke(prompt)
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }
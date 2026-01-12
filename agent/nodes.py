from .agent_state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from tools.models_utils import score_aesthetic
from tools.exif_tool import fetch_exif
from tools.captioning_tool import caption_image
from rag.retriever_fetch_tool import retrieve_photography_tips

def planner_node(state: AgentState, tool_deciding_llm):
    """A simple planner node that decides which tools to call based on the user query."""
    prompt = f"""
    You are a photography coach. Based on the user's query and the conversation history, decide which tools are required to assist the user.
    
    Conversation history:
    {state['messages']}

    User query:
    {state['user_query']}

    Available tools:
    - caption_image
    - aesthetic_score
    - extract_exif
    - retrieve_photography_tips

    Decide which tools to call. Return a JSON object with boolean values for each tool, indicating whether to call it or not.
    """
    sys_msg = SystemMessage(content=prompt)
    human_msg = HumanMessage(content=state["user_query"])
    
    response = tool_deciding_llm.invoke([sys_msg] + [human_msg])
    return {"tool_calls": response["structured_response"]} # The old state is compounded on top.

def route_after_planner(state):
    if not any(state["tool_plan"].values()):
        return END
    return "tool_executor"


def tool_node(state: AgentState):
    """Perform tool calls based on the planner's decision."""
    plan = state["tool_plan"]
    updates = {}

    if plan["caption_image"]:
        updates["caption"] = caption_image(state["image_path"])

    if plan["aesthetic_score"]:
        updates["aesthetic_dist"], updates["aesthetic_score"] = score_aesthetic(state["image_path"])

    if plan["extract_exif"]:
        updates["exif"] = fetch_exif(state["image_path"])

    if plan["retrieve_photography_tips"]:
        updates["retrieved_docs"] = retrieve_photography_tips(state["user_query"])

    return updates

def final_answer_node(state: AgentState, llm):
    prompt = f"""
    You are a professional photography coach.

    User question:
    {state["user_query"]}

    Image caption:
    {state.get("caption")}

    Aesthetic score:
    {state.get("aesthetic_score")}
    
    Aesthetic distribution:
    {state.get("aesthetic_dist")}

    EXIF data:
    {state.get("exif")}

    Photography knowledge:
    {state.get("retrieved_docs")}

    Give clear, practical advice.
    """

    response = llm.invoke(prompt)
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }
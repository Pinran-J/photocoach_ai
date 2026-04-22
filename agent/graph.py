from langgraph.graph import StateGraph, START, END
from agent.agent_state import AgentState
from agent.nodes import (
    planner_node, route_after_planner,
    tool_node,
    reflect_node, route_after_reflect,
    final_answer_node,
)

def build_graph(tool_decider_model, response_model, reflect_decider_model):
    graph = StateGraph(AgentState)

    graph.add_node("planner",      lambda s: planner_node(s, tool_decider_model))
    graph.add_node("tool_executor", tool_node)
    graph.add_node("reflect",      lambda s: reflect_node(s, reflect_decider_model))
    async def final_node(s): return await final_answer_node(s, response_model)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")

    graph.add_conditional_edges("planner", route_after_planner, ["tool_executor", "final"])
    graph.add_edge("tool_executor", "reflect")
    graph.add_conditional_edges("reflect", route_after_reflect, ["tool_executor", "final"])
    graph.add_edge("final", END)

    return graph.compile()

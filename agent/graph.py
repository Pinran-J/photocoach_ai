# graph.py
from langgraph.graph import StateGraph, START, END
from agent.agent_state import AgentState
from agent.nodes import planner_node, route_after_planner, tool_node, final_answer_node

def build_graph(planner_llm, final_llm):
    graph = StateGraph(AgentState)

    graph.add_node("planner", lambda s: planner_node(s, planner_llm))
    graph.add_node("tool_executor", tool_node)
    graph.add_node("final", lambda s: final_answer_node(s, final_llm))

    graph.add_edge(START, "planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        ["tool_executor", END]
    )

    graph.add_edge("tool_executor", "final")
    graph.add_edge("final", END)

    return graph.compile()

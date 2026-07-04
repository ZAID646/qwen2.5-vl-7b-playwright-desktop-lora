from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.nodes import ActionNode, PerceptionNode, RouterNode
from src.agent.state import AgentState
from src.memory.context import ContextCompressor
from src.vision.mock import MockVLM


def build_agent(vlm: MockVLM | None = None) -> StateGraph:
    compressor = ContextCompressor()
    perception = PerceptionNode(vlm or MockVLM(), compressor)
    action = ActionNode()
    router = RouterNode()

    graph = StateGraph(AgentState)

    graph.add_node("perception", perception)
    graph.add_node("action", action)

    graph.set_entry_point("perception")

    graph.add_edge("perception", "action")
    graph.add_conditional_edges(
        "action",
        router,
        {"continue": "perception", "done": END, "error": END},
    )

    return graph.compile()

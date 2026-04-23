from typing import List, Optional, Dict
from langgraph.graph import MessagesState
from typing_extensions import TypedDict

class ToolCalls(TypedDict):
    """Bool indicating which tools to call."""
    caption_image: bool
    aesthetic_score: bool
    extract_exif: bool
    retrieve_photography_tips: bool

class ReflectDecision(TypedDict):
    """Reflect node output — missing tools + optional rephrased RAG query."""
    caption_image: bool
    aesthetic_score: bool
    extract_exif: bool
    retrieve_photography_tips: bool
    rephrased_query: str  # non-empty → re-call RAG with this query; empty → use original

class AgentState(MessagesState):
    """State for custom RAG agent with photography tools."""

    # Inputs
    image_path: str = ""
    user_query: str = ""

    # Planner output
    tool_plan: ToolCalls = ToolCalls(
        caption_image=False,
        aesthetic_score=False,
        extract_exif=False,
        retrieve_photography_tips=False
    )

    # Tool outputs
    caption: str = ""
    exif: Dict = {}
    aesthetic_score: float = 0.0
    aesthetic_dist: List[float] = []
    gradcam_path: Optional[str] = None

    # RAG
    retrieved_docs: List[str] = []
    retrieval_query: str = ""   # overrides user_query for RAG when reflect rephrases

    # Loop control
    iterations: int = 0

    # Output
    final_response: str = ""

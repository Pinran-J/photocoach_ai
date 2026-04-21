from typing import List, Optional, Dict
from langgraph.graph import MessagesState
from typing_extensions import TypedDict

class ToolCalls(TypedDict):
    """Bool indicating which tools to call."""
    caption_image: bool # To call the caption_image tool
    aesthetic_score: bool # To call the aesthetic_score tool 
    extract_exif: bool # To call the extract_exif tool 
    retrieve_photography_tips: bool # To call the retrieve_photography_tips tool

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

    # Output
    final_response: str = ""
    
    
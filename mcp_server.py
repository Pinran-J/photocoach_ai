"""
PhotoCoach MCP Server
Exposes two tools over the Model Context Protocol so any MCP-compatible
client (e.g. Claude Desktop) can call them without running the full app.

──────────────────────────────────────────────────────────────────────────────
HOW IT WORKS
──────────────────────────────────────────────────────────────────────────────
Your model weights and Pinecone key stay on THIS server — users never see them.
They just send requests through the MCP protocol and get results back.

  User's Claude Desktop
          │
          │  MCP protocol (stdio locally / HTTPS when deployed)
          ▼
  This MCP server  ←── ResNet50 weights loaded in memory
          │         ←── PINECONE_API_KEY in environment
          │
          ├── score_aesthetic()      → runs inference locally
          └── retrieve_photography_tips() → queries your Pinecone index

──────────────────────────────────────────────────────────────────────────────
RUNNING LOCALLY (Claude Desktop – stdio transport)
──────────────────────────────────────────────────────────────────────────────
Add to ~/Library/Application Support/Claude/claude_desktop_config.json:

    {
      "mcpServers": {
        "photocoach": {
          "command": "python",
          "args": ["/absolute/path/to/mcp_server.py", "--stdio"]
        }
      }
    }

──────────────────────────────────────────────────────────────────────────────
RUNNING DEPLOYED (HTTP transport — others connect to your server URL)
──────────────────────────────────────────────────────────────────────────────
Start the server:
    python mcp_server.py          # defaults to HTTP on port 8000
    python mcp_server.py --stdio  # stdio mode for local Claude Desktop

Others add to their Claude Desktop config:
    {
      "mcpServers": {
        "photocoach": {
          "url": "https://your-deployed-server.com/mcp"
        }
      }
    }
"""

import io
import sys
import logging
import requests
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Reuse the already-loaded model and transform from the existing tools module
# so we don't duplicate weight-loading logic.
from tools.models_utils import MODEL, eval_transform

# Reuse the already-initialised Pinecone retriever.
from rag.retriever_fetch_tool import retriever

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("PhotoCoach Tools", host="0.0.0.0", port=8000)


@mcp.tool()
def score_aesthetic(image_url: str) -> dict:
    """Score the aesthetic quality of a photograph on a 1–10 scale.

    Uses a fine-tuned ResNet50 trained on photography datasets.
    Pass a publicly accessible image URL (HTTPS).

    Returns:
        mean_score    – float, expected score 1–10
        peak_score    – int, the single most likely score
        distribution  – list of 10 probabilities (index 0 = score 1)
        interpretation – human-readable quality tier
    """
    logger.info("score_aesthetic called for: %s", image_url)

    response = requests.get(image_url, timeout=15)
    response.raise_for_status()

    pil_img = Image.open(io.BytesIO(response.content)).convert("RGB")
    x = eval_transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    scores = np.arange(1, 11)
    mean_score = float((probs * scores).sum())
    peak_score = int(probs.argmax()) + 1

    if mean_score >= 7.5:
        interpretation = "Strong — professional-quality composition and exposure."
    elif mean_score >= 5.5:
        interpretation = "Good — decent shot with room for targeted improvements."
    elif mean_score >= 3.5:
        interpretation = "Developing — noticeable technical or compositional issues."
    else:
        interpretation = "Needs work — significant issues with exposure, focus, or composition."

    return {
        "mean_score": round(mean_score, 2),
        "peak_score": peak_score,
        "distribution": [round(float(p), 4) for p in probs],
        "interpretation": interpretation,
    }


@mcp.tool()
def retrieve_photography_tips(query: str) -> str:
    """Retrieve photography tips and techniques from the PhotoCoach knowledge base.

    Searches a curated vector index of photography books and guides
    using MMR retrieval to return diverse, relevant passages.

    Args:
        query: Natural language question or topic.
               Examples: "how to reduce noise", "portrait lighting setup",
               "rule of thirds", "street photography settings"

    Returns:
        Relevant passages from the knowledge base separated by '---'.
    """
    logger.info("retrieve_photography_tips called: %s", query)
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    transport = "stdio" if "--stdio" in sys.argv else "streamable-http"
    logger.info("Starting PhotoCoach MCP server (transport=%s)", transport)
    mcp.run(transport=transport)

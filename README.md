---
title: PhotoCoach AI
emoji: 📸
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# 📸 PhotoCoach AI

## Agentic Photography Coaching System (Tool-Orchestrated, Lightweight LLMs)

### PhotoCoach AI is an agentic RAG-based chatbot that provides structured photography feedback by orchestrating multiple specialized tools, without relying on multimodal LLMs.

The system demonstrates **agentic planning, state-driven execution, and real-time streaming explainability**, making intermediate decisions and tool usage visible to users.

<!-- TABLE OF CONTENTS -->
<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#-features">Features</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#-system-architecture">System Architecture</a></li>
    <li><a href="#-rag-pipeline--knowledge-base">RAG Pipeline & Knowledge Base</a></li>
    <li><a href="#-retrieval-evaluation">Retrieval Evaluation</a></li>
    <li><a href="#-mcp-server">MCP Server</a></li>
    <li><a href="#-deployment">Deployment</a></li>
    <li><a href="#-project-structure">Project Structure</a></li>
    <li><a href="#-demo">Demo</a></li>
    <li><a href="#%E2%80%8D-author">Author</a></li>
  </ol>
</details>

## About The Project

PhotoCoach AI is an **agent-based AI system** designed to analyze photographs and deliver actionable feedback on composition, aesthetics, and technical quality.

Instead of using large multimodal LLMs, the system relies on **lightweight language models combined with explicit tool execution**, allowing fine-grained control, transparency, and debuggability.

The agent dynamically plans which tools to invoke based on the user query and image context, then synthesizes their outputs into a coherent response.

---

### 🚀 Features

- 🧠 **Agentic Tool Planning**
  - Explicit planning step determines which tools to invoke per query
  - Decisions are exposed in the UI for explainability

- 📷 **Image Understanding via Tools**
  - Image captioning (BLIP)
  - Aesthetic scoring using a fine-tuned ResNet50 CNN
  - EXIF metadata extraction for technical camera settings analysis

- 📚 **Retrieval-Augmented Generation (RAG)**
  - MMR retrieval over a curated, continuously updated knowledge base
  - Sources: photography books (PDFs), Wikipedia reference articles, 8 live RSS feeds
  - ETL pipeline runs on AWS Lambda (EventBridge schedule) to ingest fresh articles weekly

- 🔄 **Real-Time Streaming Outputs**
  - Intermediate tool usage and final responses streamed to the chat UI

- 🧩 **State-Driven Workflow**
  - Built with **LangGraph** for explicit state transitions and agent control

- 🔌 **MCP Server**
  - Core tools exposed as an MCP (Model Context Protocol) server
  - Any MCP-compatible client can call `score_aesthetic` and `retrieve_photography_tips` directly

---

### Built With

- **LangGraph** – agent orchestration & state management
- **LangChain** – tool abstraction and retrieval
- **Gradio** – streaming chat interface
- **PyTorch** – ResNet50 aesthetic scoring model
- **Pinecone** – vector database (MMR retrieval)
- **OpenAI** – embeddings (`text-embedding-3-small`) and response generation (`gpt-5-nano`)
- **transformers** – BLIP image captioning
- **AWS Lambda + EventBridge** – scheduled ETL pipeline
- **Docker + Kubernetes** – containerised deployment


## 🧠 System Architecture

The system follows an explicit agent workflow:

1. **Planner Node**
   - Interprets user query and available context
   - Selects which tools to invoke (if any)

2. **Tool Nodes** (executed in parallel via `asyncio.gather`)
   - Image captioning (BLIP)
   - Aesthetic scoring (ResNet50)
   - EXIF metadata extraction
   - Photography knowledge retrieval (RAG)

3. **Final Reasoning Node**
   - Aggregates tool outputs into structured feedback
   - Streams intermediate and final messages to the UI

4. **Agentic flow graph**:

![alt text](flowchart.png)


## 📚 RAG Pipeline & Knowledge Base

The knowledge base is built from three source types and kept fresh by a scheduled ETL pipeline:

| Source | Content | Update frequency |
|---|---|---|
| Photography books (PDF) | Expert composition, lighting, and technique | Static |
| Wikipedia articles (20 pages) | Foundational concepts (exposure, aperture, depth of field, genres) | Static |
| RSS feeds (6 publications) | New tutorials and guides from PetaPixel, Photography Life, Fstoppers, Shotkit, and others | Weekly (AWS Lambda) |

**Retrieval:** MMR (Maximal Marginal Relevance) fetches 20 candidates and returns the 5 most relevant and diverse chunks, reducing redundancy from repeated sources.

**ETL:** The Lambda function runs weekly via EventBridge, extracts new RSS articles, chunks and embeds them, and upserts into Pinecone via the REST API.


## 📊 Retrieval Evaluation

Label-free evaluation comparing the original index (PDF-only, similarity search) against the current index (PDF + Wikipedia + RSS, MMR retrieval) using an LLM-as-judge pattern across 20 photography test queries:

| Metric | Old Index | New Index | Change |
|---|---|---|---|
| LLM Relevance Score (1–5) | 3.0 | 3.0 | — |
| Avg Cosine Similarity | 0.576 | 0.599 | +4.0% |
| Diversity Score (0–1) | 0.305 | 0.427 | **+40.0%** |

The +40% diversity improvement reflects MMR's ability to surface a wider range of relevant techniques per query rather than returning near-duplicate chunks from the same source.


## 🔌 MCP Server

PhotoCoach exposes its core tools as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server, allowing any compatible client to use them independently of the Gradio app.

**Exposed tools:**
- `score_aesthetic` – runs ResNet50 inference on a public image URL, returns a 1–10 score with distribution
- `retrieve_photography_tips` – MMR retrieval over the Pinecone knowledge base

**Run locally:**
```bash
docker compose up photocoach-mcp
# Server available at http://localhost:8000/mcp
python test_mcp_client.py
```

**Connect from Claude Desktop:**
```json
{
  "mcpServers": {
    "photocoach": {
      "url": "http://15.134.224.148:8000/mcp"
    }
  }
}
```


## 🚀 Deployment

The app is fully containerised. Both services share a model weight volume to avoid downloading ResNet50 twice.

**Run locally:**
```bash
cp .env.example .env   # add your OPENAI_API_KEY and PINECONE_API_KEY
docker compose up --build
# Gradio app → http://localhost:7860
# MCP server → http://localhost:8000/mcp
```

**Kubernetes (AWS EKS):** K8s manifests are provided under `k8s/` with Deployments, Services, HPA (auto-scaling on CPU), and namespace/secret configuration.


## 📁 Project Structure
```
photocoach_ai/
├── agent/
│   ├── agent_state.py           # Shared agent state (messages, tool plans, outputs)
│   ├── graph.py                 # LangGraph agent workflow definition
│   └── nodes.py                 # Planner, tool execution, and final reasoning nodes
│
├── core/
│   └── chat_interface.py        # Gradio-facing chat interface & streaming logic
│
├── rag/
│   ├── etl/
│   │   ├── extractors.py        # PDF, Wikipedia, and RSS extractors
│   │   ├── pipeline.py          # Chunking, embedding, and Pinecone upsert
│   │   └── load.py              # Pinecone REST upsert (Lambda-compatible)
│   ├── ingestion_old.py         # Legacy ingestion script
│   └── retriever_fetch_tool.py  # MMR retrieval tool
│
├── tools/
│   ├── captioning_tool.py       # BLIP image captioning
│   ├── exif_tool.py             # EXIF metadata extraction
│   └── models_utils.py          # ResNet50 aesthetic scoring
│
├── models/
│   └── aesthetic_resnet.py      # ResNet50 model definition
│
├── k8s/                         # Kubernetes manifests (Deployment, HPA, Service)
│
├── mcp_server.py                # MCP server (streamable-HTTP transport)
├── test_mcp_client.py           # MCP client for local testing
├── lambda_handler.py            # AWS Lambda entry point for ETL
├── requirements.txt             # Full app dependencies
├── requirements-etl.txt         # Minimal Lambda dependencies
├── Dockerfile                   # Gradio app image
├── Dockerfile.mcp               # MCP server image
├── docker-compose.yml           # Local dev stack
├── app.py                       # Application entry point
└── README.md
```


## 📸 Demo

![alt text](demo.gif)

A short demo showing:
- Agent planning decisions
- Tool execution steps (captioning, aesthetic scoring, EXIF, RAG retrieval)
- Final synthesised feedback streamed to the UI

---

## 🧑‍💻 Author

**Jiang Pinran**  
Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/in/jiangpinran/) | [GitHub](https://github.com/Pinran-J)

---

## 🪄 License

MIT License © 2025 Jiang Pinran

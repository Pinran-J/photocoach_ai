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
        <li><a href="#🚀-features">Features</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#🧠-system-architecture">System Architecture</a></li>
    <li><a href="#🧠-system-architecture">System Architecture</a></li>
    <li><a href="#📸-demo">Demo</a></li>
    <li><a href="#📦-future-work">Future Work</a></li>
    <li><a href="#🧑‍💻-author">Author</a></li>
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
  - Image captioning
  - Aesthetic scoring using a CNN-based model
  - EXIF metadata extraction for technical analysis

- 📚 **Retrieval-Augmented Generation (RAG)**
  - Retrieves grounded photography advice from curated knowledge sources

- 🔄 **Real-Time Streaming Outputs**
  - Intermediate tool usage and final responses streamed to the chat UI

- 🧩 **State-Driven Workflow**
  - Built with **LangGraph** for explicit state transitions and agent control

---

### Built With

- **LangGraph** – agent orchestration & state management  
- **LangChain** – tool abstraction and retrieval  
- **Gradio** – streaming chat interface  
- **PyTorch** – aesthetic scoring model  
- **Pinecone** - vector database
- **Numpy** - support for large multi-dimensional arrays/matrices
- **transformers** - BLIP framework for image comprehension


## 🧠 System Architecture

The system follows an explicit agent workflow:

1. **Planner Node**
   - Interprets user query and available context
   - Selects which tools to invoke (if any)

2. **Tool Nodes**
   - Image captioning
   - Aesthetic scoring
   - EXIF metadata extraction
   - Photography knowledge retrieval (RAG)

3. **Final Reasoning Node**
   - Aggregates tool outputs into structured feedback
   - Streams intermediate and final messages to the UI

4. **Agentic flow graph**:

- ![alt text](flowchart.png)

## 📁 Project Structure
```
photocoach_ai/
├── agent/
│   ├── agent_state.py        # Defines shared agent state (messages, tool plans, outputs)
│   ├── graph.py              # LangGraph agent workflow definition
│   ├── nodes.py              # Planner, tool execution, and final reasoning nodes
│
├── core/
│   └── chat_interface.py     # Gradio-facing chat interface & streaming logic
│
├── rag/
│   ├── ingestion.py          # Knowledge ingestion and preprocessing
│   └── retriever_fetch_tool.py # RAG retrieval tool for photography advice
│
├── tools/
│   ├── captioning_tool.py    # Image captioning tool
│   ├── exif_tool.py          # EXIF metadata extraction tool
│   └── models_utils.py       # Aesthetic scoring utilities
│
├── ui/
│   └── gradio_app.py         # Gradio UI layout and bindings
│
├── data/                     # Optional datasets / knowledge sources
├── models/                   # Trained or downloaded model artifacts
│
├── app.py                    # Application entry point
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── test.ipynb                # Development / debugging notebook
└── output.png                # Flow chart for agentic flow
```


## 📸 Demo

![alt text](demo.gif)


A short demo video and screenshots showing:
- Agent planning decisions
- Tool execution steps
- Final synthesized feedback

---

## 📦 Future Work

- Improve aesthetic scoring accuracy
- Add additional photography-specific tools
- Explore multi-agent extensions (e.g. composition vs lighting specialists)
- Deploy as a hosted service

---

## 🧑‍💻 Author

**Jiang Pinran**  
Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/in/jiangpinran/) | [GitHub](https://github.com/Pinran-J)

---

## 🪄 License

MIT License © 2025 Jiang Pinran



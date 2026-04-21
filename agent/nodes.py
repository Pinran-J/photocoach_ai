import asyncio
from agent.agent_state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools.models_utils import score_aesthetic, generate_gradcam
from tools.exif_tool import fetch_exif
from tools.captioning_tool import caption_image
from rag.retriever_fetch_tool import retrieve_photography_tips


def planner_node(state: AgentState, tool_deciding_llm):
    """Decide which tools to call based on the user query."""
    prompt = f"""
    You are a photography coach. Your job is to decide which tools to call — only call what is strictly necessary for the user's request. Do NOT call extra tools out of caution.

    Conversation history:
    {state['messages']}

    User query: {state['user_query']}
    Image present: {'Yes' if state['image_path'] else 'No'}

    Tool rules (follow exactly):
    - caption_image: call whenever an image is present AND the query involves the image content (feedback, score, composition, improvement, tips). The LLM is NOT multimodal — it cannot see the image without a caption. Skip only for pure EXIF/settings queries or text-only questions with no image.
    - aesthetic_score: call when the user asks about photo quality, score, how good it looks, or wants full/aesthetic feedback. Requires an image.
    - extract_exif: call when the user asks about camera settings, shutter speed, ISO, aperture, focal length, or technical metadata. Requires an image.
    - retrieve_photography_tips: call when the query involves ANY of: improving the photo, composition, lighting, technique, tips, advice, or coaching — whether or not an image is present. This includes "improve the composition", "how to fix the lighting", "what could be better", "score and improve my photo". Do NOT call for pure EXIF lookups or a bare "score my photo" with no improvement component.

    Examples:
    "Score my photo" → aesthetic_score=true, caption_image=true, extract_exif=false, retrieve_photography_tips=false
    "Improve the composition and give me the aesthetic score" → aesthetic_score=true, caption_image=true, retrieve_photography_tips=true, extract_exif=false
    "How can I improve this photo?" → caption_image=true, retrieve_photography_tips=true, aesthetic_score=true, extract_exif=false
    "What camera settings were used?" → extract_exif=true, caption_image=false, aesthetic_score=false, retrieve_photography_tips=false
    "How can I improve bokeh?" (no image) → retrieve_photography_tips=true, others=false
    "Give me full feedback on this photo" → all=true (image present)
    "Hello" or unrelated message → all=false

    Return a JSON object with boolean values for each tool. Nothing else.
    """
    sys_msg = SystemMessage(content=prompt)
    human_msg = HumanMessage(content=state["user_query"])

    # with_structured_output returns the TypedDict directly (not {"structured_response": ...})
    response = tool_deciding_llm.invoke([sys_msg, human_msg])
    return {"tool_plan": response}


def route_after_planner(state: AgentState):
    if not any(state["tool_plan"].values()):
        return "final"
    return "tool_executor"


async def tool_node(state: AgentState):
    """Execute all selected tools in parallel using asyncio.gather."""
    plan = state["tool_plan"]

    # Build coroutine dict — only schedule tools the planner selected
    coros = {}
    if plan.get("caption_image") and state.get("image_path"):
        coros["caption"] = caption_image.ainvoke({"image_path": state["image_path"]})
    if plan.get("aesthetic_score") and state.get("image_path"):
        coros["aesthetic"] = score_aesthetic.ainvoke({"image_path": state["image_path"]})
        coros["gradcam"] = asyncio.to_thread(generate_gradcam, state["image_path"])
    if plan.get("extract_exif") and state.get("image_path"):
        coros["exif"] = fetch_exif.ainvoke({"image_path": state["image_path"]})
    if plan.get("retrieve_photography_tips"):
        coros["rag"] = retrieve_photography_tips.ainvoke({"query": state["user_query"]})

    if not coros:
        return {}

    keys = list(coros.keys())
    outputs = await asyncio.gather(*coros.values(), return_exceptions=True)

    updates = {}
    for key, output in zip(keys, outputs):
        if isinstance(output, Exception):
            print(f"[tool_node] Tool '{key}' failed: {output}")
            continue
        if key == "aesthetic":
            updates["aesthetic_dist"], updates["aesthetic_score"] = output
        elif key == "gradcam":
            updates["gradcam_path"] = output
        elif key == "rag":
            updates["retrieved_docs"] = output
        else:
            updates[key] = output

    return updates


async def final_answer_node(state: AgentState, llm):
    score = state.get("aesthetic_score")
    dist = state.get("aesthetic_dist")

    # Build a human-readable aesthetic block that gives the LLM enough context to interpret the score
    if score is not None and dist:
        peak_score = dist.index(max(dist)) + 1
        score_block = (
            f"Mean aesthetic score: {score:.1f}/10  |  Model's peak prediction: {peak_score}/10\n"
            f"Score distribution (probability per score 1–10): {[round(p, 3) for p in dist]}\n\n"
            f"Scoring guide: 1–4 = significant technical or compositional issues; "
            f"5–6 = decent shot with clear room for improvement; "
            f"7–8 = strong composition/exposure; 9–10 = near-professional quality."
        )
    else:
        score_block = "Not evaluated."

    prompt = f"""
You are a professional photography coach. Give clear, practical, specific feedback tailored to this image.

User question:
{state["user_query"]}

Image present: {'Yes' if state['image_path'] else 'No'}

Image caption:
{state.get("caption") or "Not available."}

Aesthetic quality assessment:
{score_block}

EXIF / camera settings:
{state.get("exif") or "Not available."}

Relevant photography knowledge:
{state.get("retrieved_docs") or "Not available."}

Instructions:
- If the aesthetic score was evaluated, explicitly explain what specific elements are likely driving the score up or down, linking it to evidence in the caption and EXIF data (e.g. "your ISO 6400 at f/8 likely introduced noise, which the model penalises").
- Suggest 2–3 concrete, actionable improvements the photographer can make.
- If no image was provided and no tools were called, respond naturally to the user's question.
- Be concise. Avoid generic advice — be specific to what you can observe from the data above.
"""
    response = await llm.ainvoke(prompt)
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }

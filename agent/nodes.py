import asyncio
from agent.agent_state import AgentState, ToolCalls
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
    - caption_image: call when the coach needs to know what is in the image to give relevant advice. The LLM is NOT multimodal — it cannot see the image without a caption. Skip when the question is purely about reading technical metadata (e.g. "what camera settings were used?" needs only EXIF, not a caption).
    - aesthetic_score: call when the user wants to know how their photo performs overall — both when they explicitly ask for a score and when they ask for general improvement or full feedback where quality context would ground the advice. Skip for narrow technique-only questions (e.g. "how do I improve just the composition") where a quality rating adds no value.
    - extract_exif: call when knowing how the photo was technically captured would directly help answer the question — any question about the technical result of the shot (exposure, brightness, whether it looks over/underexposed, camera settings). Requires an image.
    - retrieve_photography_tips: call when the user needs coaching knowledge or technique guidance to improve their photography — whether general or image-specific. Skip for pure technical metadata lookups or bare score requests with no improvement intent.

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
    response = tool_deciding_llm.invoke([sys_msg, human_msg])
    return {"tool_plan": response}


def route_after_planner(state: AgentState):
    if not any(state["tool_plan"].values()):
        return "final"
    return "tool_executor"


MAX_ITERATIONS = 2  # reflect loops after the initial plan (3 total passes at most)


def reflect_node(state: AgentState, reflect_deciding_llm):
    """
    Check if gathered information is sufficient to answer the user's question.
    Can request missing tools AND rephrase the RAG query if retrieved tips were irrelevant.
    """

    # Hard stop — never loop more than MAX_ITERATIONS times
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return {
            "tool_plan": ToolCalls(
                caption_image=False, aesthetic_score=False,
                extract_exif=False, retrieve_photography_tips=False,
            ),
            "iterations": state.get("iterations", 0) + 1,
        }

    # What has already been successfully gathered
    has_caption   = bool(state.get("caption"))
    has_exif      = bool(state.get("exif"))
    has_aesthetic = bool(state.get("aesthetic_dist"))  # more reliable than score (defaults 0.0)
    has_tips      = bool(state.get("retrieved_docs"))

    gathered = [label for flag, label in [
        (has_caption,   "image caption"),
        (has_exif,      "EXIF / camera settings"),
        (has_aesthetic, "aesthetic score"),
        (has_tips,      "photography tips"),
    ] if flag]

    tips_preview = ""
    if has_tips:
        tips_preview = "\n\n".join(state["retrieved_docs"])[:600]

    prompt = f"""
    You are reviewing whether enough information has been gathered to fully answer the user's question.

    User question: {state['user_query']}
    Image present: {'Yes' if state['image_path'] else 'No'}
    Already gathered: {', '.join(gathered) if gathered else 'nothing yet'}

    Image caption (if available): {state.get("caption") or "None"}
    EXIF / camera settings (if available): {state.get("exif") or "None"}

    Retrieved photography tips so far:
    {tips_preview if tips_preview else 'None'}

    Decide:
    1. Which tools are missing and would meaningfully improve the answer?
       - caption_image: the image content is unknown but needed to give relevant advice
       - aesthetic_score: the user wants to know how their photo performs and it hasn't been evaluated
       - extract_exif: the user's question relates to technical capture (exposure, settings, brightness) — EXIF gives the actual values the camera used, which generic tips cannot substitute for. Call this if the question is about the technical result of the shot and EXIF hasn't been fetched.
       - retrieve_photography_tips: technique or coaching knowledge is needed but not yet retrieved, OR retrieved tips are on a completely different topic from the question

    2. rephrased_query: ONLY provide this if retrieve_photography_tips is True AND the existing
       tips are on a completely different topic from the user's question (e.g. user asked about
       lighting but tips are about composition). Leave it EMPTY if tips are relevant — even if
       they are general, general tips are still useful and re-fetching will return the same results.
       When you do provide a rephrased_query:
       - 5–8 keywords only, no sentences or instructions.
       - Good: "portrait bokeh shallow depth of field"
       - Bad: "Tips for portraits of a man and baby: apply rule of thirds, shallow depth of field..."
       - Use caption clues: people → "portrait", outdoors → "landscape", close-up → "macro"
       - Use EXIF clues: high ISO → "low light noise reduction", wide aperture → "bokeh"

    Rules:
    - Never return True for caption_image, aesthetic_score, or extract_exif if already gathered.
    - retrieve_photography_tips may be True even if tips were already retrieved, ONLY if the
      existing tips are on a completely wrong topic and a short rephrased_query is provided.
      Do NOT re-retrieve just because tips are general — general coaching tips are still useful.
    - If what has been gathered is sufficient, return all False and empty rephrased_query.
    """

    sys_msg = SystemMessage(content=prompt)
    human_msg = HumanMessage(content=state["user_query"])
    response = reflect_deciding_llm.invoke([sys_msg, human_msg])

    rephrased = response.get("rephrased_query", "").strip()

    # Enforce no re-calls except RAG with a rephrased query
    new_plan = ToolCalls(
        caption_image=response["caption_image"]                         and not has_caption,
        aesthetic_score=response["aesthetic_score"]                     and not has_aesthetic,
        extract_exif=response["extract_exif"]                           and not has_exif,
        retrieve_photography_tips=(
            response["retrieve_photography_tips"] and (not has_tips or bool(rephrased))
        ),
    )

    updates = {
        "tool_plan": new_plan,
        "iterations": state.get("iterations", 0) + 1,
    }
    # Only update retrieval_query if we're actually re-calling RAG with a new query
    if new_plan["retrieve_photography_tips"] and rephrased:
        updates["retrieval_query"] = rephrased

    return updates


def route_after_reflect(state: AgentState):
    if any(state["tool_plan"].values()):
        return "tool_executor"
    return "final"


async def tool_node(state: AgentState):
    """Execute all selected tools in parallel using asyncio.gather."""
    plan = state["tool_plan"]

    # Use rephrased_query for RAG if the reflect node provided one
    rag_query = state.get("retrieval_query") or state["user_query"]

    coros = {}
    if plan.get("caption_image") and state.get("image_path"):
        coros["caption"] = caption_image.ainvoke({"image_path": state["image_path"]})
    if plan.get("aesthetic_score") and state.get("image_path"):
        coros["aesthetic"] = score_aesthetic.ainvoke({"image_path": state["image_path"]})
        coros["gradcam"] = asyncio.to_thread(generate_gradcam, state["image_path"])
    if plan.get("extract_exif") and state.get("image_path"):
        coros["exif"] = fetch_exif.ainvoke({"image_path": state["image_path"]})
    if plan.get("retrieve_photography_tips"):
        coros["rag"] = retrieve_photography_tips.ainvoke({"query": rag_query})

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
            existing = state.get("retrieved_docs") or []
            existing_set = set(existing)
            new_unique = [d for d in output if d not in existing_set]
            updates["retrieved_docs"] = (existing + new_unique)[:8]  # cap at 8 unique chunks
        else:
            updates[key] = output

    return updates


async def final_answer_node(state: AgentState, llm):
    score = state.get("aesthetic_score")
    dist  = state.get("aesthetic_dist")

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

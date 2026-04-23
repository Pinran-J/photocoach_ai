from dotenv import load_dotenv
from tools.models_utils import score_aesthetic
from tools.exif_tool import fetch_exif
from tools.captioning_tool import caption_image
from rag.retriever_fetch_tool import retrieve_photography_tips
from agent.agent_state import ToolCalls, ReflectDecision
from langchain.chat_models import init_chat_model
from agent.graph import build_graph

import gradio as gr

load_dotenv()

class chat_interface:
    def __init__(self):
        tools = [
            retrieve_photography_tips,
            score_aesthetic,
            fetch_exif,
            caption_image,
        ]

        # temperature=0 + with_structured_output uses the model's native function-calling
        # schema, so partial/trailing text from the model never reaches the JSON parser.
        planner_base = init_chat_model("gpt-5-nano", temperature=0)
        tool_decider_model   = planner_base.with_structured_output(ToolCalls)
        reflect_decider_model = planner_base.with_structured_output(ReflectDecision)

        # temperature=0.3 for natural, varied coaching language
        response_model = init_chat_model("gpt-5-nano", temperature=0.3)

        self.graph = build_graph(tool_decider_model, response_model, reflect_decider_model)
        self.image = None

    def set_image(self, image_path):
        if image_path is None:
            return None, "No image uploaded."
        return image_path, "✅ Image uploaded and ready"

    async def async_rag_chat(self, message, history, image_path):
        try:
            initial_state = {
                "user_query": message,
                "image_path": image_path,
                "messages": history,
            }
            emitted_tools = set()
            emitted_tool_plan = False
            emitted_iterations = set()   # tracks reflect rounds already shown
            emitted_docs_count = 0       # how many retrieved_docs have already been displayed
            results = []
            final_msg_started = False
            # gr.update() = no-op: leaves the component unchanged until gradcam actually runs
            gradcam_current = gr.update()
            tab_update = gr.update()

            async for stream_mode, chunk in self.graph.astream(
                initial_state, stream_mode=["values", "messages"]
            ):
                if stream_mode == "values":
                    final_state = chunk

                    # ── Reflect node output ──────────────────────────────────────
                    iterations = final_state.get("iterations", 0)
                    if iterations > 0 and iterations not in emitted_iterations:
                        emitted_iterations.add(iterations)
                        plan = final_state["tool_plan"]
                        if any(plan.values()):
                            # Build a readable summary of what reflect decided to fetch
                            extra = []
                            if plan.get("caption_image"):   extra.append("caption")
                            if plan.get("aesthetic_score"): extra.append("aesthetic score")
                            if plan.get("extract_exif"):    extra.append("EXIF")
                            if plan.get("retrieve_photography_tips"):
                                rq = final_state.get("retrieval_query", "")
                                if rq:
                                    extra.append(f're-search tips: "{rq}"')
                                    # Allow the new retrieval result to be shown (count-based dedup handles duplicates)
                                    emitted_tools.discard("retrieved_docs")
                                else:
                                    extra.append("photography tips")

                            results.append(gr.ChatMessage(
                                role="assistant",
                                content=f"Missing information detected — fetching: {', '.join(extra)}.",
                                metadata={"title": "🔄 Reflect"}
                            ))
                            yield results, gradcam_current, tab_update

                    # ── Initial tool plan ────────────────────────────────────────
                    if "tool_plan" in final_state:
                        if any(final_state["tool_plan"].values()) and not emitted_tool_plan:
                            to_caption_image = "✓" if final_state["tool_plan"]["caption_image"] else "✗"
                            to_aesthetic_score = "✓" if final_state["tool_plan"]["aesthetic_score"] else "✗"
                            to_extract_exif = "✓" if final_state["tool_plan"]["extract_exif"] else "✗"
                            to_retrieve_photography_tips = "✓" if final_state["tool_plan"]["retrieve_photography_tips"] else "✗"

                            display_msg = (
                                f"Plan: Caption image {to_caption_image}, "
                                f"Aesthetic scoring {to_aesthetic_score}, "
                                f"EXIF extraction {to_extract_exif}, "
                                f"Retrieve photography tips {to_retrieve_photography_tips}."
                            )
                            results.append(gr.ChatMessage(
                                role="assistant",
                                content=display_msg,
                                metadata={"title": "🛠️ Tool plan"}
                            ))
                            emitted_tool_plan = True
                            yield results, gradcam_current, tab_update

                        elif any(final_state["tool_plan"].values()) and emitted_tool_plan:
                            if "caption" in final_state and "caption" not in emitted_tools:
                                results.append(gr.ChatMessage(
                                    role="assistant",
                                    content=final_state["caption"],
                                    metadata={"title": "🛠️ Captioner"}
                                ))
                                emitted_tools.add("caption")
                                yield results, gradcam_current, tab_update

                            if "aesthetic_score" in final_state and "aesthetic_score" not in emitted_tools:
                                results.append(gr.ChatMessage(
                                    role="assistant",
                                    content=f"Score: {final_state['aesthetic_score']:.1f}/10",
                                    metadata={"title": "🛠️ Aesthetic Scorer"}
                                ))
                                emitted_tools.add("aesthetic_score")
                                # Update heatmap and switch to Heatmap tab only when gradcam is ready
                                if final_state.get("gradcam_path"):
                                    gradcam_current = final_state["gradcam_path"]
                                    tab_update = gr.update(selected="heatmap")
                                yield results, gradcam_current, tab_update

                            if "exif" in final_state and "exif" not in emitted_tools:
                                results.append(gr.ChatMessage(
                                    role="assistant",
                                    content=f"{final_state['exif']}",
                                    metadata={"title": "🛠️ EXIF Extractor"}
                                ))
                                emitted_tools.add("exif")
                                yield results, gradcam_current, tab_update

                            if "retrieved_docs" in final_state and "retrieved_docs" not in emitted_tools:
                                all_docs = final_state["retrieved_docs"]
                                new_docs = all_docs[emitted_docs_count:]  # only show chunks added since last display
                                if new_docs:
                                    results.append(gr.ChatMessage(
                                        role="assistant",
                                        content="\n\n".join(new_docs)[:600],
                                        metadata={"title": "🛠️ Photography Tips Retriever"}
                                    ))
                                    emitted_docs_count = len(all_docs)
                                emitted_tools.add("retrieved_docs")
                                yield results, gradcam_current, tab_update

                elif stream_mode == "messages":
                    msg, metadata = chunk
                    if metadata["langgraph_node"] == "final" and msg.content:
                        if not final_msg_started:
                            results.append(gr.ChatMessage(role="assistant", content=msg.content))
                            final_msg_started = True
                        else:
                            results[-1] = gr.ChatMessage(
                                role="assistant",
                                content=results[-1].content + msg.content
                            )
                        yield results, gradcam_current, tab_update

        except Exception as e:
            print(e)
            yield (
                [gr.ChatMessage(role="assistant", content="There was an error processing your request. Please try again.")],
                gr.update(),
                gr.update(),
            )

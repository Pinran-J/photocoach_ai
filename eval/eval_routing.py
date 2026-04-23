"""
Agent Planner — Tool Routing Accuracy Evaluation
=================================================
Tests whether the planner node selects the correct tools for a given
user query. Uses a hand-labeled test set of 12 cases that cover the
main tool-selection scenarios.

Metrics reported:
  - Per-tool precision / recall (caption, aesthetic, exif, rag)
  - Overall exact-match accuracy (all 4 tools predicted correctly)
  - Confusion summary (which tools are over/under-triggered)

Run:
    python eval/eval_routing.py

Results saved to eval/routing_results.md
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from agent.agent_state import ToolCalls
from agent.nodes import planner_node

load_dotenv()

# ── Labeled test set ──────────────────────────────────────────────────────────
# Each case: (query, image_present, expected_tools, note)
# Expected tools are what the planner *should* select based on the tool rules.

TOOL_KEYS = ["caption_image", "aesthetic_score", "extract_exif", "retrieve_photography_tips"]

TEST_CASES = [
    # ── No-image queries ──────────────────────────────────────────────────────
    {
        "query": "How do I get a blurry background in portrait photography?",
        "image_present": False,
        "expected": dict(caption_image=False, aesthetic_score=False, extract_exif=False, retrieve_photography_tips=True),
        "note": "technique Q, no image → RAG only",
    },
    {
        "query": "Explain the rule of thirds in composition.",
        "image_present": False,
        "expected": dict(caption_image=False, aesthetic_score=False, extract_exif=False, retrieve_photography_tips=True),
        "note": "knowledge Q → RAG only",
    },
    {
        "query": "What metering mode is best for portrait photography?",
        "image_present": False,
        "expected": dict(caption_image=False, aesthetic_score=False, extract_exif=False, retrieve_photography_tips=True),
        "note": "settings Q, no image → RAG only",
    },
    {
        "query": "Hello! How are you today?",
        "image_present": False,
        "expected": dict(caption_image=False, aesthetic_score=False, extract_exif=False, retrieve_photography_tips=False),
        "note": "greeting → no tools",
    },
    # ── Image queries ─────────────────────────────────────────────────────────
    {
        "query": "What camera settings were used in this shot?",
        "image_present": True,
        "expected": dict(caption_image=False, aesthetic_score=False, extract_exif=True, retrieve_photography_tips=False),
        "note": "metadata-only → EXIF only",
    },
    {
        "query": "What's the shutter speed and ISO in this photo?",
        "image_present": True,
        "expected": dict(caption_image=False, aesthetic_score=False, extract_exif=True, retrieve_photography_tips=False),
        "note": "specific metadata → EXIF only",
    },
    {
        "query": "Score my photo.",
        "image_present": True,
        "expected": dict(caption_image=True, aesthetic_score=True, extract_exif=False, retrieve_photography_tips=False),
        "note": "score request → aesthetic + caption",
    },
    {
        "query": "Give me full feedback on this photo.",
        "image_present": True,
        "expected": dict(caption_image=True, aesthetic_score=True, extract_exif=True, retrieve_photography_tips=True),
        "note": "full feedback → all tools",
    },
    {
        "query": "How can I improve the composition of this image?",
        "image_present": True,
        "expected": dict(caption_image=True, aesthetic_score=False, extract_exif=False, retrieve_photography_tips=True),
        "note": "composition-only → caption + RAG, no aesthetic (narrow technique Q)",
    },
    {
        "query": "Is the exposure technically correct in this photo?",
        "image_present": True,
        "expected": dict(caption_image=True, aesthetic_score=False, extract_exif=True, retrieve_photography_tips=True),
        "note": "exposure Q → EXIF + RAG + caption",
    },
    {
        "query": "Tell me if my camera settings were optimal.",
        "image_present": True,
        "expected": dict(caption_image=True, aesthetic_score=False, extract_exif=True, retrieve_photography_tips=True),
        "note": "settings optimality → EXIF + RAG + caption",
    },
    {
        "query": "How can I improve this photo?",
        "image_present": True,
        "expected": dict(caption_image=True, aesthetic_score=True, extract_exif=False, retrieve_photography_tips=True),
        "note": "general improvement → caption + aesthetic + RAG",
    },
]


# ── Eval logic ────────────────────────────────────────────────────────────────

def run_eval():
    planner_base = init_chat_model("gpt-5-nano", temperature=0)
    tool_decider = planner_base.with_structured_output(ToolCalls)

    predictions = []
    print(f"Running planner on {len(TEST_CASES)} test cases...\n")

    for i, case in enumerate(TEST_CASES, 1):
        fake_state = {
            "user_query": case["query"],
            "image_path": "/fake/test.jpg" if case["image_present"] else None,
            "messages": [],
        }
        result = planner_node(fake_state, tool_decider)
        predicted = result["tool_plan"]
        predictions.append(predicted)

        expected = case["expected"]
        match = all(predicted[k] == expected[k] for k in TOOL_KEYS)
        status = "✓" if match else "✗"
        print(f"[{i:2d}] {status}  {case['note']}")
        if not match:
            for k in TOOL_KEYS:
                if predicted[k] != expected[k]:
                    label = "caption" if k == "caption_image" else \
                            "aesthetic" if k == "aesthetic_score" else \
                            "exif" if k == "extract_exif" else "rag"
                    direction = "FP" if predicted[k] else "FN"
                    print(f"         {direction} on {label}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    n = len(TEST_CASES)
    exact_matches = sum(
        all(predictions[i][k] == TEST_CASES[i]["expected"][k] for k in TOOL_KEYS)
        for i in range(n)
    )

    per_tool = {}
    tool_display = {
        "caption_image": "caption_image",
        "aesthetic_score": "aesthetic_score",
        "extract_exif": "extract_exif",
        "retrieve_photography_tips": "retrieve_tips",
    }
    for k in TOOL_KEYS:
        tp = sum(predictions[i][k] and TEST_CASES[i]["expected"][k] for i in range(n))
        fp = sum(predictions[i][k] and not TEST_CASES[i]["expected"][k] for i in range(n))
        fn = sum(not predictions[i][k] and TEST_CASES[i]["expected"][k] for i in range(n))
        tn = sum(not predictions[i][k] and not TEST_CASES[i]["expected"][k] for i in range(n))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc       = (tp + tn) / n
        per_tool[k] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
                           precision=precision, recall=recall, f1=f1, acc=acc)

    exact_acc = exact_matches / n

    # ── Build report ──────────────────────────────────────────────────────────
    lines = [
        "# Agent Planner — Tool Routing Accuracy",
        "",
        f"**Test cases:** {n}  ",
        "**Planner model:** gpt-5-nano (temperature=0, structured output)  ",
        "",
        f"**Overall exact-match accuracy: {exact_acc:.1%} ({exact_matches}/{n})**  ",
        "(exact-match = all 4 tool decisions correct for a single query)",
        "",
        "## Per-Tool Metrics",
        "",
        "| Tool | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |",
        "|------|----------|-----------|--------|-----|----|----|----|----|",
    ]
    for k in TOOL_KEYS:
        m = per_tool[k]
        lines.append(
            f"| {tool_display[k]} | {m['acc']:.1%} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} "
            f"| {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )

    lines += [
        "",
        "## Per-Case Results",
        "",
        "| # | Query | Image | Exact | Notes |",
        "|---|-------|-------|-------|-------|",
    ]
    for i, (case, pred) in enumerate(zip(TEST_CASES, predictions), 1):
        match = all(pred[k] == case["expected"][k] for k in TOOL_KEYS)
        q = case["query"][:52] + "..." if len(case["query"]) > 52 else case["query"]
        img = "Yes" if case["image_present"] else "No"
        status = "✓" if match else "✗"
        lines.append(f"| {i} | {q} | {img} | {status} | {case['note']} |")

    output = "\n".join(lines)
    print("\n" + output)

    os.makedirs("eval", exist_ok=True)
    with open("eval/routing_results.md", "w") as f:
        f.write(output)
    print("\nSaved to eval/routing_results.md")


if __name__ == "__main__":
    run_eval()

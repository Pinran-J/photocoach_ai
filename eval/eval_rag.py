"""
RAG Pipeline — RAGAS Evaluation
================================
Evaluates the PhotoCoach retrieval pipeline using three metrics:

  1. Faithfulness      (RAGAS) — are answer claims grounded in context?
  2. Answer Relevance  (RAGAS) — does the answer address the question?
  3. Context Relevance (LLM-as-judge) — are retrieved chunks on-topic?
     (RAGAS 0.4.x doesn't expose a reference-free context metric in
      the old-style API, so we use a single batched LLM call instead.)

Pipeline under test:
  Pinecone MMR (k=5, fetch_k=20, λ=0.6)
  → CrossEncoder reranker (ms-marco-MiniLM-L-6-v2)
  → gpt-5-nano response generation

Run:
    PYTHONPATH=. python eval/eval_rag.py

Results saved to eval/ragas_results.md
"""

import os
import warnings
warnings.filterwarnings("ignore")  # suppress RAGAS deprecation noise

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from rag.retriever_fetch_tool import retriever, reranker

load_dotenv()

JUDGE_MODEL = "gpt-4o-mini"
RESPONSE_MODEL = "gpt-5-nano"

TEST_QUERIES = [
    "how do I get a blurry background in portrait photography",
    "what camera settings should I use for night photography",
    "explain the rule of thirds in composition",
    "how does ISO affect image noise",
    "tips for shooting in harsh midday sunlight",
    "how to freeze motion in sports photography",
    "what metering mode should I use for portraits",
    "how to shoot macro photography without a macro lens",
    "what is Rembrandt lighting in portrait photography",
    "explain white balance and when to adjust it",
]

RESPONSE_PROMPT = """You are a photography coach. Use ONLY the retrieved passages below to answer the question.
Be specific and actionable.

Retrieved passages:
{context}

Question: {question}
Answer:"""


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def retrieve_and_rerank(query: str) -> list[str]:
    docs = retriever.invoke(query)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc.page_content for _, doc in ranked[:5]]


def generate_response(llm: ChatOpenAI, query: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    prompt = RESPONSE_PROMPT.format(context=context_block, question=query)
    return llm.invoke(prompt).content


def context_relevance_score(client: OpenAI, query: str, chunks: list[str]) -> float:
    """Batched LLM-as-judge: how topically relevant are the retrieved chunks?
    Returns mean score normalised to 0–1."""
    numbered = "\n\n".join(f"[{i+1}] {c[:400]}" for i, c in enumerate(chunks))
    prompt = (
        f"Query: {query}\n\nRetrieved chunks:\n{numbered}\n\n"
        f"Rate each chunk's relevance to the query on a 1–5 scale:\n"
        f"1=off-topic, 3=somewhat relevant, 5=directly answers.\n"
        f"Reply with exactly {len(chunks)} comma-separated integers, e.g.: 4,3,5,2,3"
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=20,
        temperature=0,
    )
    try:
        raw = resp.choices[0].message.content.strip()
        scores = [max(1, min(5, int(x.strip()))) for x in raw.split(",") if x.strip().isdigit()]
        if len(scores) == len(chunks):
            return round((sum(scores) / len(scores) - 1) / 4, 3)  # normalise to 0–1
    except Exception:
        pass
    return 0.5


# ── Main ──────────────────────────────────────────────────────────────────────

def run_eval():
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response_llm = ChatOpenAI(model=RESPONSE_MODEL, temperature=0.3)
    judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    samples = []
    ctx_scores = []

    print(f"Retrieving + generating responses for {len(TEST_QUERIES)} queries...\n")
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] {query[:65]}")
        contexts = retrieve_and_rerank(query)
        response = generate_response(response_llm, query, contexts)
        ctx_scores.append(context_relevance_score(openai_client, query, contexts))
        samples.append(SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=contexts,
        ))

    dataset = EvaluationDataset(samples=samples)

    print("\nRunning RAGAS evaluation (Faithfulness + Answer Relevance)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=embeddings,
    )
    df = result.to_pandas()

    faith_avg = df["faithfulness"].mean()
    ans_rel_avg = df["answer_relevancy"].mean()
    ctx_rel_avg = round(sum(ctx_scores) / len(ctx_scores), 3)

    lines = [
        "# RAG Pipeline — RAGAS Evaluation Results",
        "",
        f"**Test queries:** {len(TEST_QUERIES)}  ",
        "**Retriever:** Pinecone MMR (k=5, fetch_k=20, λ=0.6)  ",
        "**Reranker:** CrossEncoder `ms-marco-MiniLM-L-6-v2`  ",
        f"**Response model:** {RESPONSE_MODEL}  ",
        f"**Judge model:** {JUDGE_MODEL}  ",
        "",
        "| Metric | Score | Range | Interpretation |",
        "|--------|-------|-------|----------------|",
        f"| Faithfulness | **{faith_avg:.3f}** | 0–1 | Fraction of answer claims grounded in context |",
        f"| Answer Relevance | **{ans_rel_avg:.3f}** | 0–1 | How well the answer addresses the question |",
        f"| Context Relevance | **{ctx_rel_avg:.3f}** | 0–1 | LLM-judged topical relevance of retrieved chunks |",
        "",
        "## Score Guide",
        "- **Faithfulness** 1.0 = fully grounded, 0.0 = hallucinated",
        "- **Answer Relevance** 1.0 = perfectly on-topic, 0.0 = off-topic",
        "- **Context Relevance** 1.0 = all chunks directly answer the query",
        "",
        "## Per-Query Breakdown",
        "",
        "| Query | Faithfulness | Answer Relevance | Context Relevance |",
        "|-------|-------------|-----------------|-------------------|",
    ]
    for row, ctx in zip(df.itertuples(), ctx_scores):
        q = (row.user_input[:52] + "...") if len(row.user_input) > 52 else row.user_input
        lines.append(f"| {q} | {row.faithfulness:.3f} | {row.answer_relevancy:.3f} | {ctx:.3f} |")

    output = "\n".join(lines)
    print("\n" + output)

    os.makedirs("eval", exist_ok=True)
    with open("eval/ragas_results.md", "w") as f:
        f.write(output)
    print("\nSaved to eval/ragas_results.md")


if __name__ == "__main__":
    run_eval()

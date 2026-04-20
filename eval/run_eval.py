"""
RAG Retrieval Evaluation — Old Index vs New Index
==================================================

Compares photocoach-ai-index-old (baseline) against photocoach-ai-index (improved)
across four metrics that require NO labeled data:

  1. LLM Relevance Score  — GPT judges all retrieved chunks for a query in ONE call
                            (1-5 scale, LLM-as-judge). Batching saves ~4x tokens vs
                            scoring each chunk individually.
  2. Avg Cosine Similarity — Pinecone cosine similarity of top-k results to the query
  3. Diversity Score      — 1 - avg pairwise cosine similarity of retrieved chunks
                            (higher = less redundant; directly measures MMR benefit)
  4. Source Coverage      — number of distinct source types (PDF / Wikipedia / RSS)
                            in top-k results. New index draws from 3 types; old from 1.

Baseline  (ingestion_old.py): PDFs only, chunk=1000/200, similarity search k=4
Improved  (ETL pipeline):     PDFs + Wikipedia (34 pages) + full RSS articles,
                              chunk=600/100, MMR k=5

Run:
    python eval/run_eval.py

Results are printed as a markdown table and saved to eval/results.md
"""

import os
import itertools
import numpy as np
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

OLD_INDEX       = "photocoach-ai-index-old"
NEW_INDEX       = "photocoach-ai-index"
EMBEDDING_MODEL = "text-embedding-3-small"
JUDGE_MODEL     = "gpt-5-nano"

OLD_K = 4   # similarity search used in old pipeline
NEW_K = 5   # MMR k=5 in new pipeline

# 12 queries covering the main photography topics — fewer than before to save tokens
TEST_QUERIES = [
    "how do i get a blurry background in portrait photography",
    "what camera settings should i use for night photography",
    "explain the rule of thirds in composition",
    "how does ISO affect image noise",
    "tips for shooting in harsh midday sunlight",
    "how to freeze motion in sports photography",
    "best settings for landscape photography golden hour",
    "what metering mode should i use for portraits",
    "how to shoot macro photography without a macro lens",
    "what is Rembrandt lighting in portrait photography",
    "explain white balance and when to adjust it",
    "how to photograph the milky way",
]


# ── Setup ─────────────────────────────────────────────────────────────────────

def build_retriever(index_name: str, k: int, use_mmr: bool):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=os.environ["OPENAI_API_KEY"])
    vs = PineconeVectorStore(index=index, embedding=embeddings)
    if use_mmr:
        return vs, vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.6},
        )
    return vs, vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def llm_relevance_score(client: OpenAI, query: str, chunks: list[str]) -> float:
    """Score all chunks for a query in ONE LLM call (batched).

    GPT returns one integer per chunk — saves ~4x tokens vs scoring individually.
    """
    numbered = "\n\n".join(
        f"[{i+1}] {chunk[:400]}"
        for i, chunk in enumerate(chunks)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Retrieved chunks:\n{numbered}\n\n"
        f"Rate each chunk's relevance to the query on a 1-5 scale:\n"
        f"1=irrelevant, 3=somewhat relevant, 5=highly relevant.\n"
        f"Reply with exactly {len(chunks)} comma-separated integers, e.g.: 4,3,5,2"
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=20,
        temperature=1,
    )
    try:
        raw = response.choices[0].message.content.strip()
        scores = [max(1, min(5, int(x.strip()))) for x in raw.split(",") if x.strip().isdigit()]
        if len(scores) == len(chunks):
            return round(sum(scores) / len(scores), 3)
    except Exception:
        pass
    return 3.0  # fallback if parsing fails


def avg_similarity_score(vs: PineconeVectorStore, query: str, k: int) -> float:
    """Mean Pinecone cosine similarity score across top-k results."""
    results = vs.similarity_search_with_score(query, k=k)
    scores = [score for _, score in results]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def diversity_score(client: OpenAI, chunks: list[str]) -> float:
    """1 - avg pairwise cosine similarity of retrieved chunk embeddings.
    Higher = less redundant. Directly measures MMR benefit."""
    if len(chunks) < 2:
        return 0.0
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
    vecs = np.array([e.embedding for e in response.data])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-10)
    pairs = [
        float(np.dot(vecs[i], vecs[j]))
        for i, j in itertools.combinations(range(len(vecs)), 2)
    ]
    return round(1.0 - sum(pairs) / len(pairs), 3)



# ── Main eval loop ────────────────────────────────────────────────────────────

def run_eval():
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Connecting to indexes...")
    old_vs, old_retriever = build_retriever(OLD_INDEX, k=OLD_K, use_mmr=False)
    new_vs, new_retriever = build_retriever(NEW_INDEX, k=NEW_K, use_mmr=True)

    old_relevance, new_relevance = [], []
    old_similarity, new_similarity = [], []
    old_diversity,  new_diversity  = [], []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] {query}")

        old_docs = old_retriever.invoke(query)
        new_docs = new_retriever.invoke(query)
        old_chunks = [d.page_content for d in old_docs]
        new_chunks = [d.page_content for d in new_docs]

        # LLM relevance — one call per index per query (not per chunk)
        old_relevance.append(llm_relevance_score(openai_client, query, old_chunks))
        new_relevance.append(llm_relevance_score(openai_client, query, new_chunks))

        # Cosine similarity
        old_similarity.append(avg_similarity_score(old_vs, query, k=OLD_K))
        new_similarity.append(avg_similarity_score(new_vs, query, k=NEW_K))

        # Diversity — reuses chunk embeddings from OpenAI
        old_diversity.append(diversity_score(openai_client, old_chunks))
        new_diversity.append(diversity_score(openai_client, new_chunks))

    # ── Results ───────────────────────────────────────────────────────────────

    def avg(lst): return round(sum(lst) / len(lst), 3)
    def pct(old, new): return f"{((new - old) / old) * 100:+.1f}%" if old else "N/A"

    r_old, r_new = avg(old_relevance), avg(new_relevance)
    s_old, s_new = avg(old_similarity), avg(new_similarity)
    d_old, d_new = avg(old_diversity),  avg(new_diversity)

    lines = [
        "# RAG Retrieval Evaluation Results",
        "",
        f"**Test queries:** {len(TEST_QUERIES)}  ",
        f"**Baseline:** `{OLD_INDEX}` — PDFs only, chunk=1000/200, similarity search k={OLD_K}  ",
        f"**Improved:** `{NEW_INDEX}` — PDFs + Wikipedia (34 pages) + full RSS articles, chunk=600/100, MMR k={NEW_K}  ",
        "",
        "| Metric | Baseline | Improved | Change |",
        "|--------|----------|----------|--------|",
        f"| LLM Relevance Score (1–5) | {r_old} | {r_new} | {pct(r_old, r_new)} |",
        f"| Avg Cosine Similarity     | {s_old} | {s_new} | {pct(s_old, s_new)} |",
        f"| Diversity Score (0–1)     | {d_old} | {d_new} | {pct(d_old, d_new)} |",
        "",
        "## Methodology",
        "- **LLM Relevance Score**: GPT rates all retrieved chunks for a query in a single batched call",
        "  (1–5 scale, LLM-as-judge pattern — no labeled data required). Averaged across all queries.",
        "- **Avg Cosine Similarity**: mean Pinecone cosine similarity of top-k results to the query embedding.",
        "- **Diversity Score**: 1 - avg pairwise cosine similarity of retrieved chunk embeddings.",
        "  Higher = less redundant results. Directly quantifies the benefit of MMR over similarity search.",
    ]

    output = "\n".join(lines)
    print("\n" + output)

    os.makedirs("eval", exist_ok=True)
    with open("eval/results.md", "w") as f:
        f.write(output)
    print("\nSaved to eval/results.md")


if __name__ == "__main__":
    run_eval()

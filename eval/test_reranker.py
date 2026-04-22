"""
Temporary — Reranker Decision Test
===================================
Tests whether the CrossEncoder reranker meaningfully improves retrieval on the
current index (coaching sources, 300-token chunks) vs MMR-only baseline.

Run once to decide whether to keep the reranker in production.
Delete this file after the decision is made.

    python eval/test_reranker.py
"""

import os
import itertools
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import CrossEncoder

load_dotenv()

INDEX_NAME      = "photocoach-ai-index"
EMBEDDING_MODEL = "text-embedding-3-small"
JUDGE_MODEL     = "gpt-4o-mini"   # more reliable scoring than gpt-5-nano for judge tasks

BASELINE_K       = 5
BASELINE_FETCH_K = 20

RERANKED_FETCH_K = 40
RERANKED_MMR_K   = 10
RERANKED_TOP_K   = 5

# Mix of basic and specific coaching questions — keeps token cost low while
# testing whether reranker helps on queries that need precise retrieval.
TEST_QUERIES = [
    # Broad — should retrieve well regardless
    "how to get a blurry background in portrait photography",
    "best settings for photographing the milky way",
    # Specific technique — where reranker should help most
    "what aperture do I need to keep a group of 6 people all in focus",
    "how to expose a backlit subject without blowing out the background",
    "silky smooth waterfall effect what shutter speed and settings do I need",
    "how to reduce noise when shooting indoors at high ISO without a tripod",
    # Lighting / genre specific
    "Rembrandt lighting setup portrait one light source",
    "street photography tips harsh midday sunlight",
]


def build_vector_store() -> PineconeVectorStore:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return PineconeVectorStore(index=index, embedding=embeddings)


def llm_relevance_score(client: OpenAI, query: str, chunks: list[str]) -> float:
    numbered = "\n\n".join(f"[{i+1}] {chunk[:400]}" for i, chunk in enumerate(chunks))
    prompt = (
        f"Query: {query}\n\n"
        f"Retrieved chunks:\n{numbered}\n\n"
        f"Rate each chunk's relevance to the query on a 1-5 scale:\n"
        f"1 = completely off-topic, "
        f"3 = mentions the subject but gives no actionable advice, "
        f"5 = directly and specifically answers the query with practical technique.\n"
        f"Reply with exactly {len(chunks)} comma-separated integers, e.g.: 4,3,5,2,4"
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=20,
        temperature=0,
    )
    try:
        raw = response.choices[0].message.content.strip()
        scores = [max(1, min(5, int(x.strip()))) for x in raw.split(",") if x.strip().isdigit()]
        if len(scores) == len(chunks):
            return round(sum(scores) / len(scores), 3)
    except Exception:
        pass
    return 3.0


def avg_cosine(vs: PineconeVectorStore, query: str, k: int) -> float:
    results = vs.similarity_search_with_score(query, k=k)
    scores = [s for _, s in results]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def diversity(client: OpenAI, chunks: list[str]) -> float:
    if len(chunks) < 2:
        return 0.0
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
    vecs = np.array([e.embedding for e in response.data])
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-10)
    pairs = [float(np.dot(vecs[i], vecs[j])) for i, j in itertools.combinations(range(len(vecs)), 2)]
    return round(1.0 - sum(pairs) / len(pairs), 3)


def run():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Loading CrossEncoder...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("Connecting to index...")
    vs = build_vector_store()

    baseline_retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": BASELINE_K, "fetch_k": BASELINE_FETCH_K, "lambda_mult": 0.6},
    )
    candidate_retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RERANKED_MMR_K, "fetch_k": RERANKED_FETCH_K, "lambda_mult": 0.6},
    )

    rows = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] {query}")

        base_docs   = baseline_retriever.invoke(query)
        base_chunks = [d.page_content for d in base_docs]

        cand_docs = candidate_retriever.invoke(query)
        pairs     = [(query, d.page_content) for d in cand_docs]
        scores    = reranker.predict(pairs)
        rank_docs   = [d for _, d in sorted(zip(scores, cand_docs), key=lambda x: x[0], reverse=True)[:RERANKED_TOP_K]]
        rank_chunks = [d.page_content for d in rank_docs]

        base_rel  = llm_relevance_score(client, query, base_chunks)
        rank_rel  = llm_relevance_score(client, query, rank_chunks)
        base_cos  = avg_cosine(vs, query, k=BASELINE_K)
        rank_cos  = avg_cosine(vs, query, k=RERANKED_TOP_K)
        base_div  = diversity(client, base_chunks)
        rank_div  = diversity(client, rank_chunks)

        rows.append((query, base_rel, rank_rel, base_cos, rank_cos, base_div, rank_div))
        print(f"    Relevance  MMR={base_rel}  Reranked={rank_rel}")
        print(f"    Cosine     MMR={base_cos}  Reranked={rank_cos}")
        print(f"    Diversity  MMR={base_div}  Reranked={rank_div}")

    def avg(lst): return round(sum(lst) / len(lst), 3)
    def pct(a, b): return f"{((b - a) / a * 100):+.1f}%" if a else "N/A"

    br = avg([r[1] for r in rows]); rr = avg([r[2] for r in rows])
    bc = avg([r[3] for r in rows]); rc = avg([r[4] for r in rows])
    bd = avg([r[5] for r in rows]); rd = avg([r[6] for r in rows])

    output = "\n".join([
        "# Reranker Decision Test",
        "",
        f"**Index:** `{INDEX_NAME}` (coaching sources, 300-token chunks)  ",
        f"**Queries:** {len(TEST_QUERIES)}  ",
        f"**Baseline:** MMR fetch_k={BASELINE_FETCH_K}, k={BASELINE_K}  ",
        f"**Reranked:** MMR fetch_k={RERANKED_FETCH_K}, k={RERANKED_MMR_K} → CrossEncoder top {RERANKED_TOP_K}  ",
        "",
        "| Metric | MMR only | MMR + Reranker | Change |",
        "|--------|----------|----------------|--------|",
        f"| LLM Relevance (1–5) | {br} | {rr} | {pct(br, rr)} |",
        f"| Avg Cosine Similarity | {bc} | {rc} | {pct(bc, rc)} |",
        f"| Diversity (0–1) | {bd} | {rd} | {pct(bd, rd)} |",
        "",
        "## Per-query breakdown",
        "",
        "| Query | Base Rel | Rank Rel | Base Cos | Rank Cos |",
        "|-------|----------|----------|----------|----------|",
        *[f"| {r[0][:55]}… | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows],
        "",
        "## Decision guide",
        "- Keep reranker if LLM Relevance improves ≥ +0.2 (4% on a 5-pt scale).",
        "- Drop reranker if improvement is < +0.1 — latency cost outweighs marginal gain.",
    ])

    print("\n" + output)
    out = "eval/test_reranker_results.md"
    with open(out, "w") as f:
        f.write(output)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run()

# RAG Retrieval Evaluation Results

**Test queries:** 20  
**Baseline:** `photocoach-ai-index-old` — PDFs only, chunk=1000/200, similarity search k=4  
**Improved:** `photocoach-ai-index` — PDFs + Wikipedia + RSS, chunk=600/100, MMR k=5  

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| LLM Relevance Score (1–5) | 3.0 | 3.0 | +0.0% |
| Avg Cosine Similarity     | 0.575 | 0.597 | +3.8% |
| Diversity Score (0–1)     | 0.297 | 0.426 | +43.4% |

## Methodology
- **LLM Relevance Score**: GPT-4o-mini rates each retrieved chunk 1–5 for relevance to the
  query (LLM-as-judge — no labeled data required). Averaged across all chunks and queries.
- **Avg Cosine Similarity**: mean Pinecone cosine similarity score across top-k results.
  Measures how closely the retrieved chunks match the query embedding.
- **Diversity Score**: 1 - avg pairwise cosine similarity of retrieved chunk embeddings.
  Higher = less redundant results. Directly quantifies the benefit of MMR retrieval.
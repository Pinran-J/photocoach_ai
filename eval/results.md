# RAG Retrieval Evaluation Results

**Test queries:** 12  
**Baseline:** `photocoach-ai-index-old` — PDFs only, chunk=1000/200, similarity search k=4  
**Improved:** `photocoach-ai-index` — PDFs + Wikipedia (34 pages) + full RSS articles, chunk=600/100, MMR k=5  

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| LLM Relevance Score (1–5) | 3.0 | 3.0 | +0.0% |
| Avg Cosine Similarity     | 0.582 | 0.605 | +4.0% |
| Diversity Score (0–1)     | 0.297 | 0.423 | +42.4% |

## Methodology
- **LLM Relevance Score**: GPT rates all retrieved chunks for a query in a single batched call
  (1–5 scale, LLM-as-judge pattern — no labeled data required). Averaged across all queries.
- **Avg Cosine Similarity**: mean Pinecone cosine similarity of top-k results to the query embedding.
- **Diversity Score**: 1 - avg pairwise cosine similarity of retrieved chunk embeddings.
  Higher = less redundant results. Directly quantifies the benefit of MMR over similarity search.
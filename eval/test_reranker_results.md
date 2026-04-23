# Reranker Decision Test

**Index:** `photocoach-ai-index` (coaching sources, 300-token chunks)  
**Queries:** 8  
**Baseline:** MMR fetch_k=20, k=5  
**Reranked:** MMR fetch_k=40, k=10 → CrossEncoder top 5  

| Metric | MMR only | MMR + Reranker | Change |
|--------|----------|----------------|--------|
| LLM Relevance (1–5) | 3.025 | 3.3 | +9.1% |
| Avg Cosine Similarity | 0.616 | 0.616 | +0.0% |
| Diversity (0–1) | 0.408 | 0.403 | -1.2% |

## Per-query breakdown

| Query | Base Rel | Rank Rel | Base Cos | Rank Cos |
|-------|----------|----------|----------|----------|
| how to get a blurry background in portrait photography… | 2.0 | 3.8 | 0.606 | 0.606 |
| best settings for photographing the milky way… | 3.6 | 4.0 | 0.675 | 0.675 |
| what aperture do I need to keep a group of 6 people all… | 2.8 | 2.8 | 0.552 | 0.552 |
| how to expose a backlit subject without blowing out the… | 3.8 | 3.0 | 0.656 | 0.656 |
| silky smooth waterfall effect what shutter speed and se… | 3.8 | 3.0 | 0.649 | 0.649 |
| how to reduce noise when shooting indoors at high ISO w… | 3.0 | 3.0 | 0.594 | 0.594 |
| Rembrandt lighting setup portrait one light source… | 3.0 | 3.4 | 0.613 | 0.613 |
| street photography tips harsh midday sunlight… | 2.2 | 3.4 | 0.579 | 0.579 |

## Decision guide
- Keep reranker if LLM Relevance improves ≥ +0.2 (4% on a 5-pt scale).
- Drop reranker if improvement is < +0.1 — latency cost outweighs marginal gain.
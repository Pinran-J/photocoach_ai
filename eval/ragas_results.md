# RAG Pipeline — RAGAS Evaluation Results

**Test queries:** 10  
**Retriever:** Pinecone MMR (k=5, fetch_k=20, λ=0.6)  
**Reranker:** CrossEncoder `ms-marco-MiniLM-L-6-v2`  
**Response model:** gpt-5-nano  
**Judge model:** gpt-4o-mini  

| Metric | Score | Range | Interpretation |
|--------|-------|-------|----------------|
| Faithfulness | **0.917** | 0–1 | Fraction of answer claims grounded in context |
| Answer Relevance | **0.834** | 0–1 | How well the answer addresses the question |
| Context Relevance | **0.545** | 0–1 | LLM-judged topical relevance of retrieved chunks |

## Score Guide
- **Faithfulness** 1.0 = fully grounded, 0.0 = hallucinated
- **Answer Relevance** 1.0 = perfectly on-topic, 0.0 = off-topic
- **Context Relevance** 1.0 = all chunks directly answer the query

## Per-Query Breakdown

| Query | Faithfulness | Answer Relevance | Context Relevance |
|-------|-------------|-----------------|-------------------|
| how do I get a blurry background in portrait photogr... | 1.000 | 0.845 | 0.550 |
| what camera settings should I use for night photogra... | 0.963 | 0.794 | 0.700 |
| explain the rule of thirds in composition | 1.000 | 0.791 | 0.700 |
| how does ISO affect image noise | 1.000 | 0.835 | 0.600 |
| tips for shooting in harsh midday sunlight | 0.947 | 0.938 | 0.400 |
| how to freeze motion in sports photography | 0.933 | 0.680 | 0.650 |
| what metering mode should I use for portraits | 0.706 | 0.880 | 0.500 |
| how to shoot macro photography without a macro lens | 0.917 | 0.824 | 0.500 |
| what is Rembrandt lighting in portrait photography | 0.700 | 0.898 | 0.150 |
| explain white balance and when to adjust it | 1.000 | 0.857 | 0.700 |
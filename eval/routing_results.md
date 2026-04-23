# Agent Planner — Tool Routing Accuracy

**Test cases:** 12  
**Planner model:** gpt-5-nano (temperature=0, structured output)  

**Overall exact-match accuracy: 91.7% (11/12)**  
(exact-match = all 4 tool decisions correct for a single query)

## Per-Tool Metrics

| Tool | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |
|------|----------|-----------|--------|-----|----|----|----|----|
| caption_image | 100.0% | 1.00 | 1.00 | 1.00 | 6 | 0 | 0 | 6 |
| aesthetic_score | 100.0% | 1.00 | 1.00 | 1.00 | 3 | 0 | 0 | 9 |
| extract_exif | 100.0% | 1.00 | 1.00 | 1.00 | 5 | 0 | 0 | 7 |
| retrieve_tips | 91.7% | 1.00 | 0.88 | 0.93 | 7 | 0 | 1 | 4 |

## Per-Case Results

| # | Query | Image | Exact | Notes |
|---|-------|-------|-------|-------|
| 1 | How do I get a blurry background in portrait photogr... | No | ✓ | technique Q, no image → RAG only |
| 2 | Explain the rule of thirds in composition. | No | ✓ | knowledge Q → RAG only |
| 3 | What metering mode is best for portrait photography? | No | ✓ | settings Q, no image → RAG only |
| 4 | Hello! How are you today? | No | ✓ | greeting → no tools |
| 5 | What camera settings were used in this shot? | Yes | ✓ | metadata-only → EXIF only |
| 6 | What's the shutter speed and ISO in this photo? | Yes | ✓ | specific metadata → EXIF only |
| 7 | Score my photo. | Yes | ✓ | score request → aesthetic + caption |
| 8 | Give me full feedback on this photo. | Yes | ✓ | full feedback → all tools |
| 9 | How can I improve the composition of this image? | Yes | ✓ | composition-only → caption + RAG, no aesthetic (narrow technique Q) |
| 10 | Is the exposure technically correct in this photo? | Yes | ✗ | exposure Q → EXIF + RAG + caption |
| 11 | Tell me if my camera settings were optimal. | Yes | ✓ | settings optimality → EXIF + RAG + caption |
| 12 | How can I improve this photo? | Yes | ✓ | general improvement → caption + aesthetic + RAG |
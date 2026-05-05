# Experiment Method Matrix

This document summarizes the ten experimental variants used in this repository. All methods operate under the same candidate-constrained Task 4 setting: the system may only retrieve evidence from the official candidate document IDs provided for each query, and all final answer citations must resolve to that candidate set.

Unless otherwise noted, all answer generation runs in this matrix use the same LLM backend: OpenAI `gpt-5.4-mini`. The purpose of the matrix is therefore to compare retrieval, chunking, reranking, and sentence-selection design choices under a fixed answer-generation model.

The matrix is intended for paper writing, slides, and internal ablation tracking. Check marks indicate that a component is explicitly enabled in the corresponding configuration.

## Column Definitions

| Column | Meaning |
|---|---|
| `Model` | The experiment name or run variant used in the repository. |
| `Rule Chunking` | Documents are split into overlapping fixed sentence-window passages using deterministic heuristic rules. |
| `Semantic Chunking` | Documents are split by grouping adjacent sentences with high embedding similarity. |
| `Topic-Shift Chunking` | Documents are split at local semantic drift boundaries, so chunk breaks follow topic transitions between neighboring sentences. |
| `Single-Query BM25` | Retrieval uses only the original query text and a single lexical BM25 ranking. |
| `Query Expansion` | Additional deterministic query variants are generated from the original narrative, such as keyword-focused or intent-aware rewrites. |
| `PRF` | Pseudo-relevance feedback is used to extract expansion terms from top early retrieval results. |
| `RRF Fusion` | Multiple ranked lists are merged with Reciprocal Rank Fusion rather than relying on a single ranking. |
| `Evidence Reranking` | Top retrieved passages are rescored with lightweight evidence-quality features after retrieval. |
| `Citation Prior` | Passage scores are adjusted using a citation-graph prior derived from the provided citation network. |
| `Sentence Reranking (MiniLM Cross-Encoder)` | Candidate answer sentences are reranked with a cross-encoder before final answer selection. |
| `Note` | Why the variant is included in the ablation set or what experimental role it serves. |

## Method Matrix

| Model | Rule Chunking | Semantic Chunking | Topic-Shift Chunking | Single-Query BM25 | Query Expansion | PRF | RRF Fusion | Evidence Reranking | Citation Prior | Sentence Reranking (MiniLM Cross-Encoder) | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `concat_baseline` | ✓ |  |  |  |  |  |  |  |  |  | Minimal lower-bound baseline; included to show performance without a real retrieval-and-ranking stack. |
| `single_query_bm25` | ✓ |  |  | ✓ |  |  |  |  |  |  | Lexical retrieval baseline; isolates the value of a single raw-query BM25 ranking. |
| `rrf_no_rerank` | ✓ |  |  |  | ✓ | ✓ | ✓ |  |  |  | Multi-query retrieval ablation; isolates the gain from query expansion, PRF, and RRF before any reranking stage. |
| `caes_rag_rrf` | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |  | Main proposed retrieval pipeline; included as the primary citation-aware evidence-selection system. |
| `rule_minilm_sentence_rerank` | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Measures whether sentence-level cross-encoder ranking improves answer sentence selection on top of the main rule-based pipeline. |
| `semantic_current_rerank` |  | ✓ |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |  | Chunking ablation; tests whether semantically merged chunks improve downstream retrieval and evidence quality. |
| `topic_shift_current_rerank` |  |  | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ |  | Chunking ablation; tests whether topic-boundary-aware segmentation better captures long-document structure. |
| `semantic_minilm_sentence_rerank` |  | ✓ |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Combination variant; tests whether semantic chunking and sentence-level reranking provide complementary gains. |
| `topic_shift_minilm_sentence_rerank` |  |  | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Combination variant; tests whether topic-shift chunking benefits further from sentence-level cross-encoder filtering. |
| `default` | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |  | Stability control. It keeps the same retrieval setting as `caes_rag_rrf`, so it is retained to verify that final LLM outputs remain close under the same evidence pipeline. |

## Interpretation

These ten variants serve three distinct experimental purposes:

1. Retrieval strength ablation:
   `concat_baseline` -> `single_query_bm25` -> `rrf_no_rerank` -> `caes_rag_rrf`
2. Chunking strategy comparison:
   `rule` vs `semantic` vs `topic_shift`
3. Sentence selection comparison:
   without vs with MiniLM cross-encoder sentence reranking

## Note on `default` vs `caes_rag_rrf`

`default` is intentionally retained even though it uses the same retrieval, fusion, and reranking design as `caes_rag_rrf`. In the OpenAI generation setting, this duplication is useful rather than redundant: if the retrieval stack is identical, the final outputs should remain broadly similar, with only limited variation caused by LLM generation stochasticity or prompt-level effects. This makes `default` a practical stability reference for same-retrieval comparisons.

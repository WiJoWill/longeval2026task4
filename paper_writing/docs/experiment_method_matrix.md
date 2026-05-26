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

## Retrieval Implementation Notes

This section makes the retrieval labels in the matrix precise enough for the paper.

### Query Expansion

- The current "deterministic query expansion" is rule-based and fully local. It does not use an LLM, a prompt, or an external dictionary.
- Expansion starts from the query narrative alone by extracting up to 12 unique non-stopword tokens of length at least 3, in query order.
- The system can then add up to three fixed-format query rewrites before PRF: a `keywords` variant, a `temporal` variant for queries whose tokens overlap a hand-written temporal-intent list, and a `method` variant for queries whose tokens overlap a small hand-written method-intent list.
- The overall number of query variants is capped by `query_variant_limit=5`, counting the original query. In practice this means the run can contain the original query plus up to four additional ranked lists.
- Query-expansion variants are derived from the query alone. Retrieved passages are only used later for the separate PRF variant.

### PRF

- Pseudo-relevance feedback is implemented as a lightweight frequency-based term expansion, not Rocchio/RM-style weighting and not tf-idf.
- The feedback pool is taken from the top `prf_top_passages=4` passages from the first-pass original-query lexical ranking.
- Candidate feedback terms are passage tokens with length at least 4 that are not stopwords and are not already in the extracted query-keyword set.
- Terms are ranked by raw frequency across the feedback passages, and the top `top_n=6` terms are kept.
- These six terms are appended to the keyword set to form one additional `prf` query variant; they are not injected back into every other variant.

### RRF

- RRF fuses ranked passage lists produced by lexical retrieval over the original query and each enabled query variant.
- In the main repository configs, the fused lists are BM25-style lexical rankings for `original`, `keywords`, optional `temporal`, optional `method`, and optional `prf`.
- The code can also append one dense retrieval list if a `dense_model` is configured, but all current Task 4 configs leave `dense_model` empty, so the reported experiments are lexical-only before reranking.
- The RRF constant is `k=60` (`rrf_k: 60` in config).

### Citation-Graph Prior

- The "internal citation-graph prior" is a simple graph-degree prior over candidate document IDs built from the provided citation CSV, not co-citation or a learned graph model.
- For each edge in `data/longeval-sci-2026-citation-network.csv`, the code increments counts for both `citing_doc_id` and `cited_doc_id`. This makes the prior proportional to total citation-network incidence count for a document.
- The counts are normalized by dividing by the maximum observed count, yielding a prior in `[0,1]`.
- The prior is incorporated only in passage reranking, not in first-pass BM25 or RRF itself. For a passage from document `d`, the reranker adds `citation_graph_boost * prior(d)`.
- In the main reranked pipeline, `citation_graph_boost=0.05`, so the final rerank score includes a small additive citation prior term `0.05 * prior(d)`.

## Method Matrix

| Model | Rule Chunking | Semantic Chunking | Topic-Shift Chunking | Single-Query BM25 | Query Expansion | PRF | RRF Fusion | Evidence Reranking | Citation Prior | Sentence Reranking (MiniLM Cross-Encoder) | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `concat_baseline` | Y |  |  |  |  |  |  |  |  |  | Minimal lower-bound baseline; included to show performance without a real retrieval-and-ranking stack. |
| `single_query_bm25` | Y |  |  | Y |  |  |  |  |  |  | Lexical retrieval baseline; isolates the value of a single raw-query BM25 ranking. |
| `rrf_no_rerank` | Y |  |  |  | Y | Y | Y |  |  |  | Multi-query retrieval ablation; isolates the gain from query expansion, PRF, and RRF before any reranking stage. |
| `caes_rag_rrf` | Y |  |  |  | Y | Y | Y | Y | Y |  | Main proposed retrieval pipeline; included as the primary citation-aware evidence-selection system. |
| `rule_minilm_sentence_rerank` | Y |  |  |  | Y | Y | Y | Y | Y | Y | Measures whether sentence-level cross-encoder ranking improves answer sentence selection on top of the main rule-based pipeline. |
| `semantic_current_rerank` |  | Y |  |  | Y | Y | Y | Y | Y |  | Chunking ablation; tests whether semantically merged chunks improve downstream retrieval and evidence quality. |
| `topic_shift_current_rerank` |  |  | Y |  | Y | Y | Y | Y | Y |  | Chunking ablation; tests whether topic-boundary-aware segmentation better captures long-document structure. |
| `semantic_minilm_sentence_rerank` |  | Y |  |  | Y | Y | Y | Y | Y | Y | Combination variant; tests whether semantic chunking and sentence-level reranking provide complementary gains. |
| `topic_shift_minilm_sentence_rerank` |  |  | Y |  | Y | Y | Y | Y | Y | Y | Combination variant; tests whether topic-shift chunking benefits further from sentence-level cross-encoder filtering. |
| `default` | Y |  |  |  | Y | Y | Y | Y | Y |  | Stability control. It keeps the same retrieval setting as `caes_rag_rrf`, so it is retained to verify that final LLM outputs remain close under the same evidence pipeline. |

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

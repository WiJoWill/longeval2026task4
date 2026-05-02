# LongEval Task 4: Semantic Chunking And MiniLM Sentence Rerank Plan

## Summary

Keep the current rule-based chunking path as the baseline, then add two independent configurable document chunking modes:

- `semantic`: merge adjacent sentences by embedding similarity.
- `topic_shift`: split on embedding-based semantic drift between sentences.

Add a sentence-level cross-encoder reranker in the generation stage. The default model is `cross-encoder/ms-marco-MiniLM-L-6-v2`. It scores `(query, sentence)` pairs and replaces the current sentence-level heuristic ordering when configured and available. If the model is not configured, unavailable, or fails at runtime, the pipeline falls back to the current heuristic sentence ranking path.

Default pipeline behavior remains unchanged: rule chunking and the existing upstream retrieval, fusion, and passage reranking chain.

## Key Changes

### Document Chunking Modes

Add these preprocess config fields:

```yaml
preprocess:
  chunk_mode: rule
  passage_max_words: 140
  passage_stride_sentences: 1
  semantic_chunk_model:
  semantic_merge_threshold: 0.72
  topic_shift_model:
  topic_shift_boundary_threshold: 0.18
  min_sentences_per_chunk: 1
  max_sentences_per_chunk:
```

Mode behavior:

- `rule`: keep the current sentence-window implementation unchanged.
- `semantic`: split with `split_sentences()`, embed each sentence, scan left to right, and merge a new sentence into the current chunk when its similarity to the current chunk representation is at least `semantic_merge_threshold` and size limits are not exceeded.
- `topic_shift`: split with `split_sentences()`, embed each sentence, detect semantic drift between the current sentence and the previous sentence or local window center, and start a new chunk when drift exceeds `topic_shift_boundary_threshold`.

Implementation constraints:

- `semantic` and `topic_shift` are independent modes, not a combined mode.
- `semantic_chunk_model` and `topic_shift_model` may point to the same sentence-transformer model.
- If `chunk_mode` is `semantic` or `topic_shift` but the required model is missing, unavailable, or fails to load, fall back to `rule` and print a warning.
- Do not add query-aware rechunking in this version.

### MiniLM Sentence Rerank

Add these generation config fields:

```yaml
generation:
  sentence_rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  sentence_rerank_top_n: 24
```

Generation behavior:

- Keep the current hard filters in `AnswerGenerator._sentence_candidates()`:
  - OCR artifact filtering.
  - boilerplate opener filtering.
  - low-information sentence filtering.
  - length limits.
  - basic query-overlap gate.
- For candidate sentences that pass filtering, run `sentence_transformers.CrossEncoder` on `(query_text, sentence_text)` pairs.
- Use the cross-encoder relevance score as the primary sentence ranking score.
- Keep the current `_sentence_score()` only as a fallback path.
- Preserve temporal answer handling. Earliest/latest temporal sentences should still be selected from the best sentence per dated document, but "best" should be based on cross-encoder score when available.

Model policy:

- Version 1 default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Do not include `BAAI/bge-reranker-base` or larger rerankers by default in this version.
- If the package is missing, model loading fails, or prediction fails, print a warning and fall back to heuristic sentence ranking.

### Existing Retrieval Boundary

Do not replace the upstream evidence-selection stack:

- Keep BM25, query expansion, PRF, RRF, and passage-level reranking.
- The new cross-encoder only reranks sentence candidates before answer generation.
- Do not change output format, citation rules, or validator behavior.

Expected data flow:

1. Load query and candidate documents.
2. Split candidate documents with `chunk_mode`.
3. Select evidence passages with the existing `EvidenceRanker`.
4. Extract sentence candidates from selected passages.
5. Rerank candidate sentences with MiniLM when configured.
6. Generate citation-valid answer records.

## Implementation Notes

Recommended code organization:

- Extend `task4_rag/src/preprocess.py` so `split_documents_into_passages()` dispatches by `chunk_mode`.
- Keep the rule-based splitter as the compatibility path.
- Add semantic and topic-shift helpers in `preprocess.py`, or a small adjacent module if the file becomes too large.
- Extend `GenerationConfig` with `sentence_rerank_model` and `sentence_rerank_top_n`.
- Add a sentence-level cross-encoder helper in `task4_rag/src/generator.py`.
- Update `task4_rag/src/run_task4.py` to pass the new preprocess and generation config fields.
- Update default configs with explicit default fields while preserving current behavior.
- Add two experiment configs:
  - semantic chunking plus MiniLM sentence rerank.
  - topic-shift chunking plus MiniLM sentence rerank.

## Test Plan

Add focused tests that mock embeddings and cross-encoder scores so tests do not download models:

- `rule` mode remains compatible with current chunking behavior.
- `semantic` mode merges high-similarity adjacent sentences and splits low-similarity ones.
- `topic_shift` mode splits at semantic drift boundaries.
- `semantic` and `topic_shift` fall back to `rule` if no model is configured.
- sentence rerank falls back to heuristic ordering when no reranker is configured or available.
- mocked cross-encoder scores change sentence candidate ordering as expected.
- best-sentence-per-document selection uses model scores when available.
- temporal answer generation still produces earliest/latest cited sentences.
- smoke tests cover:
  - `rule` with no sentence rerank.
  - `semantic` with MiniLM sentence rerank.
  - `topic_shift` with MiniLM sentence rerank.
- output validation still enforces unique references, legal citation indices, and non-empty answers.

## Assumptions

- Only `semantic` and `topic_shift` chunking modes are added in this version.
- The modes are independent.
- MiniLM is the default sentence-level cross-encoder.
- Sentence-level rerank replaces only generation-stage sentence ordering.
- Existing hard filters remain in place before cross-encoder scoring.
- Model failure must not terminate a run.

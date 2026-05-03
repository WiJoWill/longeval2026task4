# LLM Evidence Count Summary

Counts are sentence-level evidence candidates included in each OpenAI Batch prompt.

| model | queries | avg evidence | median | min | max | zero | one | <3 | >=10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_query_bm25_openai_llm_v1` | 47 | 7.40 | 7.0 | 2 | 16 | 0 | 0 | 3 | 13 |
| `caes_rag_rrf_openai_llm_v1` | 47 | 6.70 | 7.0 | 0 | 16 | 1 | 0 | 3 | 9 |
| `default_openai_llm_v1` | 47 | 6.70 | 7.0 | 0 | 16 | 1 | 0 | 3 | 9 |
| `rrf_no_rerank_openai_llm_v1` | 47 | 6.66 | 6.0 | 0 | 16 | 1 | 1 | 4 | 10 |
| `semantic_current_openai_llm_v1` | 47 | 4.45 | 4.0 | 1 | 9 | 0 | 1 | 5 | 0 |
| `topic_shift_current_openai_llm_v1` | 47 | 4.15 | 4.0 | 1 | 9 | 0 | 3 | 6 | 0 |
| `concat_baseline_openai_llm_v1` | 47 | 2.13 | 1.0 | 0 | 11 | 13 | 13 | 32 | 1 |
| `semantic_minilm_openai_llm_v1` | 47 | 1.17 | 1.0 | 1 | 3 | 0 | 40 | 46 | 0 |
| `topic_shift_minilm_openai_llm_v1` | 47 | 1.15 | 1.0 | 1 | 3 | 0 | 41 | 46 | 0 |
| `rule_minilm_openai_llm_v1` | 47 | 1.06 | 1.0 | 0 | 2 | 1 | 42 | 47 | 0 |

Main observation: MiniLM sentence-rerank variants pass too few evidence sentences to the LLM; their medians are 1.


## Why `concat_baseline_openai_llm_v1` Can Have Zero Evidence

`concat_baseline_openai_llm_v1` does not perform query-aware retrieval ranking. It applies rule chunking, then takes the first `top_k` chunks from the candidate documents. Those chunks are often document headers, copyright/license text, abstracts for unrelated candidate docs, or otherwise weakly related text.

After chunk selection, the generator still applies sentence-level hard filtering before sending evidence to the LLM. A sentence must pass filters such as OCR/artifact checks, boilerplate checks, length constraints, and basic query-term overlap. If none of the selected chunks contain a sentence that passes these filters, the LLM receives zero sentence-level evidence.

Concrete example:

- Query id: `46395a3cf66a9f6a75c89354410d1493`
- Query: `How should pesticide class-specific findings shape prevention priorities for farm households?`
- `concat_baseline` selected doc: `290464296`
- Selected doc title: `Burden of End-Stage Kidney Disease by Type 2 Diabetes Mellitus Status in South Korea`
- Query terms included: `prevention`, `households`, `specific`, `priorities`, `shape`, `class`, `farm`, `pesticide`, `findings`
- The selected early chunks are about diabetes, kidney disease, article metadata, and copyright text.
- The inspected candidate sentences had no overlap with the query terms, so they were filtered out.
- Result: `sentence_candidates = 0`, so the LLM had no usable evidence for this query.

This explains why `concat_baseline_openai_llm_v1` has many zero-evidence cases: it can have chunks, but still no usable sentence evidence after filtering.

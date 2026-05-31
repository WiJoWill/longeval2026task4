# LLM Evidence Count Summary - Logits-Aware Cliff Cutoff

No OpenAI batch was submitted for this run; these counts are from local batch input/state preparation only.

| model | avg old | avg new | median old | median new | min | max | zero | one | <3 | >=10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_query_bm25_openai_llm_v1` | 7.40 | 7.72 | 7.0 | 8.0 | 2 | 17 | 0 | 0 | 3 | 13 |
| `caes_rag_rrf_openai_llm_v1` | 6.70 | 6.98 | 7.0 | 7.0 | 0 | 16 | 1 | 0 | 3 | 10 |
| `default_openai_llm_v1` | 6.70 | 6.98 | 7.0 | 7.0 | 0 | 16 | 1 | 0 | 3 | 10 |
| `rrf_no_rerank_openai_llm_v1` | 6.66 | 6.98 | 6.0 | 6.0 | 0 | 17 | 1 | 1 | 4 | 12 |
| `semantic_current_openai_llm_v1` | 4.45 | 4.68 | 4.0 | 5.0 | 1 | 9 | 0 | 1 | 5 | 0 |
| `topic_shift_current_openai_llm_v1` | 4.15 | 4.47 | 4.0 | 4.0 | 1 | 10 | 0 | 3 | 6 | 1 |
| `rule_minilm_openai_llm_v1` | 1.06 | 2.60 | 1.0 | 2.0 | 0 | 10 | 3 | 17 | 29 | 1 |
| `concat_baseline_openai_llm_v1` | 2.13 | 2.13 | 1.0 | 1.0 | 0 | 11 | 13 | 13 | 32 | 1 |
| `semantic_minilm_openai_llm_v1` | 1.17 | 1.98 | 1.0 | 2.0 | 0 | 7 | 3 | 20 | 35 | 0 |
| `topic_shift_minilm_openai_llm_v1` | 1.15 | 1.89 | 1.0 | 1.0 | 0 | 7 | 4 | 20 | 36 | 0 |

Selection rule: keep hard-filtered sentence candidates sorted by rerank score; stop when score falls below `answer_candidate_min_score=-10.0`, when a positive-score candidate drops below `previous_score * 0.35`, or when an adjacent signed-logit drop exceeds `answer_candidate_drop_delta=3.0`.

Main observation: MiniLM cross-encoder variants now pass more evidence than the fixed-margin version, but still much less than BM25/RRF current-rerank variants because the hard sentence filters and selected passages leave fewer candidate sentences.

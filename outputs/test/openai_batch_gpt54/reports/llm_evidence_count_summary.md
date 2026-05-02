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

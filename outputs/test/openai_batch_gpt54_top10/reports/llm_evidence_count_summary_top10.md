# LLM Evidence Count Summary - Top10 No Cutoff

Prepared locally only. No OpenAI Batch submission was made for this directory.

Batch input: `outputs/test/openai_batch_gpt54_top10/inputs/all_experiments_requests.jsonl`

Requests: `517`
Prompt rows containing `doc_id`: `517`
Prompt rows containing `timestamp`: `0`

| model | queries | avg evidence | median | min | max | zero | one | <3 | >=10 | avg docs | median docs | max docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gold_llm_all_docs_v1` | 47 | 49.70 | 50.0 | 36 | 50 | 0 | 0 | 0 | 47 | 7.83 | 8.0 | 10 |
| `caes_rag_rrf_openai_llm_v1` | 47 | 10.00 | 10.0 | 10 | 10 | 0 | 0 | 0 | 47 | 2.28 | 2.0 | 4 |
| `concat_baseline_openai_llm_v1` | 47 | 10.00 | 10.0 | 10 | 10 | 0 | 0 | 0 | 47 | 2.09 | 2.0 | 5 |
| `default_openai_llm_v1` | 47 | 10.00 | 10.0 | 10 | 10 | 0 | 0 | 0 | 47 | 2.28 | 2.0 | 4 |
| `rrf_no_rerank_openai_llm_v1` | 47 | 10.00 | 10.0 | 10 | 10 | 0 | 0 | 0 | 47 | 2.43 | 2.0 | 6 |
| `rule_minilm_openai_llm_v1` | 47 | 10.00 | 10.0 | 10 | 10 | 0 | 0 | 0 | 47 | 2.15 | 2.0 | 5 |
| `single_query_bm25_openai_llm_v1` | 47 | 10.00 | 10.0 | 10 | 10 | 0 | 0 | 0 | 47 | 2.62 | 3.0 | 5 |
| `semantic_current_openai_llm_v1` | 47 | 9.70 | 10.0 | 3 | 10 | 0 | 0 | 0 | 43 | 2.57 | 2.0 | 5 |
| `semantic_minilm_openai_llm_v1` | 47 | 9.70 | 10.0 | 3 | 10 | 0 | 0 | 0 | 43 | 2.45 | 2.0 | 6 |
| `topic_shift_current_openai_llm_v1` | 47 | 9.70 | 10.0 | 2 | 10 | 0 | 0 | 1 | 43 | 2.62 | 2.0 | 5 |
| `topic_shift_minilm_openai_llm_v1` | 47 | 9.70 | 10.0 | 2 | 10 | 0 | 0 | 1 | 43 | 2.51 | 2.0 | 6 |

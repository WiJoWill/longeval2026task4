# OpenAI Gold Fulltext Fallback Batch Stats

- model: `gpt-5.4-mini`
- requests_file: `outputs\test\openai_gold_fulltext_smoke\inputs\gold_llm_all_docs_fulltext_fallback_openai_v1_requests.jsonl`
- state_file: `outputs\test\openai_gold_fulltext_smoke\inputs\gold_llm_all_docs_fulltext_fallback_openai_v1_state.json`
- queries: 1
- docs: 10
- docs_using_fullText: 5
- docs_using_abstract_fallback: 5
- docs_using_title_fallback: 0
- docs_missing_any_text: 0
- approx_tokens_per_query_chars_div_4: min 40,629, median 40,629, mean 40,629, max 40,629

| query_id | docs | fullText | abstract_fallback | title_fallback | missing_text | evidence_chars | approx_tokens_chars_div_4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| aa42e210a361571ff4d1fad892b75d15 | 10 | 5 | 5 | 0 | 0 | 162,516 | 40,629 |

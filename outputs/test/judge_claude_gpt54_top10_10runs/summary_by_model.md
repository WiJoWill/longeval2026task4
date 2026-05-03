# Claude Judge Summary

- input: `outputs\test\judge_claude_gpt54_top10_10runs\judged_results.csv`
- rows: 470

| rank | model | n | mean correctness | mean completeness | mean avg score | mean overlap % gold | mean overlap % model | mean Jaccard % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | rrf_no_rerank_openai_llm_v1 | 47 | 3.0 | 2.468 | 2.734 | 70.78 | 92.199 | 69.509 |
| 2 | rule_minilm_openai_llm_v1 | 47 | 2.915 | 2.447 | 2.681 | 81.631 | 96.099 | 78.688 |
| 3 | single_query_bm25_openai_llm_v1 | 47 | 2.979 | 2.383 | 2.681 | 76.099 | 93.972 | 74.22 |
| 4 | caes_rag_rrf_openai_llm_v1 | 47 | 2.936 | 2.362 | 2.649 | 72.589 | 94.681 | 72.021 |
| 5 | default_openai_llm_v1 | 47 | 2.915 | 2.362 | 2.639 | 74.716 | 93.617 | 73.794 |
| 6 | topic_shift_current_openai_llm_v1 | 47 | 2.83 | 2.362 | 2.596 | 77.553 | 90.78 | 74.007 |
| 7 | semantic_current_openai_llm_v1 | 47 | 2.787 | 2.404 | 2.595 | 76.312 | 93.759 | 73.475 |
| 8 | topic_shift_minilm_openai_llm_v1 | 47 | 2.83 | 2.298 | 2.564 | 75.426 | 91.135 | 71.525 |
| 9 | semantic_minilm_openai_llm_v1 | 47 | 2.787 | 2.34 | 2.563 | 79.681 | 90.426 | 72.411 |
| 10 | concat_baseline_openai_llm_v1 | 47 | 1.574 | 1.319 | 1.446 | 13.156 | 23.83 | 12.447 |

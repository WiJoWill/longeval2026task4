# Gemini Gold Fulltext Batch Audit

- run_id: `gold_gemini_fulltext_v1`
- model: `gemini-2.5-flash`
- queries: `1`
- total fullText evidence docs: `5`
- missing fullText refs: `5`
- requests: `outputs\test\gemini_gold_fulltext_smoke\inputs\gold_gemini_fulltext_v1_requests.jsonl`
- state: `outputs\test\gemini_gold_fulltext_smoke\inputs\gold_gemini_fulltext_v1_state.json`

| metric | min | median | mean | max |
|---|---:|---:|---:|---:|
| `fulltext_evidence_docs` | 5 | 5.0 | 5.0 | 5 |
| `missing_fulltext_docs` | 5 | 5.0 | 5.0 | 5 |
| `chars` | 155028 | 155028.0 | 155028.0 | 155028 |
| `words` | 22701 | 22701.0 | 22701.0 | 22701 |
| `approx_tokens_chars_div_4` | 38757 | 38757.0 | 38757.0 | 38757 |
| `request_bytes_utf8` | 162274 | 162274.0 | 162274.0 | 162274 |

| query_id | refs | fullText docs | missing fullText | chars | words | approx tokens | request bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `aa42e210a361571ff4d1fad892b75d15` | 10 | 5 | 5 | 155028 | 22701 | 38757 | 162274 |

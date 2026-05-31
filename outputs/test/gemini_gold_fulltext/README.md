# Gemini Gold Fulltext Batch Status

Prepared but not completed.

## Prepared Input

- Clean inspection directory: `outputs/test/gemini_gold_fulltext_clean/inputs`
- Requests: `outputs/test/gemini_gold_fulltext_clean/inputs/gold_gemini_fulltext_v1_requests.jsonl`
- State: `outputs/test/gemini_gold_fulltext_clean/inputs/gold_gemini_fulltext_v1_state.json`
- Audit: `outputs/test/gemini_gold_fulltext_clean/inputs/gold_gemini_fulltext_v1_audit.md`
- Model: `gemini-2.5-flash`
- Requests: `47`

The request JSONL is valid physical JSONL:

- physical lines: `47`
- JSON parse errors: `0`
- request evidence uses non-empty `metadata.fullText`
- no fallback to `abstract`
- no chunking, retrieval, sentence extraction, or top-N truncation

## Fulltext Coverage

- Candidate query-doc refs: `470`
- Non-empty fullText evidence docs: `222`
- Missing/empty fullText refs: `248`

Per-query strict fullText evidence size:

| metric | min | median | mean | max |
|---|---:|---:|---:|---:|
| fullText docs | 2 | 5.0 | 4.7 | 8 |
| missing fullText docs | 2 | 5.0 | 5.3 | 8 |
| chars | 28,463 | 279,212 | 306,555 | 724,523 |
| words | 4,118 | 39,171 | 44,024 | 100,543 |
| approx tokens, chars/4 | 7,116 | 69,803 | 76,639 | 181,131 |

## Gemini Batch Submit Status

Upload to Gemini File API succeeded, but Batch creation failed:

```text
429 RESOURCE_EXHAUSTED
```

A tiny one-request batch probe also failed with the same error, while a synchronous `gemini-2.5-flash` `generate_content` probe succeeded. That means the API key works for Gemini, but this project/key currently has no usable Gemini Batch quota or the quota is exhausted.

No Gemini batch job was created and no batch result JSONL exists yet.

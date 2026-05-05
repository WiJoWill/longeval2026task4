# Direct Assistant LLM-Style Evaluation

## Scope

This is a direct Codex/assistant review, not an API-generated LLM judge report.

No `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` is available in this workspace, so the local `evaluate_llm_quality` CLI cannot call GPT, Claude, or Gemini. Instead, this report uses:

- the 47-query deterministic RAG quality summaries;
- spot inspection of representative records and failure modes;
- the same rubric intended for the optional LLM judge.

It should be treated as an immediate qualitative review layer, not as a reproducible benchmark artifact.

## Rubric

Scores are on a 1-5 scale:

- `1`: poor
- `2`: weak
- `3`: adequate
- `4`: good
- `5`: excellent

Dimensions:

- context relevance;
- answer relevance;
- faithfulness;
- completeness;
- citation quality;
- noise robustness;
- information integration;
- numeric factuality;
- overall.

## Method-Level Judgments

| Method | Context relevance | Answer relevance | Faithfulness | Completeness | Citation quality | Noise robustness | Information integration | Numeric factuality | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_generated_responses` | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 3 | 1 |
| `generated_responses_repaired_v1` | 2 | 1 | 4 | 1 | 4 | 2 | 2 | 2 | 2 |
| `concat_baseline` | 2 | 1 | 4 | 1 | 4 | 2 | 2 | 2 | 2 |
| `single_query_bm25_v1` | 4 | 3 | 4 | 3 | 4 | 4 | 3 | 4 | 3 |
| `rrf_no_rerank_v1` | 4 | 3 | 4 | 3 | 4 | 4 | 3 | 4 | 3 |
| `caes_rag_rrf_v1` | 4 | 3 | 4 | 3 | 4 | 4 | 3 | 4 | 3 |

## Ranking

1. `rrf_no_rerank_v1`
2. `caes_rag_rrf_v1`
3. `single_query_bm25_v1`
4. `concat_baseline`
5. `generated_responses_repaired_v1`
6. `raw_generated_responses`

The top three are close. Under the current proxy metrics, `rrf_no_rerank_v1` has the best numeric support and essentially the same context quality as the full RRF reranked run. The full run remains a strong candidate because it is designed for better evidence selection, but this particular evaluation does not show a clear advantage over RRF without reranking.

## Main Findings

### Raw Baseline

The raw baseline is not a good RAG answer generator. It often repeats the query in filler form and then lists candidate titles. That makes the lexical `answer_relevance_proxy` look artificially perfect, but the answer is not actually useful.

Strengths:

- Keeps the official candidate references.
- Often cites every candidate title.

Weaknesses:

- Includes uncited filler.
- Does not synthesize an answer.
- Uses many noisy references.
- Title-listing creates superficial faithfulness without useful generation.

Recommended use:

- Keep only as a lower-bound sanity baseline.

### Repaired Baseline

The repaired baseline fixes formatting and citation issues, but it still behaves like a citation-safe title/reference listing baseline.

Strengths:

- Structurally clean.
- Citation indices are valid.
- Faithfulness is high because cited text is mostly copied or title-like.

Weaknesses:

- Very weak answer relevance.
- Does not perform evidence selection.
- Keeps all 10 references, including noisy candidates.

Recommended use:

- Good compliance baseline, not a quality baseline.

### Concat Baseline

The concat baseline is compact and citation-safe, but it uses early candidate passages without real retrieval discipline.

Strengths:

- Cleaner than the raw baseline.
- Mostly faithful to cited content.
- Simple and reproducible.

Weaknesses:

- Low context relevance.
- Low answer relevance.
- Weak numeric factuality support.
- Poor coverage because useful documents may not appear early.

Recommended use:

- Keep as a minimal non-retrieval baseline.

### Single-Query BM25

Single-query BM25 is the first genuinely competitive method. It sharply improves context relevance and noise robustness over the baselines.

Strengths:

- Much better context relevance.
- Good citation grounding.
- Good noise robustness.
- Better answer relevance than baselines.

Weaknesses:

- Still vulnerable to generic query terms.
- Sometimes selects semantically adjacent but not answer-bearing passages.
- Extractive generation can produce awkward or incomplete answers.

Recommended use:

- Strong simple retrieval baseline.

### RRF Without Reranking

RRF without reranking currently looks slightly best under the combined proxy and direct-review picture.

Strengths:

- Best or tied-best context relevance.
- Best numeric support among current methods.
- Strong noise robustness.
- Good faithfulness.

Weaknesses:

- Answer relevance remains only adequate.
- Extracted sentences can be artifact-heavy.
- Multi-document synthesis is limited.

Recommended use:

- Treat as the current practical baseline to beat.

### Full RRF + Reranking

The full `caes_rag_rrf_v1` run is structurally strong and broadly comparable to `rrf_no_rerank_v1`, but the current reranking layer does not clearly improve judged quality.

Strengths:

- Tied-best context relevance and context precision.
- Strong citation faithfulness.
- Strong noise robustness.
- Compact evidence selection.

Weaknesses:

- Answer relevance is slightly lower than single-query BM25 and RRF without reranking.
- Some cited evidence is faithful but off-topic for the actual query.
- Reranker may still overweight generic overlap or noisy full-text snippets.

Recommended use:

- Keep as the main system, but the next work should improve document-level semantic reranking and answer synthesis.

## Cross-Method Interpretation

The main quality gap is no longer citation validity. It is relevance and synthesis.

The current systems are mostly faithful to the text they cite. The remaining problem is that the selected text is sometimes not the right evidence for the query, and the extractive generator often stops short of turning evidence into a direct scientific answer.

This means the next improvements should target:

1. document-level reranking;
2. sentence-level reranking;
3. answer synthesis;
4. a small human-judged calibration subset.

## Recommended Next Evaluation Step

Use the optional API-based LLM judge once a key is available, but run it first on a small subset:

```powershell
.\.venv\Scripts\python.exe -m task4_rag.src.evaluate_llm_quality `
  --run outputs/runs/caes_rag_rrf_v1.jsonl `
  --queries data/task4_longeval_rag-query_docids.jsonl `
  --documents data/snapshot3/longeval_sci_test-09-11_2026_fulltext/documents `
  --doc-text-fields "fullText|abstract|title" `
  --provider openai `
  --model gpt-4o-mini `
  --max-records 5 `
  --output-report outputs/reports/caes_rag_rrf_v1_llm_judge_smoke.json
```

Then compare those LLM judgments against this direct assistant review and the deterministic proxy metrics. If they agree on the same weak records, we have a reliable target for the next pipeline iteration.

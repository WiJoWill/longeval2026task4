# RAG Evaluation Methodology

## Why We Need A Second Evaluator

The original evaluator answers compliance questions:

- Did every query receive one output record?
- Is the JSONL structurally valid?
- Are references a subset of the official candidate `doc_ids`?
- Are citations valid indices into `references`?
- Are there obvious uncited or filler answer items?

Those checks are necessary for Task 4 submission hygiene, but they do not determine whether the RAG system is good. A system can be perfectly valid while citing irrelevant documents and producing unhelpful extractive text.

## Literature-Inspired Targets

The Prompt Engineering Guide summarizes RAG evaluation around retrieval quality and generation quality. It highlights three quality scores:

- context relevance;
- answer faithfulness;
- answer relevance.

It also highlights four robustness abilities from RGB-style RAG evaluation:

- noise robustness;
- negative rejection;
- information integration;
- counterfactual robustness.

RECALL is especially useful as a reminder that external context can be wrong or misleading, so generated claims should be checked against cited context rather than trusted simply because they were retrieved.

## Implemented Metrics

The new CLI is:

```powershell
.\.venv\Scripts\python.exe -m task4_rag.src.evaluate_rag_quality `
  --run outputs/runs/caes_rag_rrf_v1.jsonl `
  --queries data/task4_longeval_rag-query_docids.jsonl `
  --documents data/snapshot3/longeval_sci_test-09-11_2026_fulltext/documents `
  --doc-text-fields "fullText|abstract|title" `
  --output-report outputs/reports/caes_rag_rrf_v1_rag_quality_eval.json
```

The evaluator reports:

- `context_relevance`: average query/document keyword overlap for selected references;
- `context_precision_proxy`: share of selected references that clear a relevance threshold;
- `answer_faithfulness_proxy`: answer/cited-context overlap;
- `answer_relevance_proxy`: answer/query overlap;
- `unsupported_answer_item_rate`: answer items weakly supported by cited context;
- `noise_robustness_proxy`: share of cited references that are not classified as noisy;
- `negative_rejection_success_rate`: rejection behavior when no selected context appears relevant;
- `information_integration_proxy`: whether multi-part queries cite multiple documents;
- `numeric_claim_support_rate`: share of numeric claims found in cited context.

## Current All-Method Evaluation

All current methods were evaluated with the same reference-free RAG proxy methodology, including the raw baseline from `data/generated-responses.jsonl`.

Reports:

- `outputs/reports/caes_rag_rrf_v1_rag_quality_eval.json`
- `outputs/reports/rrf_no_rerank_v1_rag_quality_eval.json`
- `outputs/reports/single_query_bm25_v1_rag_quality_eval.json`
- `outputs/reports/concat_baseline_rag_quality_eval.json`
- `outputs/reports/generated_responses_repaired_v1_rag_quality_eval.json`
- `outputs/reports/generated_responses_raw_rag_quality_eval.json`
- `outputs/reports/rag_quality_all_methods_summary.json`

Current proxy score summary:

| Method | Context relevance | Context precision | Faithfulness | Answer relevance | Unsupported rate | Noise robustness | Numeric support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_generated_responses` | 0.256 | 0.523 | 0.833 | 1.000 | 0.167 | 0.523 | 0.764 |
| `generated_responses_repaired_v1` | 0.256 | 0.523 | 1.000 | 0.047 | 0.000 | 0.460 | 0.553 |
| `concat_baseline` | 0.279 | 0.608 | 0.982 | 0.098 | 0.000 | 0.660 | 0.489 |
| `single_query_bm25_v1` | 0.539 | 0.955 | 0.937 | 0.420 | 0.000 | 0.963 | 0.853 |
| `rrf_no_rerank_v1` | 0.558 | 0.970 | 0.942 | 0.397 | 0.000 | 0.963 | 0.905 |
| `caes_rag_rrf_v1` | 0.558 | 0.970 | 0.942 | 0.392 | 0.000 | 0.963 | 0.879 |

Interpretation:

- The raw baseline has artificially perfect `answer_relevance` because it repeats the query in a filler answer sentence. Its lower faithfulness and nonzero unsupported rate reveal that problem more clearly.
- The repaired baseline is citation-faithful because it mostly turns references into citation-bearing answer items, but it has very weak answer relevance and weak context precision.
- The real retrieval methods clearly improve context relevance, context precision, noise robustness, and numeric support.
- `rrf_no_rerank_v1` and `caes_rag_rrf_v1` are close under these proxies; the full reranked method has slightly lower numeric support in this run, while both share the same context relevance and precision.

## How To Read These Scores

High faithfulness means the answer is mostly grounded in the cited documents. It does not mean the cited documents are the right documents for the query.

Moderate answer relevance is the more important warning in the current run. It matches qualitative inspection: some outputs cite real candidate documents but drift into unrelated scientific topics because retrieval selected semantically weak evidence.

Numeric claim support is a small RECALL-style counterfactual-risk check. It is not a full contradiction detector, but it catches one useful class of unsupported factual claims.

For the raw baseline, answer relevance should not be trusted on its own because the baseline contains boilerplate that repeats the query. Use it together with unsupported rate, noise robustness, and structural validation.

## Recommended Workflow

1. Run structural validation first.
2. Run RAG quality diagnostics second.
3. Sort or inspect records with:
   - low `answer_relevance_proxy`;
   - low `noise_robustness_proxy`;
   - high `counterfactual_risk_proxy`;
   - low `context_relevance`.
4. Build a small judged subset of 8-10 queries.
5. Calibrate these proxy metrics against human labels for document relevance and answer usefulness.

## Optional LLM-As-Judge Evaluation

The repo now also supports a third evaluation layer: an optional LLM judge. This is useful when we want the model to give human-like qualitative and quantitative judgments over all 47 query outputs.

The LLM judge scores each record on a 1-5 scale:

- `context_relevance`;
- `answer_relevance`;
- `faithfulness`;
- `completeness`;
- `citation_quality`;
- `noise_robustness`;
- `information_integration`;
- `numeric_factuality`;
- `overall`.

It also returns qualitative notes:

- strengths;
- weaknesses;
- failure modes;
- recommended fix.

Run a small smoke evaluation first:

```powershell
$env:OPENAI_API_KEY = "<your key>"

.\.venv\Scripts\python.exe -m task4_rag.src.evaluate_llm_quality `
  --run outputs/runs/caes_rag_rrf_v1.jsonl `
  --queries data/task4_longeval_rag-query_docids.jsonl `
  --documents data/snapshot3/longeval_sci_test-09-11_2026_fulltext/documents `
  --doc-text-fields "fullText|abstract|title" `
  --provider openai `
  --model gpt-4o-mini `
  --max-records 3 `
  --output-report outputs/reports/caes_rag_rrf_v1_llm_judge_smoke.json
```

Run all 47 records:

```powershell
.\.venv\Scripts\python.exe -m task4_rag.src.evaluate_llm_quality `
  --run outputs/runs/caes_rag_rrf_v1.jsonl `
  --queries data/task4_longeval_rag-query_docids.jsonl `
  --documents data/snapshot3/longeval_sci_test-09-11_2026_fulltext/documents `
  --doc-text-fields "fullText|abstract|title" `
  --provider openai `
  --model gpt-4o-mini `
  --output-report outputs/reports/caes_rag_rrf_v1_llm_judge_eval.json
```

Current implementation notes:

- GPT/OpenAI judging is implemented through `task4_rag/src/llm_quality_evaluator.py`.
- Claude and Gemini are not implemented yet, but the evaluator is structured around a `JudgeClient` interface so they can be added cleanly.
- No LLM judge report has been generated in this workspace because no `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` is currently set.
- LLM-as-judge scores should be treated as a calibrated review aid, not official ground truth. They are strongest when combined with the deterministic proxy metrics and a small human-labeled subset.

Because no API key is currently available, there is also a direct assistant-side review:

- `outputs/reports/direct_assistant_llm_style_evaluation.md`

That report uses the same 1-5 rubric, the 47-query proxy summaries, and direct qualitative inspection. It is useful as an immediate review, but it is not reproducible in the same way as an API-based judge run.

## Next Pipeline Refinements

The quality evaluator points to retrieval and reranking as the next bottleneck:

- downweight generic terms such as `class`, `study`, `model`, and `system`;
- require better agreement between title relevance and passage relevance;
- add sentence-level reranking after document selection;
- add a query/document diagnostic file with selected titles and relevance proxy scores;
- optionally add an LLM judge later for answer relevance and faithfulness, but only after structural validation is stable.

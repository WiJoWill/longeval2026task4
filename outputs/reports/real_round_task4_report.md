# Real Retrieval-Generation Round Report

## Run Status

This round implements the first two recommended improvements from the previous report:

1. tighter sentence filtering in generation;
2. a better deterministic sentence selector.

The generator now filters boilerplate scientific openers, low-information phrases, OCR-like fragments, broken full-text snippets, and sentences without direct or stemmed query overlap unless the query has temporal intent. It also prefers one strong sentence per selected document before filling with additional sentences, and it uses a cited fallback from selected evidence instead of emitting uncited fallback text.

All models and ablations were rerun on the real 47-query Task 4 file:

- `data/task4_longeval_rag-query_docids.jsonl`

## Environment And Validation

- Python: `3.11.9`
- Test command: `python -m pytest task4_rag/tests`
- Test result: `12 passed`
- Main run validation: `OK`

## Main Real Run

Primary run:

- `outputs/runs/caes_rag_rrf_v1.jsonl`

Evaluation:

- `outputs/reports/caes_rag_rrf_v1_eval.json`

Metrics:

- expected queries: `47`
- records in run: `47`
- missing queries: `0`
- invalid records: `0`
- `reference_subset_match_rate`: `1.0`
- avg references per record: `3.26`
- avg answer items per record: `4.34`
- avg unique cited references per record: `2.68`
- empty-citation rate: `0.0`
- filler-answer rate: `0.0147`

RAG quality diagnostics:

- `outputs/reports/caes_rag_rrf_v1_rag_quality_eval.json`

Proxy metrics:

- context relevance: `0.558`
- context precision proxy: `0.970`
- answer faithfulness proxy: `0.942`
- answer relevance proxy: `0.392`
- unsupported answer item rate: `0.0`
- RGB-like noise robustness proxy: `0.963`
- RGB-like information integration coverage: `0.255`
- RECALL-like numeric claim support rate: `0.879`

Interpretation:

- every output record is structurally valid;
- every `references` list is a subset of the official candidate document IDs;
- the run remains compact, using about one third of the 10 candidates per query;
- the generation pass is stricter than before, so fewer weak sentences are emitted;
- citation faithfulness is high because answers are extractive, but answer relevance is much lower, which better captures the remaining semantic drift problem.

`reference_exact_match_rate` remains `0.0` by design because this system reports only selected evidence rather than all 10 candidate documents.

## Ablation Results

| Run | Invalid | Subset match | Avg refs | Avg answer items | Avg cited refs | Empty citation rate | Filler rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `generated_responses_repaired_v1` | 0 | 1.000 | 10.00 | 5.00 | 5.00 | 0.000 | 0.000 |
| `concat_baseline` | 0 | 1.000 | 2.28 | 2.13 | 1.19 | 0.000 | 0.000 |
| `single_query_bm25_v1` | 0 | 1.000 | 3.68 | 4.40 | 3.04 | 0.000 | 0.019 |
| `rrf_no_rerank_v1` | 0 | 1.000 | 3.23 | 4.32 | 2.68 | 0.000 | 0.025 |
| `caes_rag_rrf_v1` | 0 | 1.000 | 3.26 | 4.34 | 2.68 | 0.000 | 0.015 |

The raw provided `data/generated-responses.jsonl` was also re-analyzed:

- invalid records: `47`
- exact candidate reference match: `1.0`
- empty-citation rate: `0.1667`
- filler-answer rate: `0.1667`

The repaired baseline is structurally clean, but it keeps all 10 candidate references and is therefore not an evidence-selection run.

## Output Quality Reading

The stricter generator improved structural grounding and reduced answer verbosity. The main tradeoff is that answer-item counts are lower for some queries because noisy or off-topic sentences are now dropped instead of being forced into the answer.

Qualitative smoke review still shows a retrieval-side weakness: broad query terms such as `class` can pull unrelated candidate documents when those terms appear in titles or full text. The generator can suppress some bad sentences, but it cannot fully repair evidence selection once an unrelated document is ranked into the selected set.

The new quality evaluator makes this clearer than the structural evaluator. Structural metrics say the run is valid and citation-compliant. RAG-quality metrics separate that from answer usefulness: faithfulness-to-cited-context is strong, while answer relevance is only moderate.

So the current state is:

- structurally solid;
- candidate-compliant;
- more conservative in generation;
- still vulnerable to semantic drift in retrieval and reranking.

## Recommended Next Improvements

1. Improve document-level evidence selection:
   - downweight generic query terms such as `class`, `system`, `model`, and `study`;
   - require stronger title/body agreement before a document contributes answer evidence;
   - add a per-query diagnostic for selected document titles.

2. Add sentence-level reranking:
   - score sentences directly against the query after passage selection;
   - penalize citation candidates from documents whose titles have low query specificity;
   - preserve the current citation-safe fallback.

3. Add a small qualitative review set:
   - inspect 8-10 representative queries;
   - label selected documents as relevant, partially relevant, or off-topic;
   - use that as a fast regression suite for retrieval changes.

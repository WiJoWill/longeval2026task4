# First-Round Task 4 Report

## Summary

This first round focused on getting the local environment working, validating the Task 4 tooling on the real 47-query file, evaluating the provided baseline sample, and producing a repaired submission-shaped baseline run.

At the end of this round:

- the local `.venv` is set up and usable;
- the Task 4 test suite passes;
- the real Task 4 query file is now the default input in the configs;
- the provided `generated-responses.jsonl` baseline has been evaluated;
- a repaired baseline submission was generated and validated;
- a true evidence-grounded 47-query generation run is still blocked by corpus-ID mismatch.

## Environment

Local virtual environment:

- Python: `3.11.9`
- installed for this round: `PyYAML`, `pytest`, `openai`

Validation:

- test suite result: `10 passed`

## Real Task 4 Query Set

Primary input file:

- `data/task4_longeval_rag-query_docids.jsonl`

Observed query count:

- `47`

The sample files `data/query_docids.jsonl`, `data/answers.jsonl`, and `data/submission.jsonl` were treated as format examples only, not as the real evaluation set.

## Baseline Input Used

Provided baseline file:

- `data/generated-responses.jsonl`

This file was treated as the first-round baseline because it covers the real 47 Task 4 queries and already includes the official candidate `references` per query.

## Baseline Evaluation: Raw Provided Run

Evaluation artifact:

- `outputs/reports/generated_responses_raw_eval.json`

Main findings:

- expected queries: `47`
- records in run: `47`
- missing queries: `0`
- invalid records: `47`
- reference exact match rate: `1.0`
- reference set match rate: `1.0`
- avg references per record: `10.0`
- avg answer items per record: `12.0`
- answer items without citations: `94`
- empty-citation rate: `0.1667`
- filler answer items: `94`
- filler rate: `0.1667`

Main failure mode:

- every record is invalid because `metadata.narrative` is missing

Interpretation:

The provided baseline is useful as a floor because it preserves the official candidate reference lists, but it is not a valid Task 4 submission and it contains obvious non-answer filler plus uncited answer items.

## Repaired Baseline Run

Generated output:

- `outputs/runs/generated_responses_repaired_v1.jsonl`

Repair logic:

- fill `metadata.narrative` from the real Task 4 query file;
- keep `narrative_id`;
- set `team_id=our_team`;
- set `run_id=generated_responses_repaired_v1`;
- remove filler answer items such as preambles;
- remove uncited answer items;
- keep at most 5 cited answer items per query;
- preserve the provided candidate `references`.

Validation result:

- `OK`

## Baseline Evaluation: Repaired Run

Evaluation artifact:

- `outputs/reports/generated_responses_repaired_eval.json`

Main findings:

- expected queries: `47`
- records in run: `47`
- missing queries: `0`
- invalid records: `0`
- reference exact match rate: `1.0`
- reference set match rate: `1.0`
- avg references per record: `10.0`
- avg answer items per record: `5.0`
- answer items without citations: `0`
- empty-citation rate: `0.0`
- filler answer items: `0`
- filler rate: `0.0`

Interpretation:

The repaired run is submission-shaped and structurally valid. It is still only a formatting/cleanup baseline, not a truly grounded scientific RAG system, because the remaining answer content is mostly inherited from the provided baseline and not regenerated from aligned document evidence.

## Corpus Blocker

A true first-round RAG run over the 47 real queries was not possible with the current corpus path:

- configured corpus path: `data/snapshot3/longeval_sci_test-09-11_2026_fulltext/documents`

Observed issue:

- sample Task 4 candidate IDs such as `275699672` and `303362068` were not found anywhere under `data/snapshot3`

Impact:

- the system cannot load the candidate documents for the real Task 4 query set;
- evidence retrieval, reranking, and grounded generation cannot yet run on the real 47-query task;
- any claim-generation run against the current snapshot would be misleading.

## What Was Prepared For The Next Round

The repo now includes an experiment matrix for:

- concat baseline
- single-query BM25
- RRF without reranking
- RRF with reranking

Relevant configs:

- `task4_rag/configs/task4_concat_baseline.yaml`
- `task4_rag/configs/task4_single_query_bm25.yaml`
- `task4_rag/configs/task4_rrf_no_rerank.yaml`
- `task4_rag/configs/task4_rrf_rerank.yaml`

These are ready once the aligned corpus is available.

## Recommendation

The next highest-value step is to obtain or point the pipeline at the document snapshot whose `id` values match the real file `data/task4_longeval_rag-query_docids.jsonl`.

Once that alignment is fixed, the next round should be:

1. run `single_query_bm25_v1`;
2. run `rrf_no_rerank_v1`;
3. run `caes_rag_rrf_v1`;
4. compare retrieval/evidence and citation-valid output statistics across the three runs;
5. only then consider optional LLM-assisted judging for pairwise comparison.

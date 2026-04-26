# LongEval 2026 Task 4 RAG Methodology

This repository contains a reproducible pipeline for CLEF LongEval 2026 Task 4, LongEval-RAG. The task setup provides a query plus a fixed set of candidate document IDs. The system must generate an answer grounded only in those candidate documents and cite references by index.

The primary run is `caes_rag_rrf_v1`: Citation-Aware Evidence Selection with reciprocal-rank fusion and deterministic extractive generation.

## Method Summary

For each query, the pipeline:

1. loads the query text and official candidate `doc_ids`;
2. loads matching full-text or abstract records from the configured snapshot;
3. splits candidate documents into sentence-window passages;
4. builds deterministic query variants from the original narrative;
5. retrieves candidate passages with BM25-style lexical scoring;
6. fuses multi-query rankings with reciprocal-rank fusion;
7. reranks evidence using lexical overlap, title relevance, temporal cues, and an optional citation-graph prior;
8. selects a compact evidence set;
9. generates citation-indexed answer sentences from selected evidence;
10. validates the output JSONL structure and analyzes run metrics.

The run is candidate-constrained throughout: output `references` must be a subset of the official candidate IDs for that query, and each answer citation must point to an index in that `references` list.

## Current Improvements

The latest round implements three generation improvements:

- stricter sentence filtering for boilerplate, low-information phrases, OCR-like fragments, and broken full-text snippets;
- stronger query-term overlap using light stemming, so sentences must match the narrative more directly;
- a sentence selector that takes the best sentence per selected document before filling with extra claims, plus a cited fallback when the strict filter finds too little clean text.

## Main Artifacts

- Main run: `outputs/runs/caes_rag_rrf_v1.jsonl`
- Main evaluation: `outputs/reports/caes_rag_rrf_v1_eval.json`
- Summary report: `outputs/reports/real_round_task4_report.md`
- Repaired provided baseline: `outputs/runs/generated_responses_repaired_v1.jsonl`
- Raw baseline analysis: `outputs/reports/generated_responses_analysis.json`

## Latest Main Metrics

From `outputs/reports/caes_rag_rrf_v1_eval.json`:

- expected queries: `47`
- records in run: `47`
- invalid records: `0`
- missing queries: `0`
- reference subset match rate: `1.0`
- average references per record: `3.26`
- average answer items per record: `4.34`
- average unique cited references per record: `2.68`
- empty citation rate: `0.0`
- filler answer rate: `0.0147`

## RAG Quality Evaluation

The original evaluator is a structural compliance check. It verifies coverage, JSON validity, candidate-subset compliance, citation indices, and obvious filler. Those checks are necessary, but they do not measure RAG quality by themselves.

The repo now includes a second evaluator inspired by the RAG evaluation framing in the Prompt Engineering Guide, RGB, RECALL, and RAGAS:

- context relevance: whether selected references overlap with the query;
- answer faithfulness: whether answer sentences are supported by their cited documents;
- answer relevance: whether generated answers address the query;
- RGB-like robustness: noise robustness, negative rejection, and information integration proxies;
- RECALL-like counterfactual risk: whether numeric claims in answers are supported by cited context.

Run it with:

```powershell
.\.venv\Scripts\python.exe -m task4_rag.src.evaluate_rag_quality `
  --run outputs/runs/caes_rag_rrf_v1.jsonl `
  --queries data/task4_longeval_rag-query_docids.jsonl `
  --documents data/snapshot3/longeval_sci_test-09-11_2026_fulltext/documents `
  --doc-text-fields "fullText|abstract|title" `
  --output-report outputs/reports/caes_rag_rrf_v1_rag_quality_eval.json
```

Current main quality report:

- `outputs/reports/caes_rag_rrf_v1_rag_quality_eval.json`

Current proxy scores:

- context relevance: `0.558`
- context precision proxy: `0.970`
- answer faithfulness proxy: `0.942`
- answer relevance proxy: `0.392`
- unsupported answer item rate: `0.0`
- RGB-like noise robustness proxy: `0.963`
- RGB-like information integration coverage: `0.255`
- RECALL-like numeric claim support rate: `0.879`

These are diagnostic proxies, not official labels. They are most useful for comparing runs and surfacing suspicious records. In particular, lexical context relevance can overestimate semantic relevance when noisy full-text documents share generic terms with the query.

## Optional LLM Judge

For a human-like qualitative layer, the repo includes an optional LLM-as-judge evaluator:

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

It scores each query on 1-5 dimensions: context relevance, answer relevance, faithfulness, completeness, citation quality, noise robustness, information integration, numeric factuality, and overall quality. It also writes qualitative strengths, weaknesses, failure modes, and a recommended fix per query.

No LLM judge report is currently generated in this workspace because the API keys are not set.

## Repository Layout

- `data/`: task inputs, candidate mappings, baseline files, and document snapshots.
- `task4_rag/src/`: loader, preprocessing, retrieval, reranking, generation, validation, and evaluation code.
- `task4_rag/configs/`: experiment configs for the main method and ablations.
- `task4_rag/tests/`: unit tests for schema loading, generation, validation, and evaluation behavior.
- `outputs/runs/`: generated Task 4 JSONL runs.
- `outputs/reports/`: evaluation JSON files and Markdown reports.

## Setup

Use the existing virtual environment or create a new one with Python 3.11.

```powershell
.\.venv\Scripts\python.exe -m pip install -r task4_rag\requirements.txt
```

## Run The Main Model

```powershell
.\.venv\Scripts\python.exe -m task4_rag.src.run_task4 `
  --config task4_rag/configs/task4_rrf_rerank.yaml `
  --output outputs/runs/caes_rag_rrf_v1.jsonl
```

## Run All Ablations

```powershell
.\.venv\Scripts\python.exe -m task4_rag.src.run_task4 --config task4_rag/configs/task4_concat_baseline.yaml --output outputs/runs/concat_baseline.jsonl
.\.venv\Scripts\python.exe -m task4_rag.src.run_task4 --config task4_rag/configs/task4_single_query_bm25.yaml --output outputs/runs/single_query_bm25_v1.jsonl
.\.venv\Scripts\python.exe -m task4_rag.src.run_task4 --config task4_rag/configs/task4_rrf_no_rerank.yaml --output outputs/runs/rrf_no_rerank_v1.jsonl
.\.venv\Scripts\python.exe -m task4_rag.src.run_task4 --config task4_rag/configs/task4_rrf_rerank.yaml --output outputs/runs/caes_rag_rrf_v1.jsonl
```

## Evaluate A Run

```powershell
.\.venv\Scripts\python.exe -m task4_rag.src.evaluate_run `
  --run outputs/runs/caes_rag_rrf_v1.jsonl `
  --queries data/task4_longeval_rag-query_docids.jsonl `
  --output-report outputs/reports/caes_rag_rrf_v1_eval.json
```

## Repair And Evaluate The Provided Baseline

```powershell
.\.venv\Scripts\python.exe -m task4_rag.src.repair_baseline_run `
  --input data/generated-responses.jsonl `
  --queries data/task4_longeval_rag-query_docids.jsonl `
  --output outputs/runs/generated_responses_repaired_v1.jsonl
```

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest task4_rag/tests
```

Current result: `12 passed`.

## Notes And Limitations

The pipeline is structurally valid and candidate-compliant, but semantic quality is still retrieval-limited. Some broad query terms can pull unrelated candidate documents, so the next high-value work is document-level reranking and a small qualitative review set.

The evaluation workflow should now be:

1. validate structure and candidate compliance;
2. run RAG quality diagnostics;
3. inspect records with low answer relevance, low noise robustness, or high counterfactual risk;
4. use a small judged subset to calibrate whether the proxy metrics match human relevance judgments.

Useful references:

- Prompt Engineering Guide RAG evaluation overview: https://www.promptingguide.ai/research/rag#rag-evaluation
- RGB benchmark: https://arxiv.org/abs/2309.01431
- RECALL benchmark: https://arxiv.org/abs/2311.08147
- RAGAS: https://arxiv.org/abs/2309.15217

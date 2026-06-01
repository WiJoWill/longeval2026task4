# LongEval 2026 Task 4 RAG Repository

This repository contains the code, configs, evaluation artifacts, and writing notes for CLEF LongEval 2026 Task 4.

The runnable package lives in [`task4_rag/`](task4_rag/). Generated runs and reports live under `outputs/`. Paper and evaluation scratch artifacts live under `paper_writing/`, `docs/`, and `evals/`.

## Repository Layout

- [`task4_rag/src/`](task4_rag/src/) - core pipeline, validators, evaluators, and CLIs.
- [`task4_rag/configs/`](task4_rag/configs/) - experiment configurations.
- [`task4_rag/scripts/`](task4_rag/scripts/) - shell and PowerShell wrappers for common workflows.
- [`task4_rag/tests/`](task4_rag/tests/) - unit tests and fixtures.
- [`task4_rag/examples/`](task4_rag/examples/) - legacy examples and sample inputs/outputs for older CLIs.
- [`docs/`](docs/) - design notes and cleanup plans.
- [`evals/`](evals/) - evaluation outputs and LaTeX scratch artifacts.
- [`outputs/`](outputs/) - generated runs, batch inputs, reports, and smoke-test artifacts.
- [`paper_writing/`](paper_writing/) - report drafts and judge summaries.
- [`Final_paper/`](Final_paper/) - paper PDF deliverable.
- `.cache/` - local dataset cache used by some scripts; not committed.
- `data/` - expected local checkout for task inputs and snapshots; not committed.

## What The Main Pipeline Does

The main run is `caes_rag_rrf_v1`. It loads the query text and official candidate document IDs, retrieves passages over the candidate set, reranks evidence, generates citation-indexed answers, and validates the resulting JSONL.

## Setup

Install dependencies with the existing virtual environment or a fresh Python 3.11 environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r task4_rag\requirements.txt
```

If you use a different virtual environment location, adjust the path accordingly.

## Common Commands

Run the main model:

```powershell
.\.venv\Scripts\python.exe -m task4_rag.src.run_task4 `
  --config task4_rag/configs/task4_rrf_rerank.yaml `
  --output outputs/runs/caes_rag_rrf_v1.jsonl
```

Validate a run:

```powershell
.\.venv\Scripts\python.exe -m task4_rag.src.evaluate_run `
  --run outputs/runs/caes_rag_rrf_v1.jsonl `
  --queries data/task4_longeval_rag-query_docids.jsonl `
  --output-report outputs/reports/caes_rag_rrf_v1_eval.json
```

Run the RAG quality diagnostics:

```powershell
.\.venv\Scripts\python.exe -m task4_rag.src.evaluate_rag_quality `
  --run outputs/runs/caes_rag_rrf_v1.jsonl `
  --queries data/task4_longeval_rag-query_docids.jsonl `
  --documents data/snapshot3/longeval_sci_test-09-11_2026_fulltext/documents `
  --doc-text-fields "fullText|abstract|title" `
  --output-report outputs/reports/caes_rag_rrf_v1_rag_quality_eval.json
```

Repair the provided baseline run:

```powershell
.\.venv\Scripts\python.exe -m task4_rag.src.repair_baseline_run `
  --input data/generated-responses.jsonl `
  --queries data/task4_longeval_rag-query_docids.jsonl `
  --output outputs/runs/generated_responses_repaired_v1.jsonl
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest task4_rag/tests
```

## Inputs And Outputs

- Expected inputs: query/doc mapping files and document snapshots under `data/` or `.cache/`, depending on the workflow.
- Main outputs: JSONL runs under `outputs/runs/` and evaluation JSON/Markdown under `outputs/reports/`.
- Batch workflows: request payloads and state files under `outputs/batch_inputs/` and `outputs/test/`.

## Notes

- The repo keeps a few legacy or auxiliary utilities in `task4_rag/src/` and `task4_rag/examples/` for historical workflows. They are not part of the default `run_task4` path.
- If you are looking for implementation details, the package README in [`task4_rag/README.md`](task4_rag/README.md) is the deeper reference.

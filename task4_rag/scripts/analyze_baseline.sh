#!/usr/bin/env bash
set -euo pipefail

python -m task4_rag.src.evaluate_run \
  --run "${RUN_PATH:-data/generated-responses.jsonl}" \
  --queries "${QUERIES_PATH:-data/task4_longeval_rag-query_docids.jsonl}" \
  --output-report "${REPORT_PATH:-outputs/reports/generated_responses_analysis.json}"

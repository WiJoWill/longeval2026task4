#!/usr/bin/env bash
set -euo pipefail

python -m task4_rag.src.run_task4 \
  --config task4_rag/configs/task4_default.yaml \
  --mode hybrid_evidence_rag_v1 \
  --queries "${QUERIES_PATH:-data/query_docids.jsonl}" \
  --documents "${DOCUMENTS_PATH:-data/snapshot3/longeval_sci_test-09-11_2026_fulltext/documents}" \
  --output "${OUTPUT_PATH:-runs/caes_rag_rrf_v1.jsonl}"

#!/usr/bin/env bash
set -euo pipefail

python -m task4_rag.src.run_task4 \
  --config task4_rag/configs/task4_default.yaml \
  --validate-only \
  --output "${1:-runs/caes_rag_rrf_v1.jsonl}"

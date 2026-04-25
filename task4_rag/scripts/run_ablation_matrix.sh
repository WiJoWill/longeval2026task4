#!/usr/bin/env bash
set -euo pipefail

python -m task4_rag.src.run_task4 \
  --config task4_rag/configs/task4_concat_baseline.yaml \
  --output "${OUTPUT_DIR:-runs}/concat_baseline.jsonl"

python -m task4_rag.src.run_task4 \
  --config task4_rag/configs/task4_single_query_bm25.yaml \
  --output "${OUTPUT_DIR:-runs}/single_query_bm25_v1.jsonl"

python -m task4_rag.src.run_task4 \
  --config task4_rag/configs/task4_rrf_no_rerank.yaml \
  --output "${OUTPUT_DIR:-runs}/rrf_no_rerank_v1.jsonl"

python -m task4_rag.src.run_task4 \
  --config task4_rag/configs/task4_rrf_rerank.yaml \
  --output "${OUTPUT_DIR:-runs}/caes_rag_rrf_v1.jsonl"

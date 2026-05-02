"""Prepare OpenAI Batch API input files for all 10 LongEval-RAG experiments.

Phase 1 of the two-phase batch workflow:
  1. Run retrieval + evidence selection for every query in every experiment.
  2. Write one batch-input JSONL  ->outputs/batch_inputs/{run_id}_requests.jsonl
  3. Write one state JSON         ->outputs/batch_inputs/{run_id}_state.json
     The state file maps custom_id ->(sentence_candidates, references, metadata)
     so that phase 2 (apply_batch.py) can reconstruct citations from evidence_ids.

Usage:
  python -m task4_rag.src.prepare_batch --model gpt-5.5-mini [--queries ...] [--documents ...] [--max-queries N] [--output-dir outputs/batch_inputs]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .data_loader import load_task
from .evidence_ranker import EvidenceRanker, build_reference_list
from .generator import AnswerGenerator, GenerationConfig
from .preprocess import split_documents_into_passages
from .run_task4 import (
    _load_local_env,
    _optional_int,
    _parse_doc_text_fields,
    apply_cli_overrides,
    load_config,
)


# Mirrors the experiment list in run_openai_llm_experiments.ps1.
EXPERIMENTS = [
    {"config": "task4_rag/configs/task4_concat_baseline.yaml",         "run_id": "concat_baseline_openai_llm_v1"},
    {"config": "task4_rag/configs/task4_single_query_bm25.yaml",       "run_id": "single_query_bm25_openai_llm_v1"},
    {"config": "task4_rag/configs/task4_rrf_no_rerank.yaml",           "run_id": "rrf_no_rerank_openai_llm_v1"},
    {"config": "task4_rag/configs/task4_rrf_rerank.yaml",              "run_id": "caes_rag_rrf_openai_llm_v1"},
    {"config": "task4_rag/configs/task4_rule_minilm_rerank.yaml",      "run_id": "rule_minilm_openai_llm_v1"},
    {"config": "task4_rag/configs/task4_semantic_current_rerank.yaml", "run_id": "semantic_current_openai_llm_v1"},
    {"config": "task4_rag/configs/task4_topic_shift_current_rerank.yaml", "run_id": "topic_shift_current_openai_llm_v1"},
    {"config": "task4_rag/configs/task4_semantic_minilm_rerank.yaml",  "run_id": "semantic_minilm_openai_llm_v1"},
    {"config": "task4_rag/configs/task4_topic_shift_minilm_rerank.yaml", "run_id": "topic_shift_minilm_openai_llm_v1"},
    {"config": "task4_rag/configs/task4_default.yaml",                 "run_id": "default_openai_llm_v1"},
]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _load_local_env()

    model = args.model or os.getenv("OPENAI_MODEL", "gpt-5.5-mini")
    if not model:
        print("Error: --model is required (or set OPENAI_MODEL in .env)", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_config = _apply_data_overrides(load_config(EXPERIMENTS[0]["config"]), args)
    instances = load_task(
        queries_path=first_config["data"]["queries_path"],
        documents_path=first_config["data"]["documents_path"],
        candidates_path=first_config["data"].get("candidates_path"),
        candidate_limit=first_config["data"].get("candidate_limit"),
        max_queries=first_config["data"].get("max_queries"),
        document_text_fields=_parse_doc_text_fields(first_config["data"].get("doc_text_fields")),
    )

    total_requests = 0
    all_requests: list[dict[str, Any]] = []
    passage_cache: dict[tuple[Any, ...], list[Any]] = {}

    for exp in EXPERIMENTS:
        run_id = exp["run_id"]
        print(f"\n{'='*60}")
        print(f"Preparing batch for: {run_id}")
        print(f"  config:  {exp['config']}")

        config = load_config(exp["config"])
        # Inject CLI overrides for queries / documents / max_queries.
        config = _apply_data_overrides(config, args)
        # Force run_id to the batch variant name.
        config["run"]["run_id"] = run_id

        n_requests, requests_path, state_path, exp_requests = _prepare_one_experiment(
            config=config,
            run_id=run_id,
            model=model,
            output_dir=output_dir,
            instances=instances,
            passage_cache=passage_cache,
        )
        total_requests += n_requests
        all_requests.extend(exp_requests)
        print(f"  wrote {n_requests} requests  ->{requests_path}")
        print(f"  wrote state              ->{state_path}")

    merged_path = output_dir / "all_experiments_requests.jsonl"
    with merged_path.open("w", encoding="utf-8") as fh:
        for req in all_requests:
            fh.write(json.dumps(req, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Done. Total requests: {total_requests}")
    print(f"Merged file for one-shot submission ->{merged_path}")
    print(f"Submit with:  python -m task4_rag.src.apply_batch --requests-file {merged_path} --state-dir {output_dir} --output-dir <runs_dir>")
    return 0


def _prepare_one_experiment(
    config: dict[str, Any],
    run_id: str,
    model: str,
    output_dir: Path,
    instances: list[Any],
    passage_cache: dict[tuple[Any, ...], list[Any]],
) -> tuple[int, Path, Path, list[dict[str, Any]]]:
    mode = config["run"]["mode"]

    ranker = EvidenceRanker(
        method=config["retrieval"]["method"],
        dense_model=config["retrieval"].get("dense_model"),
        hybrid_alpha=float(config["retrieval"].get("hybrid_alpha", 0.25)),
        per_doc_limit=int(config["retrieval"].get("per_doc_limit", 2)),
        rrf_k=int(config["retrieval"].get("rrf_k", 60)),
        enable_query_expansion=bool(config["retrieval"].get("enable_query_expansion", True)),
        enable_prf=bool(config["retrieval"].get("enable_prf", True)),
        prf_top_passages=int(config["retrieval"].get("prf_top_passages", 4)),
        preselect_k=int(config["retrieval"].get("preselect_k", 24)),
        rerank_top_n=int(config["retrieval"].get("rerank_top_n", 24)),
        enable_rerank=bool(config["retrieval"].get("enable_rerank", True)),
        title_boost=float(config["retrieval"].get("title_boost", 0.15)),
        temporal_boost=float(config["retrieval"].get("temporal_boost", 0.1)),
        citation_graph_path=config["data"].get("citation_graph_path"),
        citation_graph_boost=float(config["retrieval"].get("citation_graph_boost", 0.05)),
        reranker_model=config["retrieval"].get("reranker_model"),
        query_variant_limit=int(config["retrieval"].get("query_variant_limit", 5)),
    )

    generator = AnswerGenerator(
        GenerationConfig(
            team_id=config["run"]["team_id"],
            run_id=run_id,
            run_type=config["run"].get("type", "automatic"),
            mode=mode,
            max_answer_sentences=int(config["generation"].get("max_answer_sentences", 5)),
            provider="openai",
            model=model,
            temperature=float(config["generation"].get("temperature", 0.0)),
            prompts_dir=config["generation"].get("prompts_dir"),
            temporal_templates=bool(config["generation"].get("temporal_templates", True)),
            sentence_rerank_model=str(config["generation"].get("sentence_rerank_model") or ""),
            sentence_rerank_top_n=_optional_int(config["generation"].get("sentence_rerank_top_n")) or 24,
            answer_candidate_score_threshold=float(config["generation"].get("answer_candidate_score_threshold", 0.0)),
            answer_candidate_score_margin=float(config["generation"].get("answer_candidate_score_margin", 0.35)),
            max_selected_answer_candidates=_optional_int(config["generation"].get("max_selected_answer_candidates")) or 50,
            multi_doc_synthesis=bool(config["generation"].get("multi_doc_synthesis", True)),
            max_synthesis_sentences=_optional_int(config["generation"].get("max_synthesis_sentences")) or 2,
        )
    )

    top_k = int(config["retrieval"].get("top_k_passages", 8))
    batch_requests: list[dict[str, Any]] = []
    state_index: dict[str, Any] = {}

    for instance in instances:
        cache_key = _passage_cache_key(config, instance.query.query_id)
        passages = passage_cache.get(cache_key)
        if passages is None:
            passages = split_documents_into_passages(
                instance.documents,
                max_words=int(config["preprocess"].get("passage_max_words", 140)),
                stride_sentences=int(config["preprocess"].get("passage_stride_sentences", 1)),
                chunk_mode=str(config["preprocess"].get("chunk_mode", "rule")),
                semantic_chunk_model=config["preprocess"].get("semantic_chunk_model"),
                semantic_merge_threshold=float(config["preprocess"].get("semantic_merge_threshold", 0.72)),
                topic_shift_model=config["preprocess"].get("topic_shift_model"),
                topic_shift_boundary_threshold=float(config["preprocess"].get("topic_shift_boundary_threshold", 0.18)),
                min_sentences_per_chunk=int(config["preprocess"].get("min_sentences_per_chunk", 1)),
                max_sentences_per_chunk=_optional_int(config["preprocess"].get("max_sentences_per_chunk")),
            )
            passage_cache[cache_key] = passages

        if mode == "concat_baseline":
            selected = list(passages)[:top_k]
            references: list[str] = []
            for p in selected:
                if p.doc_id not in references:
                    references.append(p.doc_id)
            working_passages = selected
        elif mode == "llm_all_docs":
            references = []
            for p in passages:
                if p.doc_id not in references:
                    references.append(p.doc_id)
            working_passages = list(passages)
        else:
            evidence = ranker.select(instance.query.text, passages, top_k=top_k)
            references = build_reference_list(evidence)
            working_passages = [item.passage for item in evidence]

        custom_id = f"{run_id}|{instance.query.query_id}"
        request, state = generator.build_batch_request(
            query=instance.query,
            references=references,
            passages=working_passages,
            model=model,
            custom_id=custom_id,
        )
        batch_requests.append(request)
        state_index[custom_id] = state

    requests_path = output_dir / f"{run_id}_requests.jsonl"
    state_path = output_dir / f"{run_id}_state.json"

    with requests_path.open("w", encoding="utf-8") as fh:
        for req in batch_requests:
            fh.write(json.dumps(req, ensure_ascii=False) + "\n")

    with state_path.open("w", encoding="utf-8") as fh:
        json.dump({"run_id": run_id, "model": model, "queries": state_index}, fh, ensure_ascii=False, indent=2)

    return len(batch_requests), requests_path, state_path, batch_requests


def _passage_cache_key(config: dict[str, Any], query_id: str) -> tuple[Any, ...]:
    preprocess = config["preprocess"]
    return (
        query_id,
        str(preprocess.get("chunk_mode", "rule")),
        int(preprocess.get("passage_max_words", 140)),
        int(preprocess.get("passage_stride_sentences", 1)),
        preprocess.get("semantic_chunk_model"),
        float(preprocess.get("semantic_merge_threshold", 0.72)),
        preprocess.get("topic_shift_model"),
        float(preprocess.get("topic_shift_boundary_threshold", 0.18)),
        int(preprocess.get("min_sentences_per_chunk", 1)),
        _optional_int(preprocess.get("max_sentences_per_chunk")),
    )


def _apply_data_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    config.setdefault("data", {})
    if args.queries:
        config["data"]["queries_path"] = args.queries
    if args.documents:
        config["data"]["documents_path"] = args.documents
    if args.candidates:
        config["data"]["candidates_path"] = args.candidates
    if args.max_queries is not None:
        config["data"]["max_queries"] = args.max_queries
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare OpenAI Batch API input JSONL files for all 10 LongEval-RAG experiments."
    )
    parser.add_argument("--model", help="OpenAI model name. Defaults to gpt-5.5-mini or OPENAI_MODEL env var.")
    parser.add_argument("--queries", default=".cache/longeval-sci-2026/task4_longeval_rag-query_docids.jsonl", help="Override queries JSONL path for all experiments")
    parser.add_argument(
        "--documents",
        default=(
            ".cache/longeval-sci-2026/longeval_sci_test-09-11_2026_fulltext/data/processed/"
            "doc_collection_parallel_09032026_parallel_2/snapshot-3/"
            "longeval_sci_test-09-11_2026_fulltext/documents"
        ),
        help="Override documents directory for all experiments",
    )
    parser.add_argument("--candidates", help="Override candidates path for all experiments")
    parser.add_argument("--max-queries", type=int, help="Limit number of queries per experiment (for testing)")
    parser.add_argument("--output-dir", default="outputs/batch_inputs", help="Directory for output files (default: outputs/batch_inputs)")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

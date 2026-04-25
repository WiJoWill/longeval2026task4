"""Command-line entry point for LongEval-RAG Task 4 runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .data_loader import load_task
from .evidence_ranker import EvidenceRanker
from .generator import AnswerGenerator, GenerationConfig
from .output_writer import write_jsonl
from .preprocess import split_documents_into_passages
from .validator import validate_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    mode = config["run"]["mode"]

    if args.validate_only:
        errors = validate_jsonl(args.output)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(f"OK: {args.output}")
        return 0

    instances = load_task(
        queries_path=config["data"]["queries_path"],
        documents_path=config["data"]["documents_path"],
        candidates_path=config["data"].get("candidates_path"),
        candidate_limit=config["data"].get("candidate_limit"),
        max_queries=config["data"].get("max_queries"),
        document_text_fields=_parse_doc_text_fields(config["data"].get("doc_text_fields")),
    )
    total_found_docs = sum(len(instance.documents) for instance in instances)
    total_missing_docs = sum(len(instance.missing_candidate_ids) for instance in instances)
    if total_found_docs == 0 and total_missing_docs > 0:
        raise ValueError(
            "No candidate documents were found in the configured corpus path. "
            "Check that queries/candidate doc IDs match the document snapshot and ID field."
        )
    if total_missing_docs > 0:
        print(
            f"Warning: missing {total_missing_docs} candidate documents across {len(instances)} queries.",
            file=sys.stderr,
        )

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
            run_id=config["run"]["run_id"],
            run_type=config["run"].get("type", "automatic"),
            mode=mode,
            max_answer_sentences=int(config["generation"].get("max_answer_sentences", 5)),
            provider=config["generation"].get("provider", "none"),
            model=config["generation"].get("model", ""),
            temperature=float(config["generation"].get("temperature", 0.0)),
            prompts_dir=config["generation"].get("prompts_dir"),
            temporal_templates=bool(config["generation"].get("temporal_templates", True)),
        )
    )

    records = []
    for instance in instances:
        passages = split_documents_into_passages(
            instance.documents,
            max_words=int(config["preprocess"].get("passage_max_words", 140)),
            stride_sentences=int(config["preprocess"].get("passage_stride_sentences", 1)),
        )
        if mode == "concat_baseline":
            record = generator.generate_concat_baseline(
                instance.query,
                passages,
                top_k=int(config["retrieval"].get("top_k_passages", 8)),
            )
        elif mode == "llm_all_docs":
            record = generator.generate_llm_all_docs(instance.query, passages)
        elif mode in {"hybrid_evidence_rag_v1", "extractive_evidence_only"}:
            evidence = ranker.select(
                instance.query.text,
                passages,
                top_k=int(config["retrieval"].get("top_k_passages", 8)),
            )
            record = generator.generate_from_evidence(instance.query, evidence)
        else:
            raise ValueError(f"Unknown run mode: {mode}")
        records.append(record)

    write_jsonl(records, args.output)
    errors = validate_jsonl(args.output)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"Wrote {len(records)} records to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CLEF LongEval 2026 Task 4 RAG pipeline")
    parser.add_argument("--config", default="task4_rag/configs/task4_default.yaml", help="Path to YAML config")
    parser.add_argument("--queries", help="Override query/narrative input path")
    parser.add_argument("--documents", help="Override document input path or directory")
    parser.add_argument("--candidates", help="Override candidate mapping path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--mode", choices=["concat_baseline", "llm_all_docs", "hybrid_evidence_rag_v1", "extractive_evidence_only"])
    parser.add_argument("--run-id", help="Run ID written into metadata")
    parser.add_argument("--team-id", help="Team ID written into metadata")
    parser.add_argument("--max-queries", type=int, help="Optional limit for the number of queries to run")
    parser.add_argument("--validate-only", action="store_true", help="Validate --output and exit")
    return parser


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_simple_yaml(path)

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Read the simple two-level YAML config shipped with this package.

    This fallback keeps the default pipeline runnable in minimal Python
    environments. For complex YAML syntax, install PyYAML.
    """

    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line_without_comment = raw_line.split("#", 1)[0].rstrip()
            if not line_without_comment.strip():
                continue
            indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
            line = line_without_comment.strip()
            if ":" not in line:
                raise ValueError(f"Unsupported YAML line {path}:{line_no}: {raw_line.rstrip()}")
            key, value = [part.strip() for part in line.split(":", 1)]
            if indent == 0:
                if value:
                    root[key] = _parse_scalar(value)
                    current_section = None
                else:
                    current_section = {}
                    root[key] = current_section
            elif indent == 2 and current_section is not None:
                current_section[key] = _parse_scalar(value)
            else:
                raise ValueError(f"Unsupported YAML indentation {path}:{line_no}: {raw_line.rstrip()}")
    return root


def _parse_scalar(value: str) -> Any:
    if value == "":
        return None
    lower = value.lower()
    if lower in {"null", "none", "~"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    config.setdefault("data", {})
    config.setdefault("run", {})
    if args.queries:
        config["data"]["queries_path"] = args.queries
    if args.documents:
        config["data"]["documents_path"] = args.documents
    if args.candidates:
        config["data"]["candidates_path"] = args.candidates
    if args.max_queries is not None:
        config["data"]["max_queries"] = args.max_queries
    if args.mode:
        config["run"]["mode"] = args.mode
        config["run"]["run_id"] = args.mode
    if args.run_id:
        config["run"]["run_id"] = args.run_id
    if args.team_id:
        config["run"]["team_id"] = args.team_id
    return config


def _parse_doc_text_fields(value: Any) -> list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split("|") if part.strip()]
    return [str(value)]


if __name__ == "__main__":
    raise SystemExit(main())

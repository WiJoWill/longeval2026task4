"""Prepare OpenAI Batch JSONL for gold answers from doc-level evidence.

Unlike the normal experiment batch builder, this gold workflow does not split
documents into chunks. Each candidate document becomes one evidence item, using
fullText -> abstract -> title fallback when fullText is absent.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

from .data_loader import DOC_ID_FIELDS, Query, _iter_records, _pick, load_queries
from .generator import _answer_json_schema
from .run_task4 import _load_local_env


DEFAULT_QUERIES = Path(".cache/longeval-sci-2026/task4_longeval_rag-query_docids.jsonl")
DEFAULT_DOCUMENTS = Path(
    ".cache/longeval-sci-2026/"
    "longeval_sci_test-09-11_2026_fulltext/data/processed/"
    "doc_collection_parallel_09032026_parallel_2/snapshot-3/"
    "longeval_sci_test-09-11_2026_fulltext/documents"
)
DEFAULT_OUTPUT_DIR = Path("outputs/test/openai_gold_fulltext/inputs")
DEFAULT_RUN_ID = "gold_llm_all_docs_fulltext_fallback_openai_v1"
DEFAULT_MODEL = "gpt-5.4-mini"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _load_local_env()

    model = args.model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.queries, limit=args.max_queries)
    raw_docs = _load_raw_documents(Path(args.documents), set(_candidate_doc_ids(queries)))

    requests: list[dict[str, Any]] = []
    state: dict[str, Any] = {"run_id": args.run_id, "model": model, "queries": {}}
    rows: list[dict[str, Any]] = []

    for query in queries:
        references = list(query.candidate_doc_ids)
        evidence = _document_evidence(references, raw_docs)
        custom_id = f"{args.run_id}|{query.query_id}"
        requests.append(
            _openai_batch_request(
                custom_id=custom_id,
                model=model,
                query=query,
                evidence=evidence,
                max_answer_sentences=args.max_answer_sentences,
                temperature=args.temperature,
            )
        )
        state["queries"][custom_id] = {
            "query_id": query.query_id,
            "references": references,
            "sentence_candidates": [
                {"text": item["text"], "citations": [item["reference_index"]], "doc_id": item["doc_id"]}
                for item in evidence
            ],
            "metadata": {
                "team_id": args.team_id,
                "run_id": args.run_id,
                "type": args.run_type,
                "narrative": query.text,
                "narrative_id": query.query_id,
            },
            "text_sources": {item["doc_id"]: item["text_source"] for item in evidence},
        }
        rows.append(_stats_row(query, evidence))

    requests_path = output_dir / f"{args.run_id}_requests.jsonl"
    merged_path = output_dir / "all_experiments_requests.jsonl"
    state_path = output_dir / f"{args.run_id}_state.json"
    stats_path = output_dir / f"{args.run_id}_stats.json"
    report_path = output_dir / f"{args.run_id}_stats.md"

    _write_jsonl(requests, requests_path)
    _write_jsonl(requests, merged_path)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = _stats_summary(rows)
    stats_path.write_text(json.dumps({"summary": stats, "queries": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_stats_report(stats, rows, requests_path, state_path, model), encoding="utf-8")

    print(f"Wrote OpenAI gold batch requests: {requests_path}")
    print(f"Wrote merged request file: {merged_path}")
    print(f"Wrote state: {state_path}")
    print(f"Wrote stats: {stats_path}")
    print(f"Wrote report: {report_path}")
    print(
        f"Prepared {len(requests)} requests with model={model}; "
        f"fullText={stats['fulltext_docs']}, abstract_fallback={stats['abstract_fallback_docs']}, "
        f"title_fallback={stats['title_fallback_docs']}; "
        f"approx tokens/query median={stats['approx_tokens_median']:,}, max={stats['approx_tokens_max']:,}"
    )
    print("Not submitted to OpenAI.")
    return 0


def _openai_batch_request(
    custom_id: str,
    model: str,
    query: Query,
    evidence: list[dict[str, Any]],
    max_answer_sentences: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(query, evidence, max_answer_sentences)},
            ],
            "response_format": {"type": "json_schema", "json_schema": _answer_json_schema()},
        },
    }


def _system_prompt() -> str:
    return (
        "You are a citation-aware scientific RAG assistant for CLEF LongEval 2026 Task 4. "
        "Use only the provided candidate document evidence. Do not use outside knowledge. "
        "Write concise atomic answer sentences. Each answer sentence must include the evidence_ids "
        "of the documents that support it. Return valid JSON only."
    )


def _user_prompt(query: Query, evidence: list[dict[str, Any]], max_answer_sentences: int) -> str:
    lines = [
        f"Narrative ID: {query.query_id}",
        f"Narrative: {query.text}",
        f"Maximum answer sentences: {max_answer_sentences}",
        "",
        "Each evidence item is one candidate document. Prefer fullText; abstract/title is used only when fullText is absent.",
        "Evidence:",
    ]
    for item in evidence:
        lines.extend(
            [
                f"<evidence evidence_id=\"{item['evidence_id']}\" doc_id=\"{item['doc_id']}\">",
                f"Title: {item['title']}",
                f"Text source: {item['text_source']}",
                "Evidence text:",
                item["text"] if item["text"] else "[NO_TEXT_AVAILABLE_IN_SOURCE_FILE]",
                "</evidence>",
                "",
            ]
        )
    lines.extend(
        [
            "Instructions:",
            "- Answer only from the evidence text above.",
            "- Prefer 2-5 short evidence-grounded sentences.",
            "- Use multiple evidence_ids when one answer sentence combines multiple documents.",
            "- If the evidence does not support an answer, return an empty answer list.",
            "- Return valid JSON only: {\"answer\":[{\"text\":\"...\",\"evidence_ids\":[0,1]}]}",
        ]
    )
    return "\n".join(lines)


def _candidate_doc_ids(queries: Iterable[Query]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for query in queries:
        for doc_id in query.candidate_doc_ids:
            if doc_id not in seen:
                seen.add(doc_id)
                output.append(doc_id)
    return output


def _load_raw_documents(path: Path, allowed_doc_ids: set[str]) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for row in _iter_records(path):
        doc_id = str(_pick(row, DOC_ID_FIELDS, required=True))
        if doc_id not in allowed_doc_ids:
            continue
        docs[doc_id] = dict(row)
        if set(docs) >= allowed_doc_ids:
            break
    return docs


def _document_evidence(references: list[str], raw_docs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for reference_index, doc_id in enumerate(references):
        row = raw_docs.get(doc_id, {})
        title = str(row.get("title", "") or "").strip()
        fulltext = str(row.get("fullText", "") or "").strip()
        abstract = str(row.get("abstract", "") or "").strip()
        text = fulltext or abstract or title
        source = "fullText" if fulltext else "abstract" if abstract else "title" if title else "missing"
        evidence.append(
            {
                "evidence_id": reference_index,
                "reference_index": reference_index,
                "doc_id": doc_id,
                "title": title,
                "text_source": source,
                "text": text,
            }
        )
    return evidence


def _stats_row(query: Query, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    chars = sum(len(item["text"]) for item in evidence)
    return {
        "query_id": query.query_id,
        "doc_count": len(evidence),
        "fulltext_doc_count": sum(1 for item in evidence if item["text_source"] == "fullText"),
        "abstract_fallback_doc_count": sum(1 for item in evidence if item["text_source"] == "abstract"),
        "title_fallback_doc_count": sum(1 for item in evidence if item["text_source"] == "title"),
        "missing_text_doc_count": sum(1 for item in evidence if item["text_source"] == "missing"),
        "evidence_chars": chars,
        "approx_tokens_chars_div_4": chars // 4,
    }


def _stats_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [row["approx_tokens_chars_div_4"] for row in rows] or [0]
    chars = [row["evidence_chars"] for row in rows] or [0]
    return {
        "queries": len(rows),
        "docs": sum(row["doc_count"] for row in rows),
        "fulltext_docs": sum(row["fulltext_doc_count"] for row in rows),
        "abstract_fallback_docs": sum(row["abstract_fallback_doc_count"] for row in rows),
        "title_fallback_docs": sum(row["title_fallback_doc_count"] for row in rows),
        "missing_text_docs": sum(row["missing_text_doc_count"] for row in rows),
        "approx_tokens_min": min(tokens),
        "approx_tokens_median": int(statistics.median(tokens)),
        "approx_tokens_mean": int(statistics.mean(tokens)),
        "approx_tokens_max": max(tokens),
        "chars_min": min(chars),
        "chars_median": int(statistics.median(chars)),
        "chars_mean": int(statistics.mean(chars)),
        "chars_max": max(chars),
    }


def _stats_report(stats: dict[str, Any], rows: list[dict[str, Any]], requests_path: Path, state_path: Path, model: str) -> str:
    lines = [
        "# OpenAI Gold Fulltext Fallback Batch Stats",
        "",
        f"- model: `{model}`",
        f"- requests_file: `{requests_path}`",
        f"- state_file: `{state_path}`",
        f"- queries: {stats['queries']}",
        f"- docs: {stats['docs']}",
        f"- docs_using_fullText: {stats['fulltext_docs']}",
        f"- docs_using_abstract_fallback: {stats['abstract_fallback_docs']}",
        f"- docs_using_title_fallback: {stats['title_fallback_docs']}",
        f"- docs_missing_any_text: {stats['missing_text_docs']}",
        f"- approx_tokens_per_query_chars_div_4: min {stats['approx_tokens_min']:,}, median {stats['approx_tokens_median']:,}, mean {stats['approx_tokens_mean']:,}, max {stats['approx_tokens_max']:,}",
        "",
        "| query_id | docs | fullText | abstract_fallback | title_fallback | missing_text | evidence_chars | approx_tokens_chars_div_4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['query_id']} | {row['doc_count']} | {row['fulltext_doc_count']} | "
            f"{row['abstract_fallback_doc_count']} | {row['title_fallback_doc_count']} | "
            f"{row['missing_text_doc_count']} | {row['evidence_chars']:,} | "
            f"{row['approx_tokens_chars_div_4']:,} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_jsonl(items: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=True))
            handle.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare OpenAI gold batch from doc-level fullText/abstract/title evidence.")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--team-id", default="our_team")
    parser.add_argument("--run-type", default="automatic")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-answer-sentences", type=int, default=5)
    parser.add_argument("--max-queries", type=int)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

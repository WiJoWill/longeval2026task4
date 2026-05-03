"""Prepare Gemini Batch JSONL for gold fulltext answer generation.

This path is intentionally separate from the retrieval experiments:

* no chunking
* no retrieval
* no sentence extraction
* no top-N evidence truncation
* no fallback from fullText to abstract/title
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any

from .data_loader import load_task
from .run_task4 import _load_local_env


DEFAULT_QUERIES = Path(".cache/longeval-sci-2026/task4_longeval_rag-query_docids.jsonl")
DEFAULT_DOCUMENTS = Path(
    ".cache/longeval-sci-2026/longeval_sci_test-09-11_2026_fulltext/data/processed/"
    "doc_collection_parallel_09032026_parallel_2/snapshot-3/"
    "longeval_sci_test-09-11_2026_fulltext/documents"
)
DEFAULT_OUTPUT_DIR = Path("outputs/test/gemini_gold_fulltext/inputs")
DEFAULT_RUN_ID = "gold_gemini_fulltext_v1"
DEFAULT_MODEL = "gemini-2.5-flash"


SYSTEM_PROMPT = """You are a citation-aware scientific RAG assistant for CLEF LongEval 2026 Task 4.

Use only the fullText evidence supplied by the user. Do not use outside knowledge, abstracts, titles as evidence, unstated assumptions, or facts remembered from training data.

Write short atomic answer sentences with one factual claim per sentence. Each answer sentence must include the evidence_ids of the full documents you drew from. If a claim synthesizes support from multiple documents, include all their evidence_ids.

Return valid JSON only, with this exact shape:
{"answer":[{"text":"A short factual sentence.","evidence_ids":[0,2]}]}"""


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _load_local_env()

    queries_path = Path(args.queries)
    documents_path = Path(args.documents)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = load_task(
        queries_path=queries_path,
        documents_path=documents_path,
        max_queries=args.max_queries,
        document_text_fields=["fullText"],
    )

    requests_path = output_dir / f"{args.run_id}_requests.jsonl"
    state_path = output_dir / f"{args.run_id}_state.json"
    audit_json_path = output_dir / f"{args.run_id}_audit.json"
    audit_md_path = output_dir / f"{args.run_id}_audit.md"

    state_index: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []
    total_evidence_docs = 0
    missing_fulltext_refs = 0

    with requests_path.open("w", encoding="utf-8", newline="\n") as out:
        for instance in instances:
            custom_id = f"{args.run_id}|{instance.query.query_id}"
            references = [doc.doc_id for doc in instance.documents]
            evidence, missing_doc_ids = _fulltext_evidence(instance.documents, references)
            total_evidence_docs += len(evidence)
            missing_fulltext_refs += len(missing_doc_ids)

            prompt = _user_prompt(
                query_id=instance.query.query_id,
                query_text=instance.query.text,
                max_answer_sentences=args.max_answer_sentences,
                evidence=evidence,
            )
            request = {
                "key": custom_id,
                "request": {
                    "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": args.temperature,
                        "responseMimeType": "application/json",
                        "responseSchema": _answer_schema(),
                    },
                },
            }
            out.write(json.dumps(request, ensure_ascii=False) + "\n")

            state_index[custom_id] = {
                "query_id": instance.query.query_id,
                "references": references,
                "sentence_candidates": [
                    {
                        "text": item["fullText"],
                        "citations": [item["reference_index"]],
                        "doc_id": item["doc_id"],
                        "title": item.get("title", ""),
                    }
                    for item in evidence
                ],
                "metadata": {
                    "team_id": args.team_id,
                    "run_id": args.run_id,
                    "type": "automatic",
                    "narrative": instance.query.text,
                    "narrative_id": instance.query.query_id,
                },
                "missing_fulltext_doc_ids": missing_doc_ids,
            }

            chars = sum(len(item["fullText"]) for item in evidence)
            words = sum(_word_count(item["fullText"]) for item in evidence)
            audit_rows.append(
                {
                    "query_id": instance.query.query_id,
                    "doc_refs": len(references),
                    "fulltext_evidence_docs": len(evidence),
                    "missing_fulltext_docs": len(missing_doc_ids),
                    "chars": chars,
                    "words": words,
                    "approx_tokens_chars_div_4": round(chars / 4),
                    "request_bytes_utf8": len(json.dumps(request, ensure_ascii=False).encode("utf-8")),
                }
            )

    state_path.write_text(json.dumps({"queries": state_index}, ensure_ascii=False), encoding="utf-8")
    audit = {
        "run_id": args.run_id,
        "model": args.model,
        "queries": len(audit_rows),
        "requests_path": str(requests_path),
        "state_path": str(state_path),
        "total_evidence_docs": total_evidence_docs,
        "missing_fulltext_refs": missing_fulltext_refs,
        "summary": _summary(audit_rows),
        "queries_detail": audit_rows,
    }
    audit_json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_md_path.write_text(_audit_markdown(audit), encoding="utf-8")

    print(f"Prepared {len(audit_rows)} Gemini gold fulltext requests")
    print(f"  requests: {requests_path}")
    print(f"  state:    {state_path}")
    print(f"  audit:    {audit_md_path}")
    print(f"  model:    {args.model}")
    print(f"  fullText evidence docs: {total_evidence_docs}")
    print(f"  missing fullText refs:  {missing_fulltext_refs}")
    return 0


def _fulltext_evidence(documents: list[Any], references: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    evidence: list[dict[str, Any]] = []
    missing_doc_ids: list[str] = []
    for doc in documents:
        full_text = str((doc.metadata or {}).get("fullText") or "").strip()
        if not full_text:
            missing_doc_ids.append(doc.doc_id)
            continue
        evidence.append(
            {
                "evidence_id": len(evidence),
                "reference_index": references.index(doc.doc_id),
                "doc_id": doc.doc_id,
                "title": doc.title,
                "fullText": full_text,
            }
        )
    return evidence, missing_doc_ids


def _user_prompt(query_id: str, query_text: str, max_answer_sentences: int, evidence: list[dict[str, Any]]) -> str:
    payload = {
        "narrative_id": query_id,
        "narrative": query_text,
        "max_answer_sentences": max_answer_sentences,
        "evidence": evidence,
    }
    return (
        "Generate a citation-grounded answer for the narrative below.\n\n"
        "Rules:\n"
        "- Use only each document's fullText field as evidence.\n"
        "- Do not treat title as evidence; it is only an identifier for orientation.\n"
        "- Every factual answer sentence must include evidence_ids from the supplied evidence list.\n"
        "- Use multiple evidence_ids when synthesizing across documents.\n"
        "- If the supplied fullText evidence is insufficient, return {\"answer\":[]}.\n"
        "- Return JSON only.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _answer_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["text", "evidence_ids"],
                },
            }
        },
        "required": ["answer"],
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    summary: dict[str, Any] = {}
    for key in ("fulltext_evidence_docs", "missing_fulltext_docs", "chars", "words", "approx_tokens_chars_div_4", "request_bytes_utf8"):
        values = [int(row[key]) for row in rows]
        summary[key] = {
            "min": min(values),
            "median": median(values),
            "mean": mean(values),
            "max": max(values),
        }
    return summary


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Gemini Gold Fulltext Batch Audit",
        "",
        f"- run_id: `{audit['run_id']}`",
        f"- model: `{audit['model']}`",
        f"- queries: `{audit['queries']}`",
        f"- total fullText evidence docs: `{audit['total_evidence_docs']}`",
        f"- missing fullText refs: `{audit['missing_fulltext_refs']}`",
        f"- requests: `{audit['requests_path']}`",
        f"- state: `{audit['state_path']}`",
        "",
        "| metric | min | median | mean | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, values in audit["summary"].items():
        lines.append(
            f"| `{key}` | {values['min']} | {values['median']:.1f} | {values['mean']:.1f} | {values['max']} |"
        )
    lines.extend(["", "| query_id | refs | fullText docs | missing fullText | chars | words | approx tokens | request bytes |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in audit["queries_detail"]:
        lines.append(
            f"| `{row['query_id']}` | {row['doc_refs']} | {row['fulltext_evidence_docs']} | "
            f"{row['missing_fulltext_docs']} | {row['chars']} | {row['words']} | "
            f"{row['approx_tokens_chars_div_4']} | {row['request_bytes_utf8']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Gemini Batch JSONL for gold fulltext generation.")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--team-id", default="our_team")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-answer-sentences", type=int, default=5)
    parser.add_argument("--max-queries", type=int)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

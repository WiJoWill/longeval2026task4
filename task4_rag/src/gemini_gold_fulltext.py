"""Prepare and apply Gemini Batch jobs for full-text gold answers.

This workflow is intentionally separate from the retrieval experiments:
each candidate document is passed as one full-text evidence item, without
chunking, sentence filtering, reranking, or evidence truncation.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from google.genai import types

from .data_loader import DOC_ID_FIELDS, Query, _iter_records, _pick, load_queries
from .generator import _parse_answer_json, _repair_answer
from .output_writer import write_jsonl
from .run_task4 import _load_local_env
from .validator import validate_jsonl, validate_record


DEFAULT_QUERIES = Path(".cache/longeval-sci-2026/task4_longeval_rag-query_docids.jsonl")
DEFAULT_DOCUMENTS = Path(
    ".cache/longeval-sci-2026/"
    "longeval_sci_test-09-11_2026_fulltext/data/processed/"
    "doc_collection_parallel_09032026_parallel_2/snapshot-3/"
    "longeval_sci_test-09-11_2026_fulltext/documents"
)
DEFAULT_INPUT_DIR = Path("outputs/test/gemini_gold_fulltext/inputs")
DEFAULT_OUTPUT_DIR = Path("outputs/test/gemini_gold_fulltext/runs")
DEFAULT_RUN_ID = "gold_gemini_2_5_flash_standard_v1"
DEFAULT_MODEL = "gemini-2.5-flash"
COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "submit":
        return _submit(args)
    if args.command == "generate":
        return _generate(args)
    if args.command == "process":
        return _process(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _prepare(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = load_queries(args.queries, limit=args.max_queries)
    doc_ids = _candidate_doc_ids(queries)
    raw_docs = _load_raw_documents(Path(args.documents), set(doc_ids))

    requests_path = output_dir / "gemini_gold_fulltext_requests.jsonl"
    state_path = output_dir / "gemini_gold_fulltext_state.json"
    stats_path = output_dir / "gemini_gold_fulltext_stats.json"
    report_path = output_dir / "gemini_gold_fulltext_stats.md"

    state: dict[str, Any] = {"queries": {}}
    rows: list[dict[str, Any]] = []
    with requests_path.open("w", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            custom_id = f"{args.run_id}|{query.query_id}"
            references = list(query.candidate_doc_ids)
            evidence, missing_fulltext = _fulltext_evidence(references, raw_docs)
            request = _gemini_batch_request(
                key=custom_id,
                query=query,
                evidence=evidence,
                model=args.model,
                max_answer_sentences=args.max_answer_sentences,
                temperature=args.temperature,
                thinking_budget=args.thinking_budget,
                service_tier=args.service_tier,
            )
            handle.write(json.dumps(request, ensure_ascii=False))
            handle.write("\n")

            sentence_candidates = [
                {
                    "text": item["text"],
                    "citations": [item["reference_index"]],
                    "doc_id": item["doc_id"],
                }
                for item in evidence
            ]
            state["queries"][custom_id] = {
                "query_id": query.query_id,
                "references": references,
                "sentence_candidates": sentence_candidates,
                "metadata": {
                    "team_id": args.team_id,
                    "run_id": args.run_id,
                    "type": args.run_type,
                    "narrative": query.text,
                    "narrative_id": query.query_id,
                },
                "missing_fulltext_doc_ids": missing_fulltext,
            }
            rows.append(_stats_row(query, evidence, missing_fulltext))

    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = _stats_summary(rows)
    stats_path.write_text(json.dumps({"summary": stats, "queries": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_stats_report(stats, rows, requests_path, state_path), encoding="utf-8")

    print(f"Wrote Gemini batch requests: {requests_path}")
    print(f"Wrote state: {state_path}")
    print(f"Wrote stats: {stats_path}")
    print(f"Wrote report: {report_path}")
    print(
        "Prepared "
        f"{len(rows)} queries; approx input tokens/query "
        f"min={stats['approx_tokens_min']:,} median={stats['approx_tokens_median']:,} "
        f"mean={stats['approx_tokens_mean']:,} max={stats['approx_tokens_max']:,}; "
        f"fullText docs={stats['fulltext_docs']:,}; "
        f"abstract fallback docs={stats['abstract_fallback_docs']:,}; "
        f"title fallback docs={stats['title_fallback_docs']:,}"
    )
    return 0


def _submit(args: argparse.Namespace) -> int:
    _load_local_env()
    try:
        from google import genai
    except ImportError:
        print("Error: google-genai is not installed (pip install google-genai)", file=sys.stderr)
        return 1

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY is not set", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=api_key)

    print(f"Uploading {args.requests_file} ...")
    uploaded_file = client.files.upload(
        file=str(args.requests_file),
        config=types.UploadFileConfig(display_name=args.display_name, mime_type="jsonl"),
    )
    print(f"  uploaded_file={uploaded_file.name}")

    print("Creating Gemini batch job ...")
    batch_job = client.batches.create(
        model=args.model,
        src=uploaded_file.name,
        config={"display_name": args.display_name},
    )
    job_name = batch_job.name
    (output_dir / "gemini_batch_job_name.txt").write_text(str(job_name), encoding="utf-8")
    print(f"  job_name={job_name}")

    if args.no_wait:
        print("Submitted only; not waiting for completion.")
        return 0
    return _poll_and_download(client, job_name, output_dir, args.poll_interval)


def _generate(args: argparse.Namespace) -> int:
    """Run standard synchronous Gemini requests from a prepared batch JSONL."""

    _load_local_env()
    try:
        from google import genai
    except ImportError:
        print("Error: google-genai is not installed (pip install google-genai)", file=sys.stderr)
        return 1

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY is not set", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "gemini_standard_results.jsonl"
    client = genai.Client(api_key=api_key)
    requests = list(_read_jsonl(Path(args.requests_file)))
    if args.max_requests is not None:
        requests = requests[: args.max_requests]
    done = _completed_raw_keys(raw_path)
    print(f"Loaded {len(requests)} requests; already_done={len(done)}")

    with raw_path.open("a", encoding="utf-8", newline="\n") as raw_handle:
        for index, item in enumerate(requests, start=1):
            key = str(item.get("key", ""))
            if not key or key in done:
                continue
            request = item.get("request", {})
            print(f"Generating {index}/{len(requests)} {key} ...")
            result = _call_gemini_with_retries(
                client=client,
                model=args.model,
                contents=request.get("contents", []),
                config=request.get("config", {}),
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
            raw_handle.write(json.dumps({"key": key, **result}, ensure_ascii=False))
            raw_handle.write("\n")
            raw_handle.flush()
            done.add(key)
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Wrote raw Gemini standard results: {raw_path}")
    return _process(
        argparse.Namespace(
            results_file=raw_path,
            state_file=Path(args.state_file),
            output_dir=output_dir,
            run_id=args.run_id,
            max_answer_sentences=args.max_answer_sentences,
        )
    )


def _process(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    state_index: dict[str, Any] = state.get("queries", {})
    records: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for result in _read_jsonl(Path(args.results_file)):
        custom_id = _result_key(result)
        if not custom_id:
            failed.append({"error": "missing result key", "raw": result})
            continue
        item_state = state_index.get(custom_id)
        if item_state is None:
            failed.append({"custom_id": custom_id, "error": "missing state", "raw": result})
            continue
        references = item_state["references"]
        sentence_candidates = item_state["sentence_candidates"]
        answer = _extract_gemini_answer(
            result=result,
            references=references,
            sentence_candidates=sentence_candidates,
            max_answer_sentences=args.max_answer_sentences,
            custom_id=custom_id,
            failed=failed,
        )
        record = {
            "metadata": item_state["metadata"],
            "references": references,
            "answer": answer,
        }
        try:
            validate_record(record)
        except Exception as exc:
            failed.append({"custom_id": custom_id, "error": f"validation failed: {exc}", "record": record})
        records.append(record)

    out_path = output_dir / f"{args.run_id}.jsonl"
    write_jsonl(records, out_path)
    errors = validate_jsonl(out_path)
    if failed:
        failed_path = output_dir / f"{args.run_id}_failed.jsonl"
        write_jsonl(failed, failed_path)
        print(f"Wrote failures: {failed_path}")
    print(f"Wrote run: {out_path} ({len(records)} records, validation_errors={len(errors)})")
    if errors:
        for error in errors[:10]:
            print(f"  {error}", file=sys.stderr)
    return 0 if not errors else 1


def _poll_and_download(client: Any, job_name: str, output_dir: Path, poll_interval: int) -> int:
    batch_job = client.batches.get(name=job_name)
    while batch_job.state.name not in COMPLETED_STATES:
        print(f"  status={batch_job.state.name}")
        time.sleep(poll_interval)
        batch_job = client.batches.get(name=job_name)
    print(f"  final_status={batch_job.state.name}")
    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Error: Gemini batch ended with {batch_job.state.name}: {getattr(batch_job, 'error', None)}", file=sys.stderr)
        return 1
    result_file_name = batch_job.dest.file_name
    print(f"Downloading {result_file_name} ...")
    content = client.files.download(file=result_file_name)
    results_path = output_dir / "gemini_batch_results.jsonl"
    results_path.write_bytes(content)
    print(f"  saved to {results_path}")
    return 0


def _call_gemini_with_retries(
    client: Any,
    model: str,
    contents: list[dict[str, Any]],
    config: dict[str, Any],
    max_retries: int,
    retry_sleep: int,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            if hasattr(response, "model_dump"):
                return {"response": response.model_dump(mode="json", exclude_none=True)}
            return {"response": json.loads(response.model_dump_json(exclude_none=True))}
        except Exception as exc:
            if attempt >= max_retries:
                return {"error": {"message": str(exc), "type": type(exc).__name__}}
            wait = retry_sleep * (attempt + 1)
            print(f"  Gemini error ({type(exc).__name__}); retrying in {wait}s: {exc}", file=sys.stderr)
            time.sleep(wait)


def _completed_raw_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for item in _read_jsonl(path):
        key = _result_key(item)
        if key:
            keys.add(key)
    return keys


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


def _fulltext_evidence(references: list[str], raw_docs: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    evidence: list[dict[str, Any]] = []
    missing_fulltext: list[str] = []
    for reference_index, doc_id in enumerate(references):
        row = raw_docs.get(doc_id, {})
        title = str(row.get("title", "") or "").strip()
        fulltext = str(row.get("fullText", "") or "").strip()
        abstract = str(row.get("abstract", "") or "").strip()
        text = fulltext or abstract or title
        text_source = "fullText" if fulltext else "abstract" if abstract else "title" if title else "missing"
        if not fulltext:
            missing_fulltext.append(doc_id)
        evidence.append(
            {
                "evidence_id": reference_index,
                "reference_index": reference_index,
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "text_source": text_source,
                "missing_fulltext": not bool(fulltext),
            }
        )
    return evidence, missing_fulltext


def _gemini_batch_request(
    key: str,
    query: Query,
    evidence: list[dict[str, Any]],
    model: str,
    max_answer_sentences: int,
    temperature: float,
    thinking_budget: int,
    service_tier: str,
) -> dict[str, Any]:
    prompt = _user_prompt(query, evidence, max_answer_sentences)
    config: dict[str, Any] = {
        "system_instruction": {"parts": [{"text": _system_prompt()}]},
        "temperature": temperature,
        "response_mime_type": "application/json",
        "response_schema": _answer_schema(),
        "thinking_config": {"thinking_budget": thinking_budget},
    }
    if service_tier:
        config["service_tier"] = service_tier
    return {
        "key": key,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "config": config,
        },
    }


def _system_prompt() -> str:
    return (
        "You are a citation-aware scientific RAG assistant for CLEF LongEval 2026 Task 4. "
        "Use only the provided candidate document evidence. Do not use outside knowledge. "
        "Write concise atomic answer sentences. Each answer sentence must include the evidence_ids "
        "of the full-text documents that support it. Return valid JSON only."
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
            "- Do not mention timestamps.",
            "- Prefer 2-5 short evidence-grounded sentences.",
            "- Use multiple evidence_ids when one answer sentence combines multiple documents.",
            "- If the fullText evidence does not support an answer, return an empty answer list.",
            "- Return valid JSON only: {\"answer\":[{\"text\":\"...\",\"evidence_ids\":[0,1]}]}",
        ]
    )
    return "\n".join(lines)


def _answer_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["answer"],
        "properties": {
            "answer": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["text", "evidence_ids"],
                    "properties": {
                        "text": {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "integer"}},
                    },
                },
            }
        },
    }


def _stats_row(query: Query, evidence: list[dict[str, Any]], missing_fulltext: list[str]) -> dict[str, Any]:
    chars = sum(len(item["text"]) for item in evidence)
    words = sum(len(item["text"].split()) for item in evidence)
    return {
        "query_id": query.query_id,
        "doc_count": len(evidence),
        "fulltext_doc_count": sum(1 for item in evidence if item["text_source"] == "fullText"),
        "abstract_fallback_doc_count": sum(1 for item in evidence if item["text_source"] == "abstract"),
        "title_fallback_doc_count": sum(1 for item in evidence if item["text_source"] == "title"),
        "missing_text_doc_count": sum(1 for item in evidence if item["text_source"] == "missing"),
        "missing_fulltext_doc_count": len(missing_fulltext),
        "missing_fulltext_doc_ids": missing_fulltext,
        "evidence_chars": chars,
        "evidence_words": words,
        "approx_tokens_chars_div_4": chars // 4,
    }


def _stats_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_counts = [row["approx_tokens_chars_div_4"] for row in rows] or [0]
    char_counts = [row["evidence_chars"] for row in rows] or [0]
    return {
        "queries": len(rows),
        "docs": sum(row["doc_count"] for row in rows),
        "fulltext_docs": sum(row["fulltext_doc_count"] for row in rows),
        "abstract_fallback_docs": sum(row["abstract_fallback_doc_count"] for row in rows),
        "title_fallback_docs": sum(row["title_fallback_doc_count"] for row in rows),
        "missing_text_docs": sum(row["missing_text_doc_count"] for row in rows),
        "missing_fulltext_docs": sum(row["missing_fulltext_doc_count"] for row in rows),
        "approx_tokens_min": min(token_counts),
        "approx_tokens_median": int(statistics.median(token_counts)),
        "approx_tokens_mean": int(statistics.mean(token_counts)),
        "approx_tokens_max": max(token_counts),
        "chars_min": min(char_counts),
        "chars_median": int(statistics.median(char_counts)),
        "chars_mean": int(statistics.mean(char_counts)),
        "chars_max": max(char_counts),
    }


def _stats_report(stats: dict[str, Any], rows: list[dict[str, Any]], requests_path: Path, state_path: Path) -> str:
    lines = [
        "# Gemini Gold Fulltext Batch Stats",
        "",
        f"- requests_file: `{requests_path}`",
        f"- state_file: `{state_path}`",
        f"- queries: {stats['queries']}",
        f"- docs: {stats['docs']}",
        f"- docs_using_fullText: {stats['fulltext_docs']}",
        f"- docs_using_abstract_fallback: {stats['abstract_fallback_docs']}",
        f"- docs_using_title_fallback: {stats['title_fallback_docs']}",
        f"- docs_missing_any_text: {stats['missing_text_docs']}",
        f"- docs_missing_fullText: {stats['missing_fulltext_docs']}",
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


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _result_key(result: dict[str, Any]) -> str:
    return str(result.get("key") or result.get("metadata", {}).get("key") or result.get("custom_id") or "")


def _extract_gemini_answer(
    result: dict[str, Any],
    references: list[str],
    sentence_candidates: list[dict[str, Any]],
    max_answer_sentences: int,
    custom_id: str,
    failed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if result.get("error"):
        failed.append({"custom_id": custom_id, "error": result["error"]})
        return _fallback_answer()
    try:
        raw_text = _gemini_response_text(result)
        raw_answer = _parse_answer_json(raw_text)
        return _repair_answer(raw_answer, references, max_answer_sentences, sentence_candidates)
    except Exception as exc:
        failed.append({"custom_id": custom_id, "error": str(exc), "raw": result})
        return _fallback_answer()


def _gemini_response_text(result: dict[str, Any]) -> str:
    response = result.get("response", {})
    if isinstance(response, dict) and response.get("text"):
        return str(response["text"])
    candidates = response.get("candidates") if isinstance(response, dict) else None
    if not candidates:
        raise ValueError("Gemini response has no candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [str(part.get("text", "")) for part in parts if part.get("text")]
    if not texts:
        raise ValueError("Gemini response has no text parts")
    return "\n".join(texts)


def _fallback_answer() -> list[dict[str, Any]]:
    return [{"text": "No supported answer could be generated from the provided candidate documents.", "citations": []}]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemini full-text gold answer batch workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare Gemini JSONL requests from raw fullText docs.")
    prepare.add_argument("--queries", default=str(DEFAULT_QUERIES))
    prepare.add_argument("--documents", default=str(DEFAULT_DOCUMENTS))
    prepare.add_argument("--output-dir", default=str(DEFAULT_INPUT_DIR))
    prepare.add_argument("--run-id", default=DEFAULT_RUN_ID)
    prepare.add_argument("--team-id", default="our_team")
    prepare.add_argument("--run-type", default="automatic")
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--max-answer-sentences", type=int, default=5)
    prepare.add_argument("--temperature", type=float, default=0.0)
    prepare.add_argument("--thinking-budget", type=int, default=0)
    prepare.add_argument("--service-tier", default="standard")
    prepare.add_argument("--max-queries", type=int)

    submit = subparsers.add_parser("submit", help="Upload requests, create a Gemini batch job, and optionally wait.")
    submit.add_argument("--requests-file", required=True, type=Path)
    submit.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    submit.add_argument("--model", default=DEFAULT_MODEL)
    submit.add_argument("--display-name", default=DEFAULT_RUN_ID)
    submit.add_argument("--poll-interval", type=int, default=30)
    submit.add_argument("--no-wait", action="store_true")

    generate = subparsers.add_parser("generate", help="Run standard synchronous Gemini calls from prepared requests.")
    generate.add_argument("--requests-file", required=True, type=Path)
    generate.add_argument("--state-file", required=True, type=Path)
    generate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    generate.add_argument("--run-id", default=DEFAULT_RUN_ID)
    generate.add_argument("--model", default=DEFAULT_MODEL)
    generate.add_argument("--max-answer-sentences", type=int, default=5)
    generate.add_argument("--max-requests", type=int)
    generate.add_argument("--sleep", type=float, default=0.0)
    generate.add_argument("--max-retries", type=int, default=3)
    generate.add_argument("--retry-sleep", type=int, default=30)

    process = subparsers.add_parser("process", help="Convert Gemini batch results into Task 4 run JSONL.")
    process.add_argument("--results-file", required=True, type=Path)
    process.add_argument("--state-file", required=True, type=Path)
    process.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    process.add_argument("--run-id", default=DEFAULT_RUN_ID)
    process.add_argument("--max-answer-sentences", type=int, default=5)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

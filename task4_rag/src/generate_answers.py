"""Answer generation pipeline using the OpenAI API.

Takes pre-retrieved evidence from any retrieval strategy and generates
standardized answers using the same downstream LLM.  Differences in output
quality then reflect retrieval quality, not generation variance.

Resume support: if --output already exists the pipeline reads completed
(query_id, model_name) pairs from it and skips them, so you can safely
re-run after a crash without re-processing or re-paying for completed rows.

Usage:
  python -m task4_rag.src.generate_answers \\
    --input retrieval_results.jsonl \\
    --output answers.jsonl \\
    --output_csv answers.csv \\
    --model gpt-5.4-mini

Input schema (JSONL line or CSV row):
  Required: query_id, query, model_name, retrieved_evidence
  Optional: retrieved_doc_ids, retrieved_token_count, metadata

retrieved_evidence must be a list of {"doc_id": ..., "text": ...} objects.
In CSV the list must be JSON-encoded in the cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a scientific question-answering assistant.

You must answer the user query ONLY using the provided evidence passages.

Rules:
- Do not use outside knowledge
- Do not infer unsupported claims
- If evidence is insufficient, explicitly state: \
"The provided evidence is insufficient to fully answer this question."
- Synthesize information across passages when appropriate
- Be concise and factual
- Do not mention that you are an AI model"""

_USER_TEMPLATE = """\
Query:
{query}

Evidence:
{formatted_evidence}

Generate a final answer based only on the evidence above."""

_CSV_FIELDNAMES = [
    "query_id",
    "model_name",
    "query",
    "generated_answer",
    "evidence_doc_ids",
    "evidence_count",
    "token_count",
    "generation_model",
    "latency_seconds",
    "prompt_tokens",
    "completion_tokens",
]


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------

def format_evidence(passages: list[dict]) -> str:
    """Format a list of passage dicts into a numbered evidence block."""
    if not passages:
        return "(no evidence provided)"
    lines: list[str] = []
    for i, passage in enumerate(passages, 1):
        doc_id = passage.get("doc_id", f"doc_{i}")
        text = str(passage.get("text", "")).strip()
        lines.append(f"[{i}] (doc_id: {doc_id})")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip()


def build_user_prompt(query: str, formatted_evidence: str) -> str:
    return _USER_TEMPLATE.format(query=query, formatted_evidence=formatted_evidence)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_input(path: str) -> list[dict]:
    """Load rows from JSONL or CSV.  CSV retrieved_evidence cells are JSON-decoded."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows: list[dict] = []
    if path.endswith(".jsonl"):
        with p.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    log.warning("Skipping malformed line %d: %s", lineno, exc)
    else:
        with p.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                evidence = row.get("retrieved_evidence", "")
                if isinstance(evidence, str) and evidence.strip().startswith("["):
                    try:
                        row["retrieved_evidence"] = json.loads(evidence)
                    except json.JSONDecodeError:
                        row["retrieved_evidence"] = []
                rows.append(row)
    return rows


def load_completed_keys(output_path: Path) -> set[tuple[str, str]]:
    """Return (query_id, model_name) pairs already written to output_path."""
    if not output_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with output_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                keys.add((str(rec.get("query_id", "")), str(rec.get("model_name", ""))))
            except json.JSONDecodeError:
                pass
    return keys


def write_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> int:
    """Rewrite the output JSONL as a flat CSV; returns the number of rows written."""
    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            if isinstance(row.get("evidence_doc_ids"), list):
                row["evidence_doc_ids"] = json.dumps(row["evidence_doc_ids"])
            writer.writerow(row)
    return len(records)


# ---------------------------------------------------------------------------
# OpenAI call with retry
# ---------------------------------------------------------------------------

def call_openai(
    client: Any,
    model: str,
    query: str,
    formatted_evidence: str,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> tuple[str, dict[str, int], float]:
    """Return (answer_text, usage_dict, latency_seconds).

    Retries on rate limits with exponential back-off, and on transient API
    errors with linear back-off.
    """
    import openai

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, formatted_evidence)},
    ]
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
            )
            latency = time.perf_counter() - t0
            answer = (response.choices[0].message.content or "").strip()
            usage_obj = response.usage
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }
            return answer, usage, latency

        except openai.RateLimitError as exc:
            wait = retry_delay * (2 ** (attempt - 1)) * 5
            log.warning("Rate limit (attempt %d/%d), backing off %.1fs", attempt, max_retries, wait)
            last_exc = exc
            time.sleep(wait)

        except openai.APIError as exc:
            log.warning("API error (attempt %d/%d): %s", attempt, max_retries, exc)
            last_exc = exc
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                raise

    raise RuntimeError(f"All {max_retries} retries exhausted") from last_exc


# ---------------------------------------------------------------------------
# Output record assembly
# ---------------------------------------------------------------------------

def build_output_record(
    row: dict,
    answer: str,
    usage: dict[str, int],
    latency: float,
    model: str,
    include_raw_prompt: bool = False,
    formatted_evidence: str = "",
) -> dict:
    passages = row.get("retrieved_evidence") or []
    if not isinstance(passages, list):
        passages = []
    doc_ids = [str(p.get("doc_id", "")) for p in passages if p.get("doc_id")]

    record: dict[str, Any] = {
        "query_id": str(row.get("query_id", "")),
        "model_name": str(row.get("model_name", "")),
        "query": str(row.get("query", "")),
        "generated_answer": answer,
        "evidence_doc_ids": doc_ids,
        "evidence_count": len(passages),
        "token_count": usage.get("total_tokens", 0),
        "generation_model": model,
        "latency_seconds": round(latency, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }
    for optional in ("retrieved_doc_ids", "retrieved_token_count", "metadata"):
        if row.get(optional) is not None:
            record[optional] = row[optional]
    if include_raw_prompt:
        record["raw_prompt"] = build_user_prompt(str(row.get("query", "")), formatted_evidence)
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        import openai
    except ImportError:
        print("Error: openai package not installed.  Run: pip install openai", file=sys.stderr)
        return 1

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        return 1

    client = openai.OpenAI(api_key=api_key)

    log.info("Loading input from %s", args.input)
    rows = load_input(args.input)
    log.info("Loaded %d rows", len(rows))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_keys = load_completed_keys(output_path)
    if completed_keys:
        log.info("Resuming: %d rows already done, skipping", len(completed_keys))

    total_tokens = 0
    n_processed = 0
    n_skipped = 0
    n_failed = 0

    with output_path.open("a", encoding="utf-8") as out_fh:
        for row in rows:
            query_id = str(row.get("query_id", ""))
            model_name = str(row.get("model_name", ""))
            key = (query_id, model_name)

            if key in completed_keys:
                n_skipped += 1
                continue

            query = str(row.get("query", ""))
            passages = row.get("retrieved_evidence") or []
            if not isinstance(passages, list):
                passages = []

            evidence_text = format_evidence(passages)

            try:
                answer, usage, latency = call_openai(
                    client=client,
                    model=args.model,
                    query=query,
                    formatted_evidence=evidence_text,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                )
            except Exception as exc:
                log.error("FAILED query_id=%s model=%s: %s", query_id, model_name, exc)
                n_failed += 1
                continue

            record = build_output_record(
                row=row,
                answer=answer,
                usage=usage,
                latency=latency,
                model=args.model,
                include_raw_prompt=args.include_raw_prompt,
                formatted_evidence=evidence_text,
            )
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()  # write immediately so crash = no data loss

            completed_keys.add(key)
            total_tokens += usage.get("total_tokens", 0)
            n_processed += 1

            if n_processed % args.checkpoint_every == 0:
                log.info(
                    "Progress: %d processed | %d skipped | %d failed | %d total tokens",
                    n_processed, n_skipped, n_failed, total_tokens,
                )

    log.info(
        "Done: %d processed | %d skipped (resumed) | %d failed | %d total tokens",
        n_processed, n_skipped, n_failed, total_tokens,
    )

    if args.output_csv:
        n_csv = write_csv_from_jsonl(output_path, Path(args.output_csv))
        log.info("Wrote %d rows to %s", n_csv, args.output_csv)

    return 0 if n_failed == 0 else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate answers from pre-retrieved evidence using the OpenAI API."
    )
    parser.add_argument(
        "--input", required=True,
        help="Input JSONL or CSV (columns: query_id, query, model_name, retrieved_evidence)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL path.  Appended to on resume.",
    )
    parser.add_argument("--output_csv", help="Also write a flat CSV at this path (written at end)")
    parser.add_argument(
        "--model", default="gpt-5.4-mini",
        help="OpenAI model (default: gpt-5.4-mini)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for reproducibility)",
    )
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per row (default: 3)")
    parser.add_argument("--retry_delay", type=float, default=2.0, help="Base retry delay in seconds (default: 2.0)")
    parser.add_argument(
        "--checkpoint_every", type=int, default=10,
        help="Log progress every N completed rows (default: 10)",
    )
    parser.add_argument(
        "--include_raw_prompt", action="store_true",
        help="Include the full raw user prompt in each output record",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

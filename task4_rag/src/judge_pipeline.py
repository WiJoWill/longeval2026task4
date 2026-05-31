"""LLM-as-judge evaluation pipeline using the Anthropic Claude API.

Reads model answers from CSV or JSONL, groups by query_id, and evaluates each
model's answer against the gold answer using Claude as the judge.

Usage:
  python -m task4_rag.src.judge_pipeline \\
    --input results.csv \\
    --output_csv judged_results.csv \\
    --output_jsonl judged_results.jsonl \\
    --model claude-sonnet-4-5

Input columns (CSV) or keys (JSONL):
  query_id, query, gold_answer, model_name, model_answer

Output CSV columns:
  query_id, query, gold_answer, model_name, factual_correctness, completeness, brief_rationale
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

from .run_task4 import _load_local_env

log = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """\
You are an impartial evaluator for a RAG-based scientific question-answering system.

Given:
- A natural language query
- A gold standard answer
- Multiple model-generated answers

For each model answer, evaluate it on two dimensions:

1. factual_correctness (1-5): How factually accurate is the answer compared to the gold answer?
   1 = Completely wrong or contradicts the gold answer
   2 = Mostly wrong with some correct elements
   3 = Partially correct, missing key facts or containing some errors
   4 = Mostly correct with minor errors or omissions
   5 = Fully correct and consistent with the gold answer

2. completeness (1-5): How complete is the answer relative to the gold answer?
   1 = Answer is missing almost all information from the gold answer
   2 = Answer covers a small portion of the gold answer
   3 = Answer covers about half of the gold answer
   4 = Answer covers most of the gold answer with minor omissions
   5 = Answer covers all key information from the gold answer

Return a JSON object with this exact schema:
{
  "query_id": "<the query ID>",
  "evaluations": [
    {
      "model_name": "<model name>",
      "factual_correctness": <integer 1-5>,
      "completeness": <integer 1-5>,
      "brief_rationale": "<1-2 sentence explanation>"
    }
  ]
}

Return only the JSON object. No markdown, no explanation outside the JSON."""


def _build_user_prompt(query_id: str, query: str, gold_answer: str, candidates: list[dict]) -> str:
    lines = [
        f"Query ID: {query_id}",
        f"Query: {query}",
        f"Gold Answer: {gold_answer}",
        "",
        "Model Answers to Evaluate:",
    ]
    for i, c in enumerate(candidates, 1):
        lines.append(f"\n{i}. Model: {c['model_name']}")
        lines.append(f"   Answer: {c['model_answer']}")
    return "\n".join(lines)


def _load_input(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows: list[dict] = []
    if path.endswith(".jsonl"):
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with p.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    return rows


def _group_by_query(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for row in rows:
        qid = str(row["query_id"])
        if qid not in groups:
            groups[qid] = {
                "query_id": qid,
                "query": str(row.get("query", "")),
                "gold_answer": str(row.get("gold_answer", "")),
                "candidates": [],
            }
        passthrough = {
            key: value
            for key, value in row.items()
            if key not in {"query_id", "query", "gold_answer", "model_name", "model_answer"}
        }
        groups[qid]["candidates"].append({
            "model_name": str(row.get("model_name", "")),
            "model_answer": str(row.get("model_answer", "")),
            "passthrough": passthrough,
        })
    return groups


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the code block content
        block = parts[1] if len(parts) > 1 else text
        if block.startswith("json"):
            block = block[4:]
        text = block.strip()
    return json.loads(text)


def _validate_evaluation(result: dict, expected_models: list[str]) -> list[str]:
    errors: list[str] = []
    if "query_id" not in result:
        errors.append("missing 'query_id'")
    if not isinstance(result.get("evaluations"), list):
        errors.append("missing or invalid 'evaluations' list")
        return errors
    returned_models = {e.get("model_name") for e in result["evaluations"]}
    for model in expected_models:
        if model not in returned_models:
            errors.append(f"missing evaluation for model '{model}'")
    for e in result["evaluations"]:
        for field in ("factual_correctness", "completeness"):
            val = e.get(field)
            if not isinstance(val, int) or not (1 <= val <= 5):
                errors.append(f"invalid {field}={val!r} for model '{e.get('model_name')}'")
    return errors


def _call_judge(
    client: Any,
    model: str,
    query_id: str,
    query: str,
    gold_answer: str,
    candidates: list[dict],
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict[str, Any]:
    import anthropic

    user_prompt = _build_user_prompt(query_id, query, gold_answer, candidates)
    expected_models = [c["model_name"] for c in candidates]

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text
            result = _parse_json_response(text)
            errors = _validate_evaluation(result, expected_models)
            if errors:
                log.warning("query %s attempt %d: validation errors: %s", query_id, attempt, errors)
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                log.error("query %s: using partial result after %d attempts", query_id, max_retries)
            result["query_id"] = query_id
            return result
        except json.JSONDecodeError as exc:
            log.warning("query %s attempt %d: JSON parse error: %s", query_id, attempt, exc)
            last_exc = exc
            if attempt < max_retries:
                time.sleep(retry_delay)
        except anthropic.RateLimitError as exc:
            log.warning("query %s attempt %d: rate limit, backing off %ds", query_id, attempt, int(retry_delay * attempt * 5))
            last_exc = exc
            time.sleep(retry_delay * attempt * 5)
        except anthropic.APIError as exc:
            log.warning("query %s attempt %d: API error: %s", query_id, attempt, exc)
            last_exc = exc
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                raise

    log.error("query %s: all retries exhausted, returning empty result (last error: %s)", query_id, last_exc)
    return {"query_id": query_id, "evaluations": []}


def _flatten_to_rows(result: dict, query: str, gold_answer: str) -> list[dict]:
    rows: list[dict] = []
    passthrough_by_model = {
        item.get("model_name", ""): item.get("passthrough", {})
        for item in result.get("_input_candidates", [])
    }
    for e in result.get("evaluations", []):
        row = {
            "query_id": result["query_id"],
            "query": query,
            "gold_answer": gold_answer,
            "model_name": e.get("model_name", ""),
            "factual_correctness": e.get("factual_correctness", ""),
            "completeness": e.get("completeness", ""),
            "brief_rationale": e.get("brief_rationale", ""),
        }
        row.update(passthrough_by_model.get(e.get("model_name", ""), {}))
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    parser = _build_parser()
    args = parser.parse_args(argv)
    _load_local_env()

    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
        return 1

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        return 1

    client = anthropic.Anthropic(api_key=api_key)

    log.info("Loading input from %s", args.input)
    rows = _load_input(args.input)
    groups = _group_by_query(rows)
    log.info("Loaded %d queries, %d total rows", len(groups), len(rows))

    all_flat_rows: list[dict] = []
    all_results: list[dict] = []
    n_total = len(groups)

    for i, (query_id, group) in enumerate(groups.items(), 1):
        log.info("[%d/%d] query_id=%s  models=%d", i, n_total, query_id, len(group["candidates"]))
        result = _call_judge(
            client=client,
            model=args.model,
            query_id=query_id,
            query=group["query"],
            gold_answer=group["gold_answer"],
            candidates=group["candidates"],
        )
        result["_input_candidates"] = group["candidates"]
        all_results.append(result)
        all_flat_rows.extend(_flatten_to_rows(result, group["query"], group["gold_answer"]))

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        base_fieldnames = [
            "query_id", "query", "gold_answer",
            "model_name", "factual_correctness", "completeness", "brief_rationale",
        ]
        extra_fieldnames = sorted({key for row in all_flat_rows for key in row if key not in base_fieldnames})
        fieldnames = base_fieldnames + extra_fieldnames
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_flat_rows)
        log.info("Wrote %d rows to %s", len(all_flat_rows), csv_path)

    if args.output_jsonl:
        jsonl_path = Path(args.output_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for result in all_results:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        log.info("Wrote %d records to %s", len(all_results), jsonl_path)

    if not args.output_csv and not args.output_jsonl:
        for result in all_results:
            print(json.dumps(result, ensure_ascii=False))

    log.info("Done. Evaluated %d queries.", len(all_results))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge evaluation pipeline using the Anthropic Claude API."
    )
    parser.add_argument(
        "--input", required=True,
        help="Input CSV or JSONL with columns: query_id, query, gold_answer, model_name, model_answer",
    )
    parser.add_argument("--output_csv", help="Output CSV path for flattened per-model scores")
    parser.add_argument("--output_jsonl", help="Output JSONL path for structured per-query results")
    parser.add_argument(
        "--model", default="claude-sonnet-4-5",
        help="Claude model to use for judging (default: claude-sonnet-4-5)",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

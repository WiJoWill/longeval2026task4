"""Phase 2 of the OpenAI Batch workflow: submit, poll, download, and process results.

Workflow
--------
  # Submit the merged requests file and wait for results:
  python -m task4_rag.src.apply_batch \\
      --requests-file outputs/test/batch_inputs/all_experiments_requests.jsonl \\
      --state-dir     outputs/test/batch_inputs \\
      --output-dir    outputs/test/runs

  # Or process an already-downloaded results file:
  python -m task4_rag.src.apply_batch \\
      --results-file  outputs/test/batch_inputs/batch_results.jsonl \\
      --state-dir     outputs/test/batch_inputs \\
      --output-dir    outputs/test/runs

Output
------
  outputs/test/runs/{run_id}.jsonl   — one file per experiment, TREC RAG format
  outputs/test/runs/_failed.jsonl    — records that had LLM errors (fallback applied)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .generator import _parse_answer_json, _repair_answer
from .output_writer import write_jsonl
from .run_task4 import _load_local_env
from .validator import validate_jsonl, validate_record


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _load_local_env()

    if not args.results_file and not args.requests_file:
        print("Error: provide --requests-file (to submit) or --results-file (already downloaded)", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_index = _load_state_index(Path(args.state_dir))
    if not state_index:
        print(f"Error: no *_state.json files found in {args.state_dir}", file=sys.stderr)
        return 1
    print(f"Loaded state for {len(state_index)} requests from {args.state_dir}")

    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = _submit_and_wait(
            requests_file=Path(args.requests_file),
            output_dir=output_dir,
            poll_interval=args.poll_interval,
        )
        if results_path is None:
            return 1

    print(f"\nProcessing results from {results_path} ...")
    records_by_run, failed = _process_results(
        results_path=results_path,
        state_index=state_index,
        max_answer_sentences=args.max_answer_sentences,
    )

    n_written = 0
    for run_id, records in sorted(records_by_run.items()):
        out_path = output_dir / f"{run_id}.jsonl"
        write_jsonl(records, out_path)
        errors = validate_jsonl(out_path)
        status = f"OK ({len(records)} records)" if not errors else f"{len(errors)} validation errors"
        print(f"  {run_id}: {out_path}  [{status}]")
        n_written += len(records)

    if failed:
        failed_path = output_dir / "_failed.jsonl"
        with failed_path.open("w", encoding="utf-8") as fh:
            for item in failed:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n  {len(failed)} failed requests written to {failed_path}")

    print(f"\nDone. Wrote {n_written} records across {len(records_by_run)} runs.")
    return 0


def _load_state_index(state_dir: Path) -> dict[str, Any]:
    """Load all *_state.json files into a flat custom_id → state dict."""
    index: dict[str, Any] = {}
    for state_file in state_dir.glob("*_state.json"):
        data = json.loads(state_file.read_text(encoding="utf-8"))
        for custom_id, state in data.get("queries", {}).items():
            index[custom_id] = state
    return index


def _submit_and_wait(
    requests_file: Path,
    output_dir: Path,
    poll_interval: int,
) -> Path | None:
    """Upload requests file, create batch, poll until done, download results."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("Error: openai package is not installed (pip install openai)", file=sys.stderr)
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set", file=sys.stderr)
        return None

    client = OpenAI(api_key=api_key)

    print(f"Uploading {requests_file} ...")
    with requests_file.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    print(f"  file_id: {uploaded.id}")

    print("Creating batch ...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_id = batch.id
    print(f"  batch_id: {batch_id}")

    # Persist batch_id so it can be reused if the script is interrupted.
    id_path = output_dir / "batch_id.txt"
    id_path.write_text(batch_id, encoding="utf-8")
    print(f"  saved to {id_path}")

    # Poll until terminal state.
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  status={batch.status}  "
            f"completed={counts.completed}/{counts.total}  "
            f"failed={counts.failed}"
        )
        if batch.status in terminal:
            break
        time.sleep(poll_interval)

    if batch.status != "completed":
        print(f"Error: batch ended with status '{batch.status}'", file=sys.stderr)
        return None

    print("Downloading results ...")
    content = client.files.content(batch.output_file_id)
    results_path = output_dir / "batch_results.jsonl"
    results_path.write_bytes(content.read())
    print(f"  saved to {results_path}")
    return results_path


def _process_results(
    results_path: Path,
    state_index: dict[str, Any],
    max_answer_sentences: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Parse batch output JSONL and build per-run record lists."""
    records_by_run: dict[str, list[dict[str, Any]]] = {}
    failed: list[dict[str, Any]] = []

    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        result = json.loads(line)
        custom_id: str = result.get("custom_id", "")

        # custom_id format: "{run_id}|{query_id}"
        if "|" not in custom_id:
            print(f"Warning: unexpected custom_id format '{custom_id}', skipping", file=sys.stderr)
            continue
        run_id, query_id = custom_id.split("|", 1)

        state = state_index.get(custom_id)
        if state is None:
            print(f"Warning: no state found for custom_id '{custom_id}', skipping", file=sys.stderr)
            continue

        references: list[str] = state["references"]
        sentence_candidates: list[dict[str, Any]] = state["sentence_candidates"]
        metadata: dict[str, Any] = state["metadata"]

        answer = _extract_answer(result, references, sentence_candidates, max_answer_sentences, custom_id, failed)

        record = {"metadata": metadata, "references": references, "answer": answer}
        try:
            validate_record(record)
        except Exception as exc:
            print(f"Warning: validation failed for {custom_id}: {exc}", file=sys.stderr)

        records_by_run.setdefault(run_id, []).append(record)

    return records_by_run, failed


def _extract_answer(
    result: dict[str, Any],
    references: list[str],
    sentence_candidates: list[dict[str, Any]],
    max_answer_sentences: int,
    custom_id: str,
    failed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    error = result.get("error")
    if error:
        print(f"  LLM error for {custom_id}: {error}", file=sys.stderr)
        failed.append({"custom_id": custom_id, "error": error})
        return [{"text": "No supported answer could be generated from the provided candidate documents.", "citations": []}]

    try:
        content = result["response"]["body"]["choices"][0]["message"]["content"]
        raw_answer = _parse_answer_json(content)
        return _repair_answer(raw_answer, references, max_answer_sentences, sentence_candidates)
    except Exception as exc:
        print(f"  Parse error for {custom_id}: {exc}", file=sys.stderr)
        failed.append({"custom_id": custom_id, "error": str(exc), "raw": result.get("response", {})})
        return [{"text": "No supported answer could be generated from the provided candidate documents.", "citations": []}]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 2: submit OpenAI batch, wait, download, and convert to run JSONL files."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--requests-file", help="Merged batch requests JSONL to submit (runs the full pipeline)")
    group.add_argument("--results-file",  help="Already-downloaded batch results JSONL (skip submission)")
    parser.add_argument("--state-dir",   required=True, help="Directory containing *_state.json files from prepare_batch")
    parser.add_argument("--output-dir",  required=True, help="Directory to write per-run JSONL files")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between status polls (default: 30)")
    parser.add_argument("--max-answer-sentences", type=int, default=5)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

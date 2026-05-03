"""Submit/process Gemini Batch results and convert them to Task 4 JSONL runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

from .generator import _parse_answer_json, _repair_answer
from .output_writer import write_jsonl
from .run_task4 import _load_local_env
from .validator import validate_jsonl, validate_record


TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _load_local_env()

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
        if not args.requests_file:
            print("Error: provide --requests-file or --results-file", file=sys.stderr)
            return 1
        results_path = _submit_and_wait(
            requests_file=Path(args.requests_file),
            output_dir=output_dir,
            model=args.model,
            display_name=args.display_name,
            poll_interval=args.poll_interval,
        )
        if results_path is None:
            return 1

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
        print(f"  {len(failed)} failed requests written to {failed_path}")

    print(f"Done. Wrote {n_written} records across {len(records_by_run)} runs.")
    return 0


def _submit_and_wait(
    requests_file: Path,
    output_dir: Path,
    model: str,
    display_name: str,
    poll_interval: int,
) -> Path | None:
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError:
        print("Error: google-genai is not installed (pip install google-genai)", file=sys.stderr)
        return None

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is not set", file=sys.stderr)
        return None

    client = genai.Client(api_key=api_key)

    print(f"Uploading {requests_file} to Gemini File API ...")
    uploaded = client.files.upload(
        file=str(requests_file),
        config=types.UploadFileConfig(display_name=requests_file.name, mime_type="jsonl"),
    )
    print(f"  file: {uploaded.name}")
    (output_dir / "uploaded_file.txt").write_text(str(uploaded.name), encoding="utf-8")

    print(f"Creating Gemini batch: model={model}")
    try:
        batch = client.batches.create(
            model=model,
            src=uploaded.name,
            config={"display_name": display_name},
        )
    except Exception as exc:
        error_path = output_dir / "batch_create_error.txt"
        error_path.write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        print(f"Error: Gemini batch creation failed; details saved to {error_path}", file=sys.stderr)
        return None
    batch_name = batch.name
    print(f"  batch: {batch_name}")
    (output_dir / "batch_id.txt").write_text(str(batch_name), encoding="utf-8")

    while True:
        batch = client.batches.get(name=batch_name)
        state = _state_name(batch)
        stats = getattr(batch, "batch_stats", None) or getattr(batch, "batchStats", None)
        print(f"  state={state} stats={stats}")
        if state in TERMINAL_STATES:
            break
        time.sleep(poll_interval)

    status_path = output_dir / "batch_status.json"
    status_path.write_text(_to_json(batch), encoding="utf-8")

    if _state_name(batch) != "JOB_STATE_SUCCEEDED":
        print(f"Error: Gemini batch ended with state {_state_name(batch)}", file=sys.stderr)
        return None

    result_file_name = _result_file_name(batch)
    if not result_file_name:
        print("Error: Gemini batch succeeded but no result file was found", file=sys.stderr)
        return None

    print(f"Downloading Gemini results from {result_file_name} ...")
    results_path = output_dir / "batch_results.jsonl"
    _download_file(result_file_name, api_key, results_path)
    print(f"  saved to {results_path}")
    return results_path


def _process_results(
    results_path: Path,
    state_index: dict[str, Any],
    max_answer_sentences: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    records_by_run: dict[str, list[dict[str, Any]]] = {}
    failed: list[dict[str, Any]] = []

    for line_no, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        result = json.loads(line)
        custom_id = _result_key(result)
        if not custom_id:
            failed.append({"line": line_no, "error": "missing result key", "raw": result})
            continue
        if "|" not in custom_id:
            failed.append({"custom_id": custom_id, "error": "unexpected key format", "raw": result})
            continue
        run_id, _query_id = custom_id.split("|", 1)
        state = state_index.get(custom_id)
        if state is None:
            failed.append({"custom_id": custom_id, "error": "missing state"})
            continue

        references: list[str] = state["references"]
        sentence_candidates: list[dict[str, Any]] = state["sentence_candidates"]
        metadata: dict[str, Any] = state["metadata"]
        answer = _extract_answer(result, references, sentence_candidates, max_answer_sentences, custom_id, failed)
        record = {"metadata": metadata, "references": references, "answer": answer}
        try:
            validate_record(record)
        except Exception as exc:
            failed.append({"custom_id": custom_id, "error": f"validation failed: {exc}"})
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
    error = result.get("error") or result.get("status")
    if error and not result.get("response"):
        failed.append({"custom_id": custom_id, "error": error})
        return [{"text": "No supported answer could be generated from the provided candidate documents.", "citations": []}]

    try:
        text = _response_text(result["response"])
        raw_answer = _parse_answer_json(text)
        return _repair_answer(raw_answer, references, max_answer_sentences, sentence_candidates)
    except Exception as exc:
        failed.append({"custom_id": custom_id, "error": str(exc), "raw": result.get("response", {})})
        return [{"text": "No supported answer could be generated from the provided candidate documents.", "citations": []}]


def _response_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates") or []
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    texts = [str(part.get("text", "")) for part in parts if part.get("text")]
    if not texts:
        raise ValueError("Gemini response contained no text")
    return "\n".join(texts)


def _result_key(result: dict[str, Any]) -> str:
    key = result.get("key")
    if isinstance(key, str):
        return key
    metadata = result.get("metadata") or {}
    key = metadata.get("key")
    return key if isinstance(key, str) else ""


def _state_name(batch: Any) -> str:
    state = getattr(batch, "state", "")
    return getattr(state, "name", None) or str(state)


def _result_file_name(batch: Any) -> str:
    dest = getattr(batch, "dest", None)
    file_name = getattr(dest, "file_name", None) if dest is not None else None
    if file_name:
        return str(file_name)
    response = getattr(batch, "response", None)
    file_name = getattr(response, "responses_file", None) if response is not None else None
    return str(file_name) if file_name else ""


def _download_file(file_name: str, api_key: str, output_path: Path) -> None:
    url = f"https://generativelanguage.googleapis.com/download/v1beta/{file_name}:download?alt=media"
    response = requests.get(url, headers={"x-goog-api-key": api_key}, timeout=300)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def _to_json(obj: Any) -> str:
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json(indent=2)
    if hasattr(obj, "to_json"):
        return obj.to_json()
    return json.dumps(str(obj), ensure_ascii=False, indent=2)


def _load_state_index(state_dir: Path) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for state_file in state_dir.glob("*_state.json"):
        data = json.loads(state_file.read_text(encoding="utf-8"))
        for custom_id, state in data.get("queries", {}).items():
            index[custom_id] = state
    return index


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit/process Gemini Batch and convert to Task 4 run JSONL.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--requests-file", help="Gemini batch request JSONL to submit")
    group.add_argument("--results-file", help="Already-downloaded Gemini batch results JSONL")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--display-name", default="longeval-task4-gold-fulltext")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--max-answer-sentences", type=int, default=5)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

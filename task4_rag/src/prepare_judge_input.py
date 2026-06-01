"""Build Claude judge input rows from gold and model run JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare LLM judge input with deterministic citation doc overlap metrics.")
    parser.add_argument("--gold-run", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--include", nargs="*", help="Optional run_id/model names to include.")
    args = parser.parse_args(argv)

    gold_records = _load_run(args.gold_run)
    run_paths = sorted(
        path
        for path in args.runs_dir.glob("*.jsonl")
        if not path.name.startswith("_") and path.name != "batch_results.jsonl"
    )
    if args.include:
        include = set(args.include)
        run_paths = [path for path in run_paths if path.stem in include]

    rows: list[dict[str, Any]] = []
    for run_path in run_paths:
        model_name = run_path.stem
        if model_name == args.gold_run.stem:
            continue
        model_records = _load_run(run_path)
        for query_id, gold in gold_records.items():
            model = model_records.get(query_id)
            if model is None:
                rows.append(_missing_row(gold, model_name))
                continue
            rows.append(_row(gold, model, model_name))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.output_csv)
    _write_jsonl(rows, args.output_jsonl)
    print(f"Wrote judge CSV: {args.output_csv} ({len(rows)} rows)")
    print(f"Wrote judge JSONL: {args.output_jsonl}")
    print(f"Queries: {len(gold_records)}; models: {len(run_paths)}")
    return 0


def _load_run(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for item in _read_jsonl(path):
        qid = str(item.get("metadata", {}).get("narrative_id", ""))
        if qid:
            records[qid] = item
    return records


def _row(gold: dict[str, Any], model: dict[str, Any], model_name: str) -> dict[str, Any]:
    gold_docs = _cited_doc_ids(gold)
    model_docs = _cited_doc_ids(model)
    overlap = sorted(gold_docs & model_docs)
    union = gold_docs | model_docs
    return {
        "query_id": gold["metadata"]["narrative_id"],
        "query": gold["metadata"]["narrative"],
        "gold_answer": _answer_text(gold),
        "model_name": model_name,
        "model_answer": _answer_text(model),
        "gold_cited_doc_ids": "|".join(sorted(gold_docs, key=_sort_key)),
        "model_cited_doc_ids": "|".join(sorted(model_docs, key=_sort_key)),
        "overlap_doc_ids": "|".join(sorted(overlap, key=_sort_key)),
        "gold_cited_doc_count": len(gold_docs),
        "model_cited_doc_count": len(model_docs),
        "overlap_doc_count": len(overlap),
        "overlap_pct_gold": _pct(len(overlap), len(gold_docs)),
        "overlap_pct_model": _pct(len(overlap), len(model_docs)),
        "overlap_jaccard_pct": _pct(len(overlap), len(union)),
    }


def _missing_row(gold: dict[str, Any], model_name: str) -> dict[str, Any]:
    gold_docs = _cited_doc_ids(gold)
    return {
        "query_id": gold["metadata"]["narrative_id"],
        "query": gold["metadata"]["narrative"],
        "gold_answer": _answer_text(gold),
        "model_name": model_name,
        "model_answer": "",
        "gold_cited_doc_ids": "|".join(sorted(gold_docs, key=_sort_key)),
        "model_cited_doc_ids": "",
        "overlap_doc_ids": "",
        "gold_cited_doc_count": len(gold_docs),
        "model_cited_doc_count": 0,
        "overlap_doc_count": 0,
        "overlap_pct_gold": 0.0,
        "overlap_pct_model": 0.0,
        "overlap_jaccard_pct": 0.0,
    }


def _cited_doc_ids(record: dict[str, Any]) -> set[str]:
    references = record.get("references", [])
    output: set[str] = set()
    for item in record.get("answer", []):
        for citation in item.get("citations", []):
            if isinstance(citation, int) and 0 <= citation < len(references):
                output.add(str(references[citation]))
    return output


def _answer_text(record: dict[str, Any]) -> str:
    return " ".join(str(item.get("text", "")).strip() for item in record.get("answer", []) if str(item.get("text", "")).strip())


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def _sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):020d}") if value.isdigit() else (1, value)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

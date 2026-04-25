"""CLI for analyzing Task 4 run files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import analyze_run, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a Task 4 TREC RAG run")
    parser.add_argument("--run", required=True, help="Path to submission JSONL")
    parser.add_argument("--queries", default="data/task4_longeval_rag-query_docids.jsonl", help="Path to Task 4 query-docid JSONL")
    parser.add_argument("--output-report", help="Optional path for a JSON report")
    args = parser.parse_args(argv)

    summary = analyze_run(args.run, args.queries)
    if args.output_report:
        write_report(summary, args.output_report)
    print(json.dumps(summary.report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

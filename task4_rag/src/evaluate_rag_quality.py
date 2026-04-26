"""CLI for reference-free RAG quality diagnostics."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .rag_quality_evaluator import analyze_rag_quality, write_rag_quality_report
from .run_task4 import _parse_doc_text_fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze RAG quality beyond structural JSONL validity")
    parser.add_argument("--run", required=True, help="Path to submission JSONL")
    parser.add_argument("--queries", default="data/task4_longeval_rag-query_docids.jsonl", help="Path to Task 4 query-docid JSONL")
    parser.add_argument("--documents", required=True, help="Path to document file or directory")
    parser.add_argument("--doc-text-fields", default="fullText|abstract|title", help="Document text field priority, separated by |")
    parser.add_argument("--output-report", help="Optional path for a JSON quality report")
    args = parser.parse_args(argv)

    report = analyze_rag_quality(
        run_path=args.run,
        query_path=args.queries,
        documents_path=args.documents,
        doc_text_fields=_parse_doc_text_fields(args.doc_text_fields),
    )
    if args.output_report:
        write_rag_quality_report(report, args.output_report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

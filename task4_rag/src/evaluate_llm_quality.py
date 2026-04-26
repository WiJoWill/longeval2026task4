"""CLI for optional LLM-as-judge RAG evaluation."""

from __future__ import annotations

import argparse
import json

from .llm_quality_evaluator import LLMJudgeConfig, analyze_with_llm_judge, write_llm_judge_report
from .run_task4 import _parse_doc_text_fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Task 4 RAG run with an optional LLM judge")
    parser.add_argument("--run", required=True, help="Path to submission JSONL")
    parser.add_argument("--queries", default="data/task4_longeval_rag-query_docids.jsonl")
    parser.add_argument("--documents", required=True, help="Path to document file or directory")
    parser.add_argument("--doc-text-fields", default="fullText|abstract|title")
    parser.add_argument("--provider", default="openai", choices=["openai"], help="LLM judge provider")
    parser.add_argument("--model", default="gpt-4o-mini", help="Judge model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-records", type=int, help="Limit records for smoke/cost control")
    parser.add_argument("--max-doc-chars", type=int, default=1200, help="Max characters per reference sent to the judge")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Delay between judge calls")
    parser.add_argument("--output-report", required=True, help="Path for JSON report")
    args = parser.parse_args(argv)

    report = analyze_with_llm_judge(
        run_path=args.run,
        query_path=args.queries,
        documents_path=args.documents,
        doc_text_fields=_parse_doc_text_fields(args.doc_text_fields),
        config=LLMJudgeConfig(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_records=args.max_records,
            max_doc_chars=args.max_doc_chars,
            sleep_seconds=args.sleep_seconds,
        ),
    )
    write_llm_judge_report(report, args.output_report)
    print(json.dumps(report["score_means"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

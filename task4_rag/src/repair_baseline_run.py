"""CLI for repairing the provided baseline JSONL into valid Task 4 format."""

from __future__ import annotations

import argparse
import json

from .baseline_repair import repair_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair a baseline Task 4 run")
    parser.add_argument("--input-run", required=True, help="Input JSONL run to repair")
    parser.add_argument("--queries", default="data/task4_longeval_rag-query_docids.jsonl", help="Task 4 query file")
    parser.add_argument("--output-run", required=True, help="Output repaired JSONL run")
    parser.add_argument("--team-id", default="our_team")
    parser.add_argument("--run-id", default="generated_responses_repaired_v1")
    parser.add_argument("--type", default="automatic")
    parser.add_argument("--max-answer-sentences", type=int, default=5)
    args = parser.parse_args(argv)

    result = repair_run(
        input_run_path=args.input_run,
        query_path=args.queries,
        output_run_path=args.output_run,
        team_id=args.team_id,
        run_id=args.run_id,
        run_type=args.type,
        max_answer_sentences=args.max_answer_sentences,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Split an OpenAI Batch JSONL into approximate token-budget shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split OpenAI batch requests by approximate input token budget.")
    parser.add_argument("--requests-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-approx-tokens", type=int, default=1_600_000)
    parser.add_argument("--prefix", default="shard")
    args = parser.parse_args(argv)

    requests = [json.loads(line) for line in args.requests_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    shards: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for request in requests:
        request_tokens = _approx_tokens(request)
        if current and current_tokens + request_tokens > args.max_approx_tokens:
            shards.append(current)
            current = []
            current_tokens = 0
        current.append(request)
        current_tokens += request_tokens
    if current:
        shards.append(current)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, shard in enumerate(shards, start=1):
        path = args.output_dir / f"{args.prefix}_{index:02d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for request in shard:
                handle.write(json.dumps(request, ensure_ascii=True))
                handle.write("\n")
        manifest.append(
            {
                "path": str(path),
                "requests": len(shard),
                "approx_tokens": sum(_approx_tokens(request) for request in shard),
                "custom_ids": [request.get("custom_id", "") for request in shard],
            }
        )
    manifest_path = args.output_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in manifest:
        print(f"{item['path']}: requests={item['requests']} approx_tokens={item['approx_tokens']:,}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


def _approx_tokens(request: dict[str, Any]) -> int:
    body = request.get("body", {})
    messages = body.get("messages", [])
    chars = 0
    for message in messages:
        chars += len(str(message.get("content", "")))
    # Keep a small overhead for JSON framing and schema.
    return (chars // 4) + 500


if __name__ == "__main__":
    raise SystemExit(main())

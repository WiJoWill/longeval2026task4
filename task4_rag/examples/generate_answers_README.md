# generate_answers — Answer Generation Pipeline

Standardizes answer generation across retrieval experiments.  Each retrieval
strategy produces evidence; this pipeline calls the same LLM for all of them,
so quality differences in the final output reflect retrieval, not generation.

---

## Setup

```bash
pip install openai
export OPENAI_API_KEY=sk-...
```

---

## Input format

JSONL (one object per line) or CSV.  Required fields:

| Field | Type | Description |
|---|---|---|
| `query_id` | string | Unique query identifier |
| `query` | string | Natural-language question |
| `model_name` | string | Retrieval strategy that produced the evidence |
| `retrieved_evidence` | list / JSON string | List of `{"doc_id": ..., "text": ...}` |

Optional fields: `retrieved_doc_ids`, `retrieved_token_count`, `metadata`

In CSV, `retrieved_evidence` must be a JSON-encoded string in the cell.

---

## Basic usage

```bash
python -m task4_rag.src.generate_answers \
  --input  task4_rag/examples/generate_answers_input.jsonl \
  --output outputs/answers.jsonl \
  --output_csv outputs/answers.csv \
  --model gpt-5.4-mini
```

---

## All CLI options

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Input JSONL or CSV |
| `--output` | required | Output JSONL (appended — safe to re-run) |
| `--output_csv` | — | Also write a flat CSV at this path |
| `--model` | `gpt-5.4-mini` | OpenAI model name |
| `--temperature` | `0.0` | Sampling temperature (keep low for reproducibility) |
| `--max_retries` | `3` | API retries per row |
| `--retry_delay` | `2.0` | Base retry delay in seconds |
| `--checkpoint_every` | `10` | Log progress every N rows |
| `--include_raw_prompt` | off | Add `raw_prompt` field to each output record |

---

## Resume after interruption

The `--output` JSONL is written one line at a time with an immediate flush.
On restart the pipeline reads the existing file, identifies completed
`(query_id, model_name)` pairs, and skips them automatically.

```bash
# First run (interrupted at row 47)
python -m task4_rag.src.generate_answers --input results.jsonl --output answers.jsonl

# Re-run — picks up from row 48, no duplicate API calls
python -m task4_rag.src.generate_answers --input results.jsonl --output answers.jsonl
```

---

## Output fields

| Field | Description |
|---|---|
| `query_id` | From input |
| `model_name` | Retrieval strategy name from input |
| `query` | From input |
| `generated_answer` | LLM-generated answer |
| `evidence_doc_ids` | `doc_id` values from evidence passages |
| `evidence_count` | Number of passages in evidence |
| `token_count` | Total tokens used (prompt + completion) |
| `generation_model` | Model name used for generation |
| `latency_seconds` | Wall-clock time for the API call |
| `prompt_tokens` | Prompt token count |
| `completion_tokens` | Completion token count |

---

## Using as a module

```python
from task4_rag.src.generate_answers import (
    format_evidence,
    build_user_prompt,
    call_openai,
    build_output_record,
    load_input,
)

passages = [{"doc_id": "doc_1", "text": "Some relevant text."}]
evidence_text = format_evidence(passages)
answer, usage, latency = call_openai(client, "gpt-5.4-mini", query, evidence_text)
```

---

## Example files

- `generate_answers_input.jsonl` — 4 rows, 2 queries × 2 retrieval models
- `generate_answers_output_example.jsonl` — expected output shape

# CLEF LongEval 2026 Task 4 RAG Pipeline

This package implements a research-grade, modular pipeline for **CLEF LongEval 2026 Task 4: LongEval-RAG**. It is designed for the official setup where each query comes with a fixed set of candidate document IDs and the answer must be generated only from those candidate documents.

The primary method is `caes_rag_rrf_v1`, short for **Citation-Aware Evidence Selection with RRF fusion**. The default path stays reproducible and offline: deterministic query expansion, lexical retrieval over the candidate set, reciprocal-rank fusion, heuristic reranking, temporal-aware evidence selection, and citation-valid extractive answer generation.

## What The Pipeline Does

For each query, the pipeline:

1. loads the query text and its fixed candidate `doc_id` set;
2. loads only the candidate documents from the configured corpus path;
3. splits documents into passages;
4. builds multiple retrieval queries from the original narrative;
5. retrieves passages with BM25-style lexical scoring;
6. fuses rankings with Reciprocal Rank Fusion (RRF);
7. reranks top evidence with overlap, title, temporal, and optional citation-graph features;
8. selects a compact, diverse evidence set;
9. generates TREC RAG JSONL with sentence-level citations.

## Supported Methods

- `concat_baseline`
  Simple baseline that concatenates early candidate passages and emits citation-safe extractive claims.

- `llm_all_docs`
  Sends all candidate passages to an LLM when configured. Falls back to extractive output if no LLM is configured.

- `hybrid_evidence_rag_v1`
  Main method. Performs multi-query retrieval, RRF fusion, reranking, citation-aware evidence selection, and concise answer generation.

- `extractive_evidence_only`
  Uses the same retrieval and evidence selection stack but returns extractive evidence sentences only.

## Data Inputs

The current default config expects local, non-committed inputs:

- queries: `data/task4_longeval_rag-query_docids.jsonl`
- documents: `data/snapshot3/longeval_sci_test-09-11_2026_fulltext/documents`
- citation graph prior: `data/longeval-sci-2026-citation-network.csv`

The query file can already contain both query text and candidate doc IDs, for example:

```json
{"query_id":"...","question":"...","doc_ids":[123,456]}
```

The document loader supports common fields including:

- IDs: `doc_id`, `docid`, `id`, `paper_id`
- text: `fullText`, `full_text`, `text`, `body`, `contents`, `abstract`
- timestamps: `publishedDate`, `publication_date`, `published`, `timestamp`, `date`, `createdDate`, `year`

By default, the loader prefers `fullText`, then `abstract`, then `title`.

## Output Format

Each line of the output JSONL follows TREC RAG format:

```json
{
  "metadata": {
    "team_id": "our_team",
    "run_id": "caes_rag_rrf_v1",
    "type": "automatic",
    "narrative": "<query text>",
    "narrative_id": "<query id>"
  },
  "references": ["docid_a", "docid_b", "docid_c"],
  "answer": [
    {
      "text": "A short factual sentence.",
      "citations": [0]
    }
  ]
}
```

`citations` are integer indices into `references`. The validator rejects malformed metadata, duplicate references, and out-of-range citations.

## Retrieval Design

The retrieval stack is intentionally stronger than a concat baseline while staying candidate-constrained:

- deterministic multi-query expansion from the query narrative;
- pseudo-relevance-feedback expansion from top early passages;
- BM25-style lexical retrieval over candidate passages;
- RRF fusion across query variants;
- optional dense scoring with scientific embeddings;
- optional cross-encoder reranking;
- temporal-aware evidence selection that tries to preserve earlier/later support when the query asks for progression over time;
- optional citation-graph prior from the provided citation network file.

This transfers a few useful instincts from longitudinal retrieval work:

- do not trust one query string only;
- fuse multiple reasonable retrieval views instead of overcommitting to one ranking;
- preserve temporal spread when the question is explicitly comparative or evolutionary;
- keep evidence compact before generation.

## Generation Design

The generator is citation-aware by construction:

- every answer item has `text` and `citations`;
- citations always point into the output `references` list;
- offline mode produces extractive or lightly templated evidence-grounded claims;
- temporal questions can produce earlier/later evidence sentences when timestamps are available;
- optional LLM mode is prompt-constrained to use only supplied evidence and return JSON.

## Configuration

Default config: `task4_rag/configs/task4_default.yaml`

Key knobs:

```yaml
retrieval:
  method: rrf
  enable_query_expansion: true
  enable_prf: true
  top_k_passages: 10
  rrf_k: 60
  dense_model:
  reranker_model:

generation:
  provider: none
  model:
  temporal_templates: true
```

Optional dense/scientific embedding choices:

- `allenai/specter2_base`
- `allenai/scibert_scivocab_uncased`
- `BAAI/bge-m3`

Optional reranker examples:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Commands

Run the main method:

```bash
python -m task4_rag.src.run_task4 \
  --config task4_rag/configs/task4_default.yaml \
  --output runs/caes_rag_rrf_v1.jsonl
```

Run only a few queries during debugging:

```bash
python -m task4_rag.src.run_task4 \
  --config task4_rag/configs/task4_default.yaml \
  --max-queries 2 \
  --output runs/debug.jsonl
```

Validate a run:

```bash
python -m task4_rag.src.run_task4 \
  --config task4_rag/configs/task4_default.yaml \
  --validate-only \
  --output runs/caes_rag_rrf_v1.jsonl
```

Shell wrappers are in `task4_rag/scripts/`.

## Baseline Analysis

You can analyze a baseline or submitted run against the real Task 4 query file with:

```bash
python -m task4_rag.src.evaluate_run \
  --run data/generated-responses.jsonl \
  --queries data/task4_longeval_rag-query_docids.jsonl
```

This reports:

- query coverage;
- format validity;
- exact/set match of `references` against the official candidate `doc_ids`;
- empty-citation answer rate;
- filler-answer rate;
- answer/reference length statistics.

The provided `data/generated-responses.jsonl` is useful as a baseline floor because it matches the candidate reference lists but still fails important Task 4 requirements such as complete metadata and citation-grounded answer writing.

## Ablation Matrix

The repo now includes these experiment configs:

- `task4_rag/configs/task4_concat_baseline.yaml`
- `task4_rag/configs/task4_single_query_bm25.yaml`
- `task4_rag/configs/task4_rrf_no_rerank.yaml`
- `task4_rag/configs/task4_rrf_rerank.yaml`

These let you isolate the impact of:

- no retrieval stack at all (`concat_baseline`);
- single-query lexical evidence ranking;
- multi-query expansion plus RRF fusion without reranking;
- the full retrieval stack with reranking and citation-aware evidence selection.

## Important Corpus Check

The runner now raises an error if none of the candidate document IDs can be found in the configured corpus path. This is deliberate: a query file and a snapshot directory must refer to the same underlying document ID space. If they do not match, retrieval and grounded answer generation are impossible.

## Relationship To RAG Guidance

The Prompt Engineering Guide's RAG overview emphasizes using external retrieved knowledge instead of relying on static parametric memory and treating retrieved documents as the grounding context for generation. It frames RAG as a way to improve factual consistency and reliability on knowledge-intensive tasks by retrieving supporting documents and using them as generation context. That is the basic shape here, but this package goes further for LongEval Task 4 by adding multi-query retrieval, fusion, reranking, citation discipline, and temporal-aware evidence selection. Source: https://www.promptingguide.ai/techniques/rag

For evaluation, the RAG guide itself does not require ChatGPT or Claude as judges. In this repo, the recommended order is:

1. automatic validation and structural metrics first;
2. retrieval/evidence diagnostics next;
3. optional LLM-as-judge analysis only after the run is structurally valid and grounded.

An LLM judge can be useful later for pairwise helpfulness or faithfulness comparisons, but it should be optional rather than the primary metric for Task 4.

## Literature Notes

- Lewis et al. 2020, **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**.
- Izacard and Grave 2021, **Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering**.
- Gao et al. 2023, **Enabling Large Language Models to Generate Text with Citations** / ALCE.
- Rashkin et al. 2023, **Measuring Attribution in Natural Language Generation Models** / AIS.
- Beltagy et al. 2019, **SciBERT: A Pretrained Language Model for Scientific Text**.
- Cohan et al. 2020, **SPECTER: Document-level Representation Learning using Citation-informed Transformers**.
- Cancellieri et al. 2025, LongEval longitudinal evaluation overview.
- Kanhabua et al. 2015, temporal information retrieval survey; see also Piryani et al. 2025 for later temporal IR survey work.

## Tests

```bash
pytest task4_rag/tests
```

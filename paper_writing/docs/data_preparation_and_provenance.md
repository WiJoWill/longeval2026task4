# Data Preparation and Provenance Tracking

Although the data preparation stage is relatively simple, it is still important to describe it explicitly because the entire retrieval, generation, and evaluation pipeline depends on correct candidate-document alignment and reliable evidence provenance.

## Query File Format

The query input is stored as JSONL, with one query per line. Each record contains:

- a query identifier;
- the query text;
- the official candidate document IDs for that query.

A minimal example is:

```json
{"query_id":"q-real-1","question":"How did the later study strengthen the earlier validation?","doc_ids":[101,202]}
```

In the implementation, the loader supports several common field aliases for robustness. For example, query IDs may appear as `narrative_id`, `query_id`, `qid`, or `id`, while query text may appear as `narrative`, `query`, `text`, or `question`.

## Document File Format

The document input is also read from JSONL, JSON, CSV, TSV, or a directory containing such files. Each document record should contain:

- a document identifier;
- a title;
- a usable text field, preferably `fullText`;
- an optional publication or timestamp field.

A minimal example is:

```json
{"id":"101","title":"Earlier study","abstract":"Earlier abstract.","fullText":"Full text evidence from the earlier study describing the initial validation setup.","publishedDate":"2024-05-01"}
{"id":"202","title":"Later study","abstract":"Later abstract.","fullText":"Full text evidence from the later study describing additional validation analysis.","publishedDate":"2025-06-15"}
```

As with the query loader, the document loader supports multiple field aliases. For document IDs, it accepts fields such as `doc_id`, `docid`, `id`, or `paper_id`.

## Candidate-Constrained Join Logic

The join between queries and documents is fully candidate-constrained.

The loading procedure is:

1. read all query records;
2. extract the candidate `doc_ids` associated with each query;
3. collect the union of all candidate document IDs needed for the run;
4. load only those documents from the document collection;
5. for each query, attach documents back in the order defined by its candidate list.

This means the system never retrieves outside the official candidate set. All later stages, including chunking, retrieval, reranking, and answer generation, operate only within the candidate pool supplied for each query.

## Text Selection Logic

For document text, the loader follows a simple priority rule:

`fullText -> abstract -> title`

Whenever `fullText` is available, it is used as the main evidence source. If `fullText` is missing or empty, the loader falls back to `abstract`. If both are unavailable, the loader can still fall back to `title` so that the record remains visible to the pipeline.

This fallback should be understood as a data-availability decision rather than a preferred methodological choice. Full text is always preferred when present.

## Citation and Document-ID Provenance

An important engineering feature of the pipeline is deterministic provenance tracking from answer citations back to source document IDs.

After retrieval and evidence selection, the system converts the selected evidence passages into an ordered `references` list. This list contains deduplicated document IDs in first-evidence order.

The final answer does not cite raw document IDs directly. Instead, each answer sentence stores citation indices into the `references` list.

The reconstruction path is therefore:

`citation index -> references[index] -> document ID`

This design ensures that every answer citation in the final JSONL output can be traced back to a specific supporting document.

## Provenance in Batch LLM Generation

For batch LLM generation, provenance tracking includes one additional layer.

Instead of sending full run records to the model, the batch pipeline sends compact sentence-level evidence items identified by `evidence_id`. At the same time, it stores a state file containing:

- the `references` list;
- the sentence candidates;
- the citation mapping for each evidence item;
- the corresponding source document IDs.

When the model returns `evidence_ids`, the pipeline reconstructs final citations by mapping:

`evidence_id -> sentence candidate -> citation index -> references[index] -> document ID`

This allows asynchronous batch generation while preserving exact evidence provenance.

## Pre-Model Engineering Contributions

Several engineering decisions happen before any retrieval or generation model is applied, and they are worth mentioning in the paper:

- flexible schema normalization for heterogeneous query and document files;
- candidate-constrained document loading rather than unrestricted corpus loading;
- full-text-first text composition with abstract fallback when necessary;
- deterministic provenance tracking from answer citations back to document IDs;
- passage construction before retrieval, so ranking happens over evidence units rather than whole documents;
- stateful batch reconstruction, which preserves evidence-to-document mapping even in asynchronous generation workflows.

## Short Paper-Style Summary

The data preparation stage normalizes query and document schemas into a unified internal format. Each query contains an identifier, the query text, and the official candidate document IDs. Documents are loaded only if their IDs appear in the candidate pool of at least one query, which keeps the entire pipeline candidate-constrained by design. For document text, we prioritize `fullText` and fall back to `abstract` only when full text is unavailable. After retrieval and evidence selection, we maintain deterministic provenance by converting selected evidence into an ordered `references` list of document IDs, while answer citations are stored as integer indices into that list. This makes every cited answer sentence traceable back to its supporting source document.

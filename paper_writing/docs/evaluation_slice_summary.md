# Evaluation Slice Summary

For the current evaluation slice, the system is evaluated on **47 queries**. Each query is paired with the official **10 candidate documents**, so the slice contains **470 query-document instances** in total.

The candidate count is therefore fixed at **10 documents per query** throughout this slice. This fixed-width candidate setting is important because all retrieval, reranking, and answer generation remain strictly constrained to the provided candidate pool rather than an open corpus.

## Text Availability

Document text is loaded with the priority rule:

`fullText -> abstract -> title`

Within the 470 query-document instances in the current slice:

- **222 / 470 (47.2%)** use **full text**;
- **248 / 470 (52.8%)** fall back to **abstract only** because no usable full text is available;
- **0 / 470 (0.0%)** require **title-only** fallback.

There are therefore no title-only cases in the present evaluation slice, and no candidate documents are dropped for missing text.

## Role of Timestamps

Timestamps are **not used only for provenance or post hoc analysis**. In the current retrieval pipeline, they are also used as an **active retrieval/reranking feature** when the query has temporal intent.

More concretely:

- document timestamps are loaded and preserved in the internal document representation;
- the evidence reranker includes a configurable `temporal_boost` feature;
- for temporally phrased queries, evidence selection can explicitly encourage coverage of earlier and later documents;
- the extractive generator can also build temporal comparison sentences from timestamp-derived years.

At the same time, timestamps are **not exposed directly in the final batch LLM prompts** used in the `openai_batch_gpt54_top10` evaluation inputs (`prompt_with_timestamp = 0`). In other words, timestamps influence retrieval-side evidence selection in the main pipeline, but they are not shown verbatim to the answer-generation model in that batch setting.

## Short Paper-Style Summary

The current evaluation slice contains 47 queries, each paired with 10 official candidate documents. Across the resulting 470 query-document instances, 47.2% provide usable full text and 52.8% require abstract fallback, while no title-only fallback cases occur. Timestamps are preserved for provenance, but they are not merely metadata: in the main retrieval configurations they also function as temporal retrieval/reranking signals, although they are not directly passed to the batch LLM prompts.

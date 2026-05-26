# Passage Selection, Evidence Budget, and Chunking Summary

This note summarizes the implemented defaults in the Task 4 RAG pipeline for passage selection, evidence budgeting, and document chunking.

## Passage Selection and Evidence Budget

### How many passages are selected?

For the main RAG runs, the retriever selects `top_k_passages = 20` evidence passages per query after ranking and reranking. Before that final selection step, it keeps a `preselect_k = 24` shortlist for reranking.

### Is the budget fixed per query?

Yes, in the main experiment configs the retrieval-side budget is fixed per query:

- final selected passages: `20`
- preranking shortlist: `24`
- PRF seed passages: `4`

The only exception is the `llm_all_docs` / gold-style setup, which bypasses retrieval-side evidence truncation and passes all passages from all candidate documents.

### How many passages per document?

There is no fixed per-document quota in the default configs. The selector can take multiple passages from the same document until the overall `top_k_passages` budget is filled. A configurable `per_doc_limit` exists in the code, but it is unset in the main configs, so the effective behavior is:

- no hard per-document cap by default;
- near-duplicate passages are filtered;
- for temporal queries, earliest and latest dated documents may be forced into the selected set before filling the remaining slots.

### How much total evidence is passed to the generator?

This depends on which stage is meant by "evidence passed to the generator."

At the retrieval stage, the generator receives the `20` selected passages.

At the actual LLM prompt stage, the generator does **not** send all 20 passages verbatim. Instead, it:

1. extracts up to the first `5` candidate sentences from each selected passage;
2. filters and scores those sentences;
3. keeps at most `max_selected_answer_candidates = 10` sentence-level evidence items;
4. sends those `10` evidence items to the LLM as the evidence payload.

So the practical LLM evidence budget in the main runs is **up to 10 sentence-level evidence items per query**, not 20 full passages.

The answer budget is separate: `max_answer_sentences = 5`.

## Chunking

### Rule-based chunking: window sizes and overlaps

The default rule-based chunker uses:

- `passage_max_words = 140`
- `passage_stride_sentences = 1`

Operationally, it scans a document sentence by sentence, accumulates sentences until adding the next sentence would exceed 140 words, emits that chunk, and then starts the next chunk while keeping the last `1` sentence from the previous chunk as overlap.

So the rule-based chunker is a variable-length sentence window with:

- maximum chunk size: about `140` word tokens;
- overlap: `1` sentence between adjacent chunks.

### How are semantic chunks formed?

Semantic chunking first splits the document into sentences, embeds each sentence, and then scans left to right while maintaining the current chunk representation as the average embedding of the sentences already in that chunk.

A new sentence is merged into the current chunk unless one of the following triggers a split:

- cosine similarity between the new sentence embedding and the current chunk-average embedding falls below `semantic_merge_threshold = 0.72`;
- adding the sentence would push the chunk above `140` words;
- an optional maximum sentence cap would be exceeded.

In the main semantic configs:

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- merge threshold: `0.72`
- minimum sentences before a similarity-based split: `1`
- no explicit maximum sentence count is set

### How are topic-shift chunks defined concretely?

Topic-shift chunking also starts from sentence segmentation and sentence embeddings, but it uses **local drift between adjacent sentences** rather than similarity to the running chunk average.

For sentence `i`, it computes:

`drift = 1 - cosine(embedding[i-1], embedding[i])`

A new chunk starts when:

- the current chunk already has at least `1` sentence, and
- the adjacent-sentence drift exceeds `topic_shift_boundary_threshold = 0.18`,

or when the size constraints are hit:

- adding the sentence would exceed `140` words;
- an optional maximum sentence cap would be exceeded.

In the main topic-shift configs:

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- boundary threshold: `0.18`
- minimum sentences before a drift-based split: `1`
- no explicit maximum sentence count is set

## Short Paper-Style Summary

In the main Task 4 RAG pipeline, each query receives a fixed retrieval-side evidence budget of 20 selected passages, drawn from a reranked shortlist of 24 candidates. There is no default hard cap on how many of those passages may come from one document, although duplicate-like passages are suppressed and temporal queries may force coverage of earliest and latest dated documents. The LLM itself sees a smaller sentence-level evidence budget: the pipeline extracts candidate sentences from the selected passages and passes at most 10 evidence sentences to the model, while constraining the final answer to 5 sentences. For chunking, the rule-based baseline uses approximately 140-word sentence windows with a 1-sentence overlap. The semantic variant merges adjacent sentences when their embedding similarity to the current chunk remains at least 0.72, while the topic-shift variant starts a new chunk when adjacent-sentence semantic drift exceeds 0.18; both variants also respect the same 140-word size cap.

# Paper Revision Plan After Official LongEval-RAG Results

This memo compares the current draft `CLEF_2026_paper_56.pdf` with the official organizer report `analysis.pdf`, and suggests how to redirect the paper from mainly exploratory local analysis toward methodology-driven interpretation of the official results.

## 1. Main Message To Change

The current paper says the clearest winner is `rrf_no_rerank` under the local Claude-judge protocol. This is no longer the right headline once the official organizer results are available.

Recommended new headline:

> Under the official LongEval-RAG evaluation, the strongest overall variant is `rule-minilm`: a rule-based chunking pipeline with query expansion, PRF, RRF, lightweight reranking, citation prior, and MiniLM sentence-level reranking. Its advantage is not mainly lexical ROUGE, but balanced citation precision, semantic similarity, and nugget coverage.

This lets the paper become less about "exploring many pipeline variants" and more about a methodological lesson:

> In candidate-constrained RAG, simple rule-based passage construction remains robust, but adding a compact sentence-level neural reranker can improve the final evidence allocation when the upstream chunks are stable.

## 2. Official Results To Add

Precede the current local-judge Table 4 with an official-results table. 
Important interpretation:

- `rule-minilm` is the best balanced official run: best BERTScore, best retrieval precision, and best nugget coverage.
- `topic-shift-current` has the best ROUGE-L, so topic-shift chunking may improve lexical overlap with the reference answer.
- `topic-shift-minilm` has the best TFC1, suggesting MiniLM-selected sentences contain more query-term evidence, but this does not translate into best nugget coverage.
- All submitted systems are far above the official naive baseline, especially on retrieval precision.

## 3. How To Reframe The Abstract

Current abstract says the largest separation is between retrieval systems and concatenation, and that `rrf_no_rerank` is the best local-judge run. Revise it to emphasize official results.

Suggested replacement idea:

> Official evaluation shows that the rule-based chunking plus MiniLM sentence-reranking variant achieves the strongest balanced performance, obtaining the highest BERTScore, retrieval precision, and nugget coverage among our submissions. The result suggests that the main improvement did not come from more complex semantic or topic-shift chunking, but from pairing stable rule-based evidence units with sentence-level neural selection before generation. Local LLM-judge experiments remain useful for diagnosis, but they rank variants differently from the official gold-answer and nugget-based evaluation, highlighting the importance of multi-metric RAG evaluation.

## 4. Section-by-Section Edits

### Introduction

Revise the contribution list:

1. Keep the candidate-constrained provenance pipeline contribution.
2. Replace "cross-provider evaluation protocol" as the central empirical contribution with "official multi-metric evaluation plus local diagnostic evaluation."
3. Add a new contribution: identifying that `rule-minilm` gives the best official balance of semantic answer similarity, citation precision, and nugget coverage.

Suggested sentence:

> The official results show that our best-performing configuration is not the most complex chunking variant, but a rule-based chunking pipeline augmented with MiniLM sentence reranking, which achieves the highest BERTScore, retrieval precision, and nugget coverage among our submissions.

### System Pipeline

The current method section is mostly good. Add more emphasis to why `rule-minilm` is methodologically meaningful:

- Rule chunks are stable, local, and less sensitive to embedding-boundary errors.
- RRF, query expansion, and PRF broaden passage recall inside the fixed candidate pool.
- MiniLM is applied late, at sentence-selection time, where the comparison unit is smaller and closer to the final answer sentence.
- This late neural reranking may be safer than using embeddings to redefine all chunks upstream.

Suggested paragraph:

> The `rule-minilm` variant separates passage construction from neural sentence selection. Passage boundaries remain deterministic and overlap-preserving, while MiniLM is used only after candidate evidence has been narrowed to sentence-level answer candidates. This design reduces the risk that embedding-based segmentation will discard useful local context, while still allowing a neural model to prioritize sentences that are semantically close to the query.

### Experimental Variants

Keep the method matrix, but add a column or note distinguishing "chunk-level MiniLM" from "sentence-level MiniLM." In the current draft, MiniLM appears both in semantic/topic chunking and in sentence reranking; this can confuse readers.

Clarify terminology:

- `current` = no MiniLM sentence reranker.
- `minilm` = same retrieval/chunking family, plus `cross-encoder/ms-marco-MiniLM-L-6-v2` sentence reranking.
- `semantic` and `topic-shift` use `all-MiniLM-L6-v2` embeddings for chunking, which is different from the cross-encoder sentence reranker.

### Evaluation Protocol

This section needs the biggest structural change.

Recommended order:

1. Official organizer evaluation: ROUGE-L, BERTScore, retrieval precision, nugget coverage, TFC1.
2. Local diagnostic evaluation: Claude judge, overlap diagnostics, structural validity.
3. Explain that the official evaluation is the primary result for the paper; local judging is used to understand disagreements and failure modes.

Suggested sentence:

> After receiving the official organizer evaluation, we treat the organizer metrics as the primary empirical evidence and use our local LLM-judge protocol as a diagnostic supplement rather than as the main ranking criterion.

### Results And Analysis

Replace the current claim "lexical fusion variants perform best" with a more nuanced official-results reading:

1. `rule-minilm` is the strongest balanced system.
2. Rule-based chunking remains competitive and may be more robust than semantic/topic chunking.
3. MiniLM sentence reranking helps in several official metrics, but its effect depends on the upstream chunking method.
4. Local LLM judging and official evaluation disagree, especially for `rrf_no_rerank`; this is itself an important RAG evaluation finding.

Suggested new subsections:

- "Official Evaluation Favors Rule-Based Chunking With Late MiniLM Reranking"
- "Different Metrics Reward Different Behaviors"
- "Current vs. MiniLM: Late Sentence Reranking Is Helpful But Not Uniform"
- "Why Local LLM Judging Ranked RRF Differently"

## 5. Current vs. MiniLM Discussion

Add a focused comparison because it is one of the most useful lessons from the official table.

### Semantic family

`semantic-minilm` vs. `semantic-current`:

- ROUGE-L improves: 0.160 vs. 0.156.
- BERTScore improves: 0.157 vs. 0.150.
- Nugget coverage improves slightly: 0.291 vs. 0.282.
- TFC1 improves strongly: 0.349 vs. 0.262.
- Retrieval precision drops: 0.922 vs. 0.933.

Interpretation:

> In the semantic-chunking family, MiniLM sentence reranking appears to improve answer-content matching and query-term evidence, but may select from a slightly noisier document set or cite marginally less precise evidence.

### Topic-shift family

`topic-shift-minilm` vs. `topic-shift-current`:

- ROUGE-L drops: 0.157 vs. 0.167.
- BERTScore improves: 0.163 vs. 0.156.
- Retrieval precision improves: 0.940 vs. 0.929.
- Nugget coverage is similar/slightly lower: 0.288 vs. 0.292.
- TFC1 improves strongly: 0.373 vs. 0.215.

Interpretation:

> In the topic-shift family, MiniLM improves semantic similarity, citation precision, and query-term concentration, but the non-MiniLM version better matches the lexical shape of the gold answer. This suggests that MiniLM changes the style and focus of selected evidence rather than simply improving every metric.

### Rule family

`rule-minilm` should be contrasted with `caes-rag-rrf`, `default`, and `rrf-no-rerank`:

- It beats `caes-rag-rrf` on BERTScore, precision, nugget coverage, and TFC1.
- It beats `rrf-no-rerank` on BERTScore, precision, nugget coverage, and TFC1, despite similar ROUGE-L.
- It is not the best ROUGE-L method, but it is the most convincing across evidence and content metrics.

Interpretation:

> The best official result comes from using MiniLM after rule-based chunking, not from using embedding similarity to define the chunks themselves. This points to sentence-level allocation as a more effective intervention than upstream semantic segmentation in the current pipeline.

## 6. Discussion Points To Add

### Why `rule-minilm` likely works well

Add these hypotheses:

- Rule chunks preserve local sentence context with predictable overlap.
- Query expansion, PRF, and RRF produce a diverse lexical candidate pool.
- Lightweight reranking and citation prior keep evidence candidate-constrained.
- MiniLM sentence reranking operates late, where it can choose answer-worthy sentences without changing document segmentation.
- This division of labor may avoid error propagation from semantic chunking while still benefiting from neural matching.

### Why semantic/topic chunking did not dominate

Possible explanations:

- The candidate pool is already small: 10 documents per query, so chunk boundaries matter less than final sentence allocation.
- Embedding-based chunking may split or merge scientific text in ways that are locally coherent but not optimized for the gold answer.
- Scientific abstracts and full texts often contain dense terminology; lexical overlap and title/body signals may remain strong.
- Semantic chunking can improve some metrics while harming others, so it needs tuning against the official metric target.

### Why official and local rankings differ

The local Claude judge ranked `rrf_no_rerank` first, but official metrics favor `rule-minilm`.

Use this as an evaluation insight:

> Local LLM judging rewarded broadly plausible answer quality, while the official metrics separately measure gold-answer overlap, semantic similarity, cited-document precision, nugget coverage, and query-term axioms. The disagreement shows that single-judge evaluation can hide improvements in evidence precision and nugget coverage.

## 7. Concrete Claims To Soften Or Remove

Revise these current-draft claims:

- "The top-scoring run is the lexical-fusion variant..."  
  Replace with: "Under local LLM judging, lexical fusion was strongest; under official evaluation, `rule-minilm` is strongest overall."

- "Semantic chunking, topic-shift chunking, and sentence-level reranking do not produce a stable gain..."  
  Replace with: "Semantic and topic-shift chunking do not dominate, but sentence-level MiniLM reranking is beneficial when paired with rule-based chunks."

- "Future gains are most likely to come from better semantic evidence discrimination..."  
  Keep, but make more specific: "especially late-stage sentence allocation and metric-aware reranking."

## 8. Suggested New Conclusion

Possible replacement conclusion:

> The official LongEval-RAG results refine the main lesson of our study. Candidate-constrained provenance control is necessary, but not sufficient; the largest gains come from selecting answer-worthy evidence inside the fixed candidate set. Among our variants, the strongest official configuration is `rule-minilm`, which combines deterministic rule-based chunks with late MiniLM sentence reranking. This method achieves the best BERTScore, retrieval precision, and nugget coverage among our submissions, suggesting that neural reranking is most useful when applied after stable lexical retrieval and passage construction. At the same time, metric disagreements between local LLM judging and official evaluation show that RAG systems should be assessed with multiple complementary measures rather than a single judge score.

## 9. Priority Edit Checklist

1. Replace the abstract result sentence with the official `rule-minilm` finding.
2. Add the official-results table near the start of Section 6.
3. Move the Claude judge table to a supplemental/diagnostic subsection.
4. Add a `current` vs. `minilm` subsection.
5. Reinterpret `rrf_no_rerank` as a strong local-judge baseline, not the official winner.
6. Expand the discussion of `rule-minilm` methodology.
7. Update the conclusion to emphasize late sentence reranking and multi-metric evaluation.


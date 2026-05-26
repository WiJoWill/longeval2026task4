# Final Result Analysis

We can draw several clear and defensible conclusions from the final results.

## Main Conclusions

1. `concat_baseline` is clearly not sufficient.

   Under the Claude judge, its `avg score` is only `1.446`, and its `mean Jaccard` is only `12.447%`, far below the retrieval-based methods. This shows that simply concatenating a few early candidate passages, without a serious retrieval and evidence-selection stack, is not enough for this task.

2. The main performance gain comes from the retrieval stack rather than from later complex chunking or sentence reranking.

   The most important improvement happens when moving from `single_query_bm25` to `rrf_no_rerank`. This indicates that multi-query expansion, PRF, and RRF fusion are the strongest current sources of improvement.

   In fact, `rrf_no_rerank_openai_llm_v1` is ranked first by the Claude judge, with `mean avg score = 2.734`, higher than `caes_rag_rrf_openai_llm_v1` at `2.649`.

3. More complex reranking, semantic chunking, and topic-shift chunking do not yet provide a stable final-answer benefit.

   From the judge summary, the top-performing methods are still dominated by the rule-based retrieval path:

   - `rrf_no_rerank`
   - `rule_minilm`
   - `single_query_bm25`
   - `caes_rag_rrf`
   - `default`

   By contrast, the `semantic_*` and `topic_shift_*` variants consistently appear lower in the ranking. On this 47-query evaluation set, more complex chunk boundaries and sentence-level reranking do not yet translate into a stable gain in final answer quality.

4. `default` and `caes_rag_rrf` are indeed very close, which makes this control comparison useful.

   Their judge scores are nearly identical:

   - `caes_rag_rrf`: `2.649`
   - `default`: `2.639`

   Their evidence overlap statistics are also very similar:

   - gold overlap: `72.589` vs `74.716`
   - Jaccard: `72.021` vs `73.794`

   This supports the expectation that, under the same retrieval pipeline, LLM generation introduces some variation, but not enough to create a method-level difference. Keeping `default` therefore provides a meaningful same-retrieval stability control.

5. The system is structurally mature, but semantic relevance remains the main bottleneck.

   The structural metrics are now strong across the major runs:

   - all major runs have `invalid = 0`
   - `reference_subset_match = 1.0`
   - `unsupported_rate = 0.0`

   This means citation format, candidate compliance, and basic grounding are already under control.

   However, the final judge scores are still modest. Even the best method reaches only `2.734/5` on average. This suggests that the main remaining problem is no longer format validity, but whether the system retrieves the most relevant evidence and turns it into a complete and semantically correct answer.

6. The automatic proxy metrics and the LLM judge do not fully agree, which is itself an important result.

   For example, the proxy-based evaluation gives relatively high `answer_relevance` to some semantic variants, such as `semantic_current` and `semantic_minilm`, compared with `caes_rag_rrf`.

   However, the Claude judge does not rank these methods at the top.

   This suggests that lexical or proxy-style relevance metrics can capture part of the improvement, but may also overestimate answers that appear query-related on the surface while still containing off-topic snippets, OCR artifacts, or weak cross-document synthesis. The LLM judge is more sensitive to these quality failures.

## Most Defensible Takeaway

If we reduce the findings to one core statement, it would be this:

> The main performance gain comes from strengthening candidate-constrained retrieval with multi-query expansion and rank fusion. By contrast, more complex downstream components such as semantic/topic-shift chunking and sentence-level reranking do not yet produce a consistent final-answer benefit on the 47-query evaluation set.

## What The Rankings Suggest

Based on the current results, the strongest set of methods to retain in the main comparison is:

- `single_query_bm25` as a strong lexical baseline
- `rrf_no_rerank` as the current top judge-scoring system
- `caes_rag_rrf` as the full proposed method
- `default` as the same-retrieval stability control

The following methods are still useful as exploratory variants, but they do not currently look like the strongest final path:

- `semantic_current`
- `topic_shift_current`
- `semantic_minilm`
- `topic_shift_minilm`

## Why the Full Method Does Not Clearly Beat `rrf_no_rerank`

A reasonable interpretation is that the current reranking signals are not yet discriminative enough at the semantic level. In some cases, features such as generic lexical overlap, title overlap, temporal cues, or citation priors may promote borderline-relevant documents rather than the truly best evidence.

In other words, the current reranking stage is not yet consistently better than relying on the fused retrieval ranking directly.

## Priority Next Step

The highest-priority next step is to improve the semantic-model-based reranking setup, especially its scoring parameters, weighting scheme, and selection behavior. The current results do not show a consistent final-answer gain from the semantic variants, but this is more plausibly a tuning problem than evidence that semantic reranking is intrinsically ineffective.

Two lower-priority but still useful next steps follow naturally from the fixed-10-document setting used in Task 4.

First, we should add more explicit fixed-10 document diagnostics. Since the system is always given the same 10 candidate documents, a useful analysis is to separate cases where the right document is present but ranked too low, from cases where the document is selected but not actually used well in the final answer. This would help us localize whether the failure happens at document scoring, evidence selection, or answer construction.

Second, we should improve answer coverage and evidence allocation within the fixed candidate pool. Because the system cannot retrieve outside the 10 provided documents, part of the remaining performance gap is likely due to how evidence is distributed across the final answer rather than to missing recall. In particular, the system should better decide when to synthesize across multiple partially relevant documents, when to cover multiple aspects of the query, and when to avoid wasting answer space on marginal evidence.

# On-Going Result Analysis

## Rouge Limitation Under Same-Retrieval Comparisons

One useful current observation is that `Rouge` appears to be limited as a standalone indicator of RAG quality in this project.

In our experiment configs, `default` and `caes_rag_rrf` use the same retrieval pipeline: the same rule-based chunking, the same multi-query expansion, the same PRF, the same RRF fusion, and the same lightweight evidence reranking. In other words, they are effectively a same-retrieval stability comparison rather than two genuinely different retrieval systems.

Because of that, any score differences between these two runs in lexical overlap metrics such as `Rouge` are likely driven mainly by differences in the final generated wording rather than by differences in retrieval quality itself. The selected evidence may be very similar while the surface form of the final answer changes enough to move the Rouge score.

This matters for interpretation:

- `Rouge` is still useful as an end-to-end text similarity signal.
- But `Rouge` should not be treated as a clean measure of retrieval or evidence-selection quality.
- In same-retrieval comparisons, `Rouge` variation can overstate differences that are really just generation variation.

So, for analyzing actual RAG capability, especially grounding and evidence quality, metrics such as citation precision, nugget coverage, and average nugget grade are more informative than Rouge alone.

## Working Interpretation

A practical interpretation for the current paper is:

> When two runs share the same retrieval stack, differences in Rouge mainly reflect answer phrasing differences introduced during generation. This suggests that Rouge is better viewed as a surface-form end-to-end metric than as a direct measure of core RAG ability.

This also supports a broader methodological point: RAG evaluation should not rely on lexical-overlap metrics alone, because they can underrepresent grounded but differently phrased answers and overemphasize stylistic variation in final generation.

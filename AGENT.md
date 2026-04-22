# AGENT.md

## Project Context

This repository is for **CLEF LongEval 2026 Task 4: LongEval-RAG**.

Task 4 focuses on retrieval-augmented generation over evolving scientific knowledge. For each query, the system receives a query/narrative and a fixed set of candidate document IDs. The system must generate an answer **only using the provided candidate documents**. The final output must follow the **TREC RAG JSONL format**, where each answer sentence includes citations as integer indices into a `references` list.

The main research direction is:

> Citation-aware evidence selection and atomic response generation for longitudinal scientific RAG.

Our primary method is called:

> **CAES-RAG: Citation-Aware Evidence Selection for LongEval-RAG**

The system should prioritize:
- evidence grounding;
- citation correctness;
- robust JSONL output;
- reproducibility;
- modularity;
- optional dense/scientific embeddings;
- temporal awareness where metadata is available.

---

## Core Task Requirements

For each query, produce one JSONL line:

```json
{
  "metadata": {
    "team_id": "our_team",
    "run_id": "hybrid_evidence_rag_v1",
    "type": "automatic",
    "narrative": "<query text>",
    "narrative_id": "<query id>"
  },
  "references": ["docid_1", "docid_2"],
  "answer": [
    {
      "text": "A short factual answer sentence.",
      "citations": [0]
    },
    {
      "text": "Another short factual sentence.",
      "citations": [0, 1]
    }
  ]
}
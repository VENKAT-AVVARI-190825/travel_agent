# Policy RAG (Retrieval-Augmented Generation)

Grounds the trip-planning agents in **corporate travel policy** so plans respect
cabin-class rules, booking lead times, hotel/meal caps, preferred suppliers,
approvals, and payment rules — instead of inventing recommendations.

## Why RAG here
The trip data (flights, hotels, weather) is live and comes from tools. But
*policy* is stable, private, enterprise knowledge — a textbook RAG use case:
embed it once, retrieve the relevant passages at plan time, and inject them into
the planner prompt as hard constraints.

## Pipeline
```
rag/policies/*.md  ──chunk──▶  text-embedding-3-small  ──▶  ChromaDB (on disk)
                                                                │
plan request ──▶ embed query ──▶ top-k cosine search ──────────┘
                                     │
                                     ▼
                    policy context injected into the planner prompt
                    (+ exposed as the `search_travel_policy` tool)
```

## Design choices (the parts interviewers probe)
- **Vector DB:** ChromaDB, persistent/on-disk. Swappable for pgvector/Pinecone
  at scale; the `PolicyStore` interface stays the same.
- **Embeddings:** `text-embedding-3-small` — strong cost/latency/quality balance
  for short policy chunks.
- **Chunking:** heading-aware, ~800 chars with 150-char overlap, so each chunk
  is self-contained but boundary context isn't lost.
- **Retrieval:** top-k cosine with a distance threshold (`max_distance`) so
  irrelevant chunks are dropped rather than padding the prompt.
- **Fail-soft:** if the index isn't built or a key is missing, the retriever
  returns a structured error and planning continues un-grounded.

## Build the index
```bash
python rag/ingest_policies.py
```
Re-run whenever files in `rag/policies/` change (ingest resets and re-indexes).

## Use it
- **In the agent:** Phase 4's planner retrieves policy eagerly and can re-query
  via the `search_travel_policy` tool ([../phases/phase4_langgraph/trip_agents.py](../phases/phase4_langgraph/trip_agents.py)).
- **Standalone:** `python -m rag.policy_store` runs an ingest + retrieval demo.
- **Over MCP:** exposed as `search_travel_policy` by the [MCP server](../mcp_server/README.md).

## Production notes
- Move Chroma → pgvector/managed store; separate the index lifecycle (re-embed
  on document updates) from the request path.
- Add hybrid search (BM25 + dense) and a re-ranker for higher precision.
- Evaluate retrieval with DeepEval (context precision/recall, faithfulness).

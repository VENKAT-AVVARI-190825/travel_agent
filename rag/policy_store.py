"""
Policy RAG Store
----------------
A thin, self-contained Retrieval-Augmented-Generation layer over the corporate
travel-policy corpus in ``rag/policies``.

Design notes (interview-relevant):
- **Vector DB:** ChromaDB (persistent, on-disk) — already a project dependency.
  In production this would move to a managed/pgvector-style store, but the
  ingest/retrieve interface here stays the same.
- **Embeddings:** OpenAI ``text-embedding-3-small`` (see ``config.MODELS``).
  Chosen for a good cost/latency/quality trade-off on short policy chunks.
- **Chunking:** heading-aware, ~800 chars with 150 char overlap so a retrieved
  chunk keeps enough surrounding context to be self-explanatory.
- **Retrieval:** top-k cosine similarity with a distance threshold so obviously
  irrelevant chunks are dropped instead of padding the prompt.
"""
from __future__ import annotations

import os
import glob
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

# Resolve config with a direct-import fallback (matches the toolkit pattern).
try:
    from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODELS
    EMBED_MODEL = MODELS.get("embedding", "text-embedding-3-small")
except ImportError:  # pragma: no cover - fallback for direct execution
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    EMBED_MODEL = "text-embedding-3-small"

# On-disk locations (kept inside the rag/ package so ingest is reproducible).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLICY_DIR = os.path.join(_THIS_DIR, "policies")
DEFAULT_PERSIST_DIR = os.path.join(_THIS_DIR, "chroma_db")
COLLECTION_NAME = "travel_policies"


def _chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 150) -> List[Dict[str, Any]]:
    """Heading-aware chunker.

    Splits on markdown headings first so each section stays coherent, then
    packs paragraphs up to ``chunk_size`` with a sliding ``overlap`` so context
    isn't lost at chunk boundaries.
    """
    # Split into sections at markdown headings, keeping the heading with its body.
    sections: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    chunks: List[Dict[str, Any]] = []
    idx = 0
    for section in sections:
        if not section:
            continue
        # If a section is small enough, keep it whole.
        if len(section) <= chunk_size:
            chunks.append({"text": section, "source": source, "chunk": idx})
            idx += 1
            continue
        # Otherwise slide a window with overlap across the section.
        start = 0
        while start < len(section):
            piece = section[start : start + chunk_size]
            chunks.append({"text": piece.strip(), "source": source, "chunk": idx})
            idx += 1
            start += chunk_size - overlap
    return chunks


class PolicyStore:
    """Persistent Chroma-backed store for the travel-policy corpus."""

    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR):
        self.persist_dir = persist_dir
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            api_base=OPENAI_BASE_URL,
            model_name=EMBED_MODEL,
        )

    def _collection(self):
        return self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def is_ready(self) -> bool:
        """True if the collection exists and has at least one document."""
        try:
            return self._collection().count() > 0
        except Exception:
            return False

    def ingest(self, policy_dir: str = DEFAULT_POLICY_DIR, reset: bool = True) -> int:
        """Chunk, embed, and index every ``.md`` file in ``policy_dir``.

        Returns the number of chunks indexed. Idempotent when ``reset=True``.
        """
        if reset:
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass  # first run: nothing to delete

        collection = self._collection()

        all_chunks: List[Dict[str, Any]] = []
        for path in sorted(glob.glob(os.path.join(policy_dir, "*.md"))):
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            all_chunks.extend(_chunk_text(text, source=os.path.basename(path)))

        if not all_chunks:
            return 0

        collection.add(
            ids=[f"{c['source']}::{c['chunk']}" for c in all_chunks],
            documents=[c["text"] for c in all_chunks],
            metadatas=[{"source": c["source"], "chunk": c["chunk"]} for c in all_chunks],
        )
        return len(all_chunks)

    def retrieve(self, query: str, top_k: int = 4, max_distance: float = 0.75) -> List[Dict[str, Any]]:
        """Return the top-k most relevant policy chunks for ``query``.

        Chunks whose cosine distance exceeds ``max_distance`` are dropped so we
        don't pad the prompt with irrelevant policy text.
        """
        collection = self._collection()
        if collection.count() == 0:
            return []

        res = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        hits: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            if dist is not None and dist > max_distance:
                continue
            hits.append(
                {
                    "content": doc,
                    "source": (meta or {}).get("source", "unknown"),
                    "distance": round(float(dist), 4) if dist is not None else None,
                }
            )
        return hits

    @staticmethod
    def format_context(hits: List[Dict[str, Any]]) -> str:
        """Render retrieved chunks into a prompt-ready, citation-tagged block."""
        if not hits:
            return "No applicable travel-policy sections were found."
        lines = []
        for i, hit in enumerate(hits, 1):
            lines.append(f"[Policy {i} — source: {hit['source']}]\n{hit['content']}")
        return "\n\n".join(lines)


if __name__ == "__main__":
    store = PolicyStore()
    n = store.ingest()
    print(f"Indexed {n} policy chunks into {store.persist_dir}")
    demo = store.retrieve("What cabin class can I book for a 3 hour flight?")
    print("\n--- Retrieval demo ---")
    print(PolicyStore.format_context(demo))

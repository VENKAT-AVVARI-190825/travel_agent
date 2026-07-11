"""
Ingest the travel-policy corpus into the Chroma vector store.

Usage:
    python rag/ingest_policies.py

Re-run any time the markdown files in ``rag/policies`` change. Ingest is
idempotent (it resets the collection before re-indexing).
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.policy_store import PolicyStore, DEFAULT_POLICY_DIR, DEFAULT_PERSIST_DIR


def main() -> None:
    store = PolicyStore()
    count = store.ingest(policy_dir=DEFAULT_POLICY_DIR, reset=True)
    print(f"✓ Indexed {count} policy chunks")
    print(f"  Corpus:  {DEFAULT_POLICY_DIR}")
    print(f"  Vectors: {DEFAULT_PERSIST_DIR}")

    # Smoke-test a couple of representative queries.
    for q in [
        "cabin class for a short domestic flight",
        "hotel nightly rate cap",
        "how far in advance must international flights be booked",
    ]:
        hits = store.retrieve(q, top_k=2)
        top = hits[0]["source"] if hits else "no match"
        print(f"  '{q}' -> {top}")


if __name__ == "__main__":
    main()

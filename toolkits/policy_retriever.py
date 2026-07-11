"""
Policy Retriever Toolkit
------------------------
Agent-facing wrapper around the RAG :class:`PolicyStore`. Follows the same
convention as the other toolkits: methods return a plain ``dict`` and never
raise, so an agent node can call this without a try/except and degrade
gracefully when the index hasn't been built yet.
"""
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class PolicyRetriever:
    """Retrieve relevant corporate-travel-policy passages for a query."""

    def __init__(self):
        self._store = None
        self._init_error = None
        try:
            from rag.policy_store import PolicyStore
            self._store = PolicyStore()
        except Exception as exc:  # missing deps / key — degrade, don't crash
            self._init_error = str(exc)

    def search(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Return policy context for ``query``.

        Shape:
            {"policy_context": str, "results": [...], "grounded": bool}
        or  {"error": str} on failure.
        """
        if self._store is None:
            return {"error": f"Policy store unavailable: {self._init_error}"}
        if not query or not query.strip():
            return {"error": "Query cannot be empty"}

        try:
            if not self._store.is_ready():
                return {
                    "error": "Policy index is empty. Run: python rag/ingest_policies.py",
                    "grounded": False,
                }
            hits = self._store.retrieve(query.strip(), top_k=top_k)
            from rag.policy_store import PolicyStore
            return {
                "policy_context": PolicyStore.format_context(hits),
                "results": hits,
                "grounded": bool(hits),
            }
        except Exception as exc:
            return {"error": f"Policy retrieval failed: {exc}"}


if __name__ == "__main__":
    retriever = PolicyRetriever()
    out = retriever.search("What is the hotel rate cap for domestic travel?")
    print(out.get("policy_context", out))

"""
Web search service using the Tavily API.

Error handling mirrors the weather tool (see interview Q8): retryable faults
(timeouts, connection drops, 429/5xx) are retried with exponential backoff;
terminal faults (bad key, bad input, other 4xx) fail fast. Every failure is
logged and returned as a structured, fail-soft
``{"error", "error_type", "retryable"}`` dict so the calling agent can decide
whether to retry or fall back.
"""
import os
import time
import logging
import requests
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class WebSearchService:
    """Web search service using Tavily API"""

    BASE_URL = "https://api.tavily.com/search"

    RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    MAX_RETRIES = 3
    BASE_BACKOFF = 0.5  # seconds; doubled each attempt

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

        if not self.api_key:
            raise ValueError("Tavily API key is required. Set TAVILY_API_KEY in environment variables or .env file")

    @staticmethod
    def _err(message: str, error_type: str, retryable: bool) -> dict:
        """Build a structured, fail-soft error the agent can branch on."""
        return {"error": message, "error_type": error_type, "retryable": retryable}

    def _backoff(self, attempt: int, reason: str) -> None:
        wait = self.BASE_BACKOFF * (2 ** (attempt - 1))
        logger.warning("Tavily search attempt %d/%d failed (%s); retrying in %.1fs",
                       attempt, self.MAX_RETRIES, reason, wait)
        time.sleep(wait)

    def search(self, query: str, max_results: int = 5):
        """
        Search the web for information.

        Returns:
            {"query": str, "results": [{"title", "url", "content"}]} on success,
            or {"error": str, "error_type": str, "retryable": bool} on failure.
        """
        if not query or not query.strip():
            return self._err("Search query cannot be empty", "invalid_input", retryable=False)
        if max_results < 1 or max_results > 20:
            return self._err("max_results must be between 1 and 20", "invalid_input", retryable=False)

        payload = {
            "api_key": self.api_key,
            "query": query.strip(),
            "num_results": max_results,
        }

        last_reason = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = requests.post(self.BASE_URL, json=payload, timeout=15)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                # Transient network fault -> back off and retry.
                last_reason = str(e)
                self._backoff(attempt, f"network error: {e}")
                continue

            # Terminal auth error -> retrying won't help.
            if response.status_code == 401:
                logger.error("Tavily search unauthorized (invalid API key)")
                return self._err("Invalid API key - check your Tavily API credentials",
                                 "auth_error", retryable=False)

            if response.status_code in self.RETRYABLE_STATUS:
                last_reason = f"HTTP {response.status_code}"
                self._backoff(attempt, f"retryable status {response.status_code}")
                continue

            if not response.ok:
                # Other terminal HTTP error (4xx) -> do not retry.
                logger.error("Tavily search terminal status %s", response.status_code)
                return self._err(f"Search API error: {response.status_code}",
                                 "client_error", retryable=False)

            # Success: parse results (shape bugs are terminal, not retryable).
            try:
                data = response.json()
            except ValueError as e:
                logger.error("Tavily search returned invalid JSON: %s", e)
                return self._err("Search service returned an invalid response",
                                 "bad_response", retryable=False)

            results = [
                {
                    "title": r.get("title", "No title"),
                    "url": r.get("url", ""),
                    "content": r.get("content", "No content available"),
                }
                for r in data.get("results", [])
            ]
            return {"query": query, "results": results}

        # Exhausted retries on a transient fault.
        logger.error("Tavily search failed after %d retries (%s)", self.MAX_RETRIES, last_reason)
        return self._err(f"Search service unavailable after {self.MAX_RETRIES} retries",
                         "unavailable", retryable=True)

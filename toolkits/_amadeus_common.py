"""
Shared retry/backoff helper for the Amadeus toolkits.

The Amadeus SDK raises ``ResponseError`` (and typed subclasses) rather than the
``requests`` exceptions the weather/web tools use, so the retryable-vs-terminal
classification lives here and is reused by the flight, hotel, and experience
toolkits (see interview Q8):

  - **Retryable:** transport/network errors, server errors (5xx), rate limits
    (429) -> retried with exponential backoff.
  - **Terminal:** client errors (4xx), auth (401), not-found (404) -> raised
    immediately so the caller can return its empty contract.
"""
import time
import logging

from amadeus import ResponseError

# Typed subclasses are nicer to branch on, but their import path varies by SDK
# version; fall back to status-code inspection when they aren't importable.
try:
    from amadeus.client.errors import ServerError, NetworkError
except Exception:  # pragma: no cover - version-dependent
    ServerError = None
    NetworkError = None

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_BACKOFF = 0.5  # seconds; doubled each attempt
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(err: "ResponseError") -> bool:
    """True if the Amadeus error could plausibly succeed on a retry."""
    if NetworkError is not None and isinstance(err, NetworkError):
        return True
    if ServerError is not None and isinstance(err, ServerError):
        return True
    response = getattr(err, "response", None)
    # A missing response usually means a transport/network failure -> retry.
    if response is None:
        return True
    return getattr(response, "status_code", None) in RETRYABLE_STATUS


def call_with_retry(label: str, fn, *args, **kwargs):
    """Call an Amadeus SDK method with retry + backoff on transient failures.

    Returns the SDK response on success. Re-raises the ``ResponseError`` when it
    is terminal or when retries are exhausted, so the caller's existing
    ``except ResponseError`` block can return its empty contract unchanged.
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except ResponseError as e:
            last_err = e
            if _is_retryable(e) and attempt < MAX_RETRIES:
                wait = BASE_BACKOFF * (2 ** (attempt - 1))
                logger.warning("Amadeus %s attempt %d/%d failed (%s); retrying in %.1fs",
                               label, attempt, MAX_RETRIES, e, wait)
                time.sleep(wait)
                continue
            body = getattr(getattr(e, "response", None), "body", e)
            logger.error("Amadeus %s failed (terminal or retries exhausted): %s", label, body)
            raise
    # Defensive: loop always returns or raises, but keep mypy/readers happy.
    raise last_err

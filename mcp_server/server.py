"""
TravelMate MCP Server
=====================
Exposes the shared toolkits (flights, hotels, experiences, weather, web search,
datetime) and the policy-RAG retriever over the **Model Context Protocol** so
that *any* MCP-capable client — Claude Desktop, an IDE, or another team's
agent — can consume them over a standard transport instead of importing our
Python objects directly.

Why this matters (interview-relevant):
- **Decoupling:** tools become a versioned, independently-owned service. The
  agent team consumes; a platform/payments team could own and govern the tools.
- **Reuse & governance:** one MCP endpoint, many clients; auth, rate limiting,
  and audit can live at the protocol boundary.
- **Interoperability:** the same tools work for our LangGraph agent and for a
  human using Claude Desktop, with no per-client glue code.

Run (stdio transport, the default for local MCP clients):
    python mcp_server/server.py

Toolkits that need API keys (Amadeus, Tavily) are initialised lazily and fail
soft: a missing key returns a structured error instead of crashing the server.
"""
from __future__ import annotations

import os
import sys
import json
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("travelmate-tools")

# --- Lazy singletons -------------------------------------------------------
# Toolkit constructors raise when credentials are absent, so we build them on
# first use and cache either the instance or the initialisation error.
_cache: dict = {}


def _get(name: str):
    """Lazily build and cache a toolkit; return (instance, error_string)."""
    if name in _cache:
        entry = _cache[name]
        return entry.get("obj"), entry.get("err")
    obj, err = None, None
    try:
        if name == "flight":
            from toolkits.amadeus_flight_tool import AmadeusFlightToolkit
            obj = AmadeusFlightToolkit()
        elif name == "hotel":
            from toolkits.amadeus_hotel_search import AmadeusHotelToolkit
            obj = AmadeusHotelToolkit()
        elif name == "experience":
            from toolkits.amadeus_experience_tool import AmadeusExperienceToolkit
            obj = AmadeusExperienceToolkit()
        elif name == "weather":
            from toolkits.weather_tool import WeatherTool
            obj = WeatherTool()
        elif name == "web":
            from toolkits.web_search_service import WebSearchService
            obj = WebSearchService()
        elif name == "datetime":
            from toolkits.current_datetime import DateTimeTool
            obj = DateTimeTool()
        elif name == "policy":
            from toolkits.policy_retriever import PolicyRetriever
            obj = PolicyRetriever()
    except Exception as exc:
        err = str(exc)
    _cache[name] = {"obj": obj, "err": err}
    return obj, err


def _dump(value) -> str:
    """Serialise a tool result to a compact JSON string for the transport."""
    try:
        return json.dumps(value, indent=2, default=str)
    except Exception:
        return str(value)


# --- Tools -----------------------------------------------------------------

@mcp.tool()
def search_flights(origin: str, destination: str, departure_date: str,
                   return_date: Optional[str] = None, adults: int = 1) -> str:
    """Search flights between two cities (dates as YYYY-MM-DD) via Amadeus."""
    toolkit, err = _get("flight")
    if err:
        return _dump({"error": f"Flight toolkit unavailable: {err}"})
    offers = toolkit.flight_search(origin, destination, departure_date,
                                   return_date=return_date, adults=adults)
    return _dump({"flights": offers, "count": len(offers)})


@mcp.tool()
def search_hotels(city: str, check_in: str, check_out: str, adults: int = 1) -> str:
    """Search hotels and offers in a city (dates as YYYY-MM-DD) via Amadeus."""
    toolkit, err = _get("hotel")
    if err:
        return _dump({"error": f"Hotel toolkit unavailable: {err}"})
    hotel_ids, hotels = toolkit.hotel_list(city)
    if not hotel_ids:
        return _dump({"error": f"No hotels found for '{city}'", "hotels": []})
    # Cap the fan-out to keep the Amadeus offer call within limits.
    hotel_ids, hotels = hotel_ids[:10], hotels[:10]
    offers = toolkit.hotel_search(hotel_ids, hotels, check_in, check_out, adults)
    return _dump({"hotels": hotels, "offers": offers})


@mcp.tool()
def search_experiences(city: str, radius_km: int = 20, max_results: int = 10) -> str:
    """Search activities/experiences in a city via Amadeus."""
    toolkit, err = _get("experience")
    if err:
        return _dump({"error": f"Experience toolkit unavailable: {err}"})
    activities = toolkit.experience_search(city, radius_km=radius_km, max_results=max_results)
    return _dump({"experiences": activities, "count": len(activities)})


@mcp.tool()
def get_weather(city: str, start_date: str, end_date: str) -> str:
    """Get the weather forecast for a city over a date range (YYYY-MM-DD)."""
    toolkit, err = _get("weather")
    if err:
        return _dump({"error": f"Weather toolkit unavailable: {err}"})
    return _dump(toolkit.get_weather_range(city, start_date, end_date))


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via Tavily for general travel information."""
    toolkit, err = _get("web")
    if err:
        return _dump({"error": f"Web search unavailable: {err}"})
    return _dump(toolkit.search(query, max_results=max_results))


@mcp.tool()
def current_datetime(timezone: Optional[str] = None) -> str:
    """Get the current date/time, optionally in a named timezone (e.g. Asia/Kolkata)."""
    toolkit, err = _get("datetime")
    if err:
        return _dump({"error": f"Datetime toolkit unavailable: {err}"})
    return _dump(toolkit.get_current_datetime(timezone=timezone))


@mcp.tool()
def search_travel_policy(query: str, top_k: int = 4) -> str:
    """Retrieve corporate travel-policy passages (RAG) relevant to the query.

    Covers cabin class, booking lead time, hotel/meal caps, preferred suppliers,
    approvals, and payment rules. Requires the policy index to be built first
    (python rag/ingest_policies.py).
    """
    toolkit, err = _get("policy")
    if err:
        return _dump({"error": f"Policy retriever unavailable: {err}"})
    return _dump(toolkit.search(query, top_k=top_k))


if __name__ == "__main__":
    # Default transport is stdio, which is what local MCP clients expect.
    mcp.run()

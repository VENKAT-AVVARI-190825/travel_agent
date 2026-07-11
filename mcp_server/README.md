# TravelMate MCP Server

Exposes the shared toolkits and the policy-RAG retriever over the **Model
Context Protocol (MCP)**, so any MCP-capable client (Claude Desktop, an IDE,
another team's agent) can call them over a standard transport — no importing our
Python objects.

## Why MCP (vs. in-process LangChain tools)
The tools began as in-process `@tool` functions bound to the agent. Behind MCP
they become a **versioned, independently-owned service**:
- **Decoupling** — tool ownership separates from agent ownership.
- **Governance** — auth, rate limiting, and audit live at the protocol boundary.
- **Reuse** — one endpoint serves the LangGraph agent *and* a human in Claude
  Desktop, with no per-client glue.

## Tools exposed
| MCP tool | Backed by |
|---|---|
| `search_flights` | `AmadeusFlightToolkit` |
| `search_hotels` | `AmadeusHotelToolkit` |
| `search_experiences` | `AmadeusExperienceToolkit` |
| `get_weather` | `WeatherTool` (Open-Meteo) |
| `web_search` | `WebSearchService` (Tavily) |
| `current_datetime` | `DateTimeTool` |
| `search_travel_policy` | `PolicyRetriever` (RAG / ChromaDB) |

Toolkits needing API keys are initialised lazily and **fail soft** — a missing
key returns a structured error instead of crashing the server.

## Install
```bash
pip install mcp        # already added to requirements.txt
```

## Run (stdio transport)
```bash
python mcp_server/server.py
```

## Connect from Claude Desktop
Add to `claude_desktop_config.json` (adjust the absolute path):
```json
{
  "mcpServers": {
    "travelmate": {
      "command": "python",
      "args": ["/absolute/path/to/travel_agent/mcp_server/server.py"]
    }
  }
}
```
Restart Claude Desktop; the seven tools appear and can be invoked in chat.

## Inspect / debug
```bash
# The MCP Inspector gives a UI to list and call tools:
npx @modelcontextprotocol/inspector python mcp_server/server.py
```

## Production notes
- Put the server behind an authenticated transport (HTTP/SSE) with per-client
  rate limits for remote use.
- Add request/response logging with correlation IDs for audit.
- Never let traveller PII flow into tool args/logs unredacted (see policy).

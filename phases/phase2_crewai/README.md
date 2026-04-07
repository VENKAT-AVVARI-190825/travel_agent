# Phase 2: CrewAI Sequential Agent Implementation

## Overview
Phase 2 implements a sequential agent workflow using CrewAI framework for intelligent trip planning. Three specialized agents work in sequence to collect information, plan itineraries, and optimize travel plans.

## Architecture

### Sequential Workflow
```
User Input → InfoCollector → Planner → Optimizer → Final Plan
```

### Agents

#### 1. InfoCollector Agent
- **Role**: Travel Requirements Specialist
- **Goal**: Extract and validate complete travel requirements from user requests
- **Tools**: Web Search, Current DateTime
- **Output**: Structured JSON with travel requirements

#### 2. Planner Agent  
- **Role**: Travel Itinerary Specialist
- **Goal**: Create comprehensive travel itineraries with flights, hotels, and activities
- **Tools**: Flight Search, Hotel Search, Experience Search, Weather, Web Search, DateTime
- **Output**: Complete day-by-day travel itinerary

#### 3. Optimizer Agent
- **Role**: Travel Cost Optimizer
- **Goal**: Optimize travel plans for cost, timing, and customer satisfaction
- **Tools**: Web Search (primary for price comparison)
- **Output**: Optimized travel plan with cost breakdown

## Key Features

### ✅ Implemented
- Sequential agent execution (InfoCollector → Planner → Optimizer)
- Database persistence for all agent outputs and workflow states
- Tool integration with real API calls (Amadeus, Weather, Web Search)
- Robust error handling with fallback mechanisms
- Pydantic data model validation
- User feedback loops and approval workflow
- FastAPI endpoints integration
- Streamlit UI integration

### 🔧 Tool Integration
- **Amadeus APIs**: Flights, Hotels, Experiences
- **Weather API**: Open-Meteo for forecasts
- **Web Search**: Tavily API as fallback for all tools
- **Fallback Strategy**: Web search used when any API fails

### 📊 Database Operations
- Trip creation and status tracking
- Chat history persistence
- Trip plan storage with metadata
- User approval workflow tracking

## Files Structure

```
phases/phase2_crewai/
├── trip_agents.py          # Agent definitions with tools
├── trip_orchestrator.py    # CrewAI orchestration logic
└── README.md              # This file
```

## Usage

### API Endpoints
- `POST /api/v1/plan_trip` - Main trip planning endpoint
- `POST /api/v1/approve` - Trip approval/rejection with feedback

### Example Request
```python
# Complete trip request
user_input = "I want to plan a leisure trip from Bangalore to Goa from December 15-18, 2026, for 2 adults and 2 children with a budget of 2000 INR."

# API call
response = orchestrator.plan_trip(user_input, user_id=1)
```

### Example Response
```json
{
  "success": true,
  "trip_id": 54,
  "message": "Trip planned successfully! Bangalore to Goa for 4 travelers.",
  "requirements": {
    "origin": "Bangalore",
    "destination": "Goa",
    "trip_startdate": "2026-12-15",
    "trip_enddate": "2026-12-18",
    "no_of_adults": 2,
    "no_of_children": 2,
    "budget": 2000.0,
    "currency": "INR"
  },
  "plan": {
    "itinerary": "Day-by-day travel plan...",
    "daily_budget": 500.0,
    "total_estimated_cost": 2000.0
  },
  "optimization": {
    "recommendations": "Cost optimization suggestions...",
    "cost_savings": 0.0,
    "final_plan": "Optimized travel plan..."
  },
  "status": "completed"
}
```

## Testing

### Test Cases Implemented
1. **Complete Query Test**: Full trip details in single request
2. **Multi-turn Conversation**: Incomplete info requiring follow-up
3. **Error Handling**: API failures with fallback mechanisms
4. **Database Persistence**: All outputs saved correctly

### Test Results
- ✅ Trip ID 54 created successfully
- ✅ All agents executed in sequence  
- ✅ Database operations completed
- ✅ Status: "completed"
- ✅ Destination parsing: "Goa" (fixed regex issue)

## Configuration

### LLM Setup
```python
llm = LLM(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model="openai/gpt-4o-mini"
)
```

### Environment Variables Required
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_BASE_URL`: OpenAI base URL
- `AMADEUS_CLIENT_ID`: Amadeus API client ID
- `AMADEUS_CLIENT_SECRET`: Amadeus API client secret
- `TAVILY_API_KEY`: Tavily web search API key

## Status
🚀 **PRODUCTION READY** - All Phase 2 requirements implemented and tested successfully.

## Next Steps
Ready to proceed to Phase 3 (AutoGen) or Phase 4 (LangGraph) implementations.
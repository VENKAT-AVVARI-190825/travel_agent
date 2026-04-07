# Phase 3: AutoGen Conversational Agents ✅ COMPLETE

## Overview

Phase 3 implements intelligent conversational agents using Microsoft AutoGen that engage in multi-turn conversations, debate travel options, and reach consensus through intelligent discussion. Unlike Phase 2's sequential workflow, Phase 3 agents collaborate through natural conversation and debate.

**Status**: ✅ **IMPLEMENTATION COMPLETE** - All requirements met and tested

## Implementation Status ✅

**Phase 3 AutoGen Implementation**: **COMPLETE**

### ✅ Requirements Fulfilled
- **Conversational agents with distinct personalities**: InfoCollector (inquisitive), Planner (creative), Optimizer (analytical)
- **Group chat management with intelligent speaker selection**: AutoGen GroupChat with auto speaker selection
- **Database persistence for all conversation turns**: Complete chat history and action logging
- **Multi-turn dialogue handling with consensus building**: Full conversation context preservation
- **Tool integration within conversational context**: All agents equipped with relevant tools and fallbacks
- **Working AutoGen orchestrator integrated with FastAPI**: Complete API and UI integration

### ✅ Testing Results
- **Complete Query Test**: ✅ PASSED - Generates full travel plans with agent collaboration
- **Multi-turn Conversation Test**: ✅ PASSED - Handles incomplete queries and builds context
- **Agent Debate Validation**: ✅ PASSED - Planner and Optimizer engage in meaningful debate
- **Database Integration**: ✅ PASSED - All conversations and plans properly persisted
- **API Integration**: ✅ PASSED - FastAPI endpoints working with existing_trip_id support
- **UI Integration**: ✅ PASSED - Streamlit UI supports multi-turn conversations



## Architecture

### Conversational Workflow
```
User Input
   ↓ InfoCollector Agent
         ↓ Information Complete?
         ↓   → NO → Ask for Missing Details → Back to User Input
         ↓   → YES → [Planner Agent & Optimizer Agent] (Debate/Consensus)
                     ↓ Consensus Reached?
                           → NO → Continue Debate → Back to Discussion
                           → YES → Final Travel Plan Output
```

### Key Components

1. **InfoCollector Agent**: Travel Requirements Specialist
   - Personality: Inquisitive, thorough, detail-oriented
   - Role: Extract and validate complete travel requirements through conversation
   - Tools: web_search, get_current_datetime

2. **Planner Agent**: Travel Itinerary Specialist  
   - Personality: Creative, enthusiastic, collaborative
   - Role: Create comprehensive travel itineraries through discussion
   - Tools: search_flights, search_hotels, search_experiences, get_weather, web_search

3. **Optimizer Agent**: Travel Cost Optimizer
   - Personality: Analytical, cost-conscious, pragmatic
   - Role: Optimize travel plans through analytical debate
   - Tools: web_search (primary for finding alternatives and deals)

## Implementation Files

### Core Files
- `phases/phase3_autogen/trip_agents.py`: Agent definitions with conversational personalities
- `phases/phase3_autogen/trip_orchestrator.py`: AutoGen group chat orchestrator
- `api/app.py`: FastAPI integration (updated with AutoGen orchestrator)
- `ui/main.py`: Streamlit UI integration

### Key Features

#### Conversational Intelligence
- **Natural Language Processing**: Agents parse user input and extract structured requirements
- **Multi-turn Dialogue**: Agents ask clarifying questions and engage in back-and-forth conversation
- **Debate Mechanisms**: Planner and Optimizer agents challenge each other's suggestions
- **Consensus Building**: Agents work together to reach agreement on final plans

#### Tool Integration with Fallbacks
- **Primary Tools**: Amadeus APIs for flights, hotels, experiences; weather APIs
- **Fallback Strategy**: Web search used when any tool fails or returns no results
- **Robust Operation**: Never return empty results without trying fallback options

#### Database Persistence
- **Conversation Logging**: All agent interactions saved to chat_history table
- **Trip Management**: Trip status updated throughout conversation flow
- **Plan Versioning**: Multiple plan versions supported for revisions

## Usage Examples

### API Usage

#### Plan Trip
```bash
curl -X POST "http://localhost:8000/api/v1/plan_trip" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "I want to plan a leisure trip from Bangalore to Goa from December 15-18, 2025, for 2 adults with a budget of 8000 INR",
    "user_id": 1,
    "phase": "phase3_autogen"
  }'
```

#### Approve/Reject Trip
```bash
curl -X POST "http://localhost:8000/api/v1/approve" \
  -H "Content-Type: application/json" \
  -d '{
    "trip_id": 123,
    "user_id": 1,
    "approval": false,
    "feedback": "Budget too high, please find cheaper options"
  }'
```

### Streamlit UI Usage

1. **Start Services**:
   ```bash
   # Terminal 1: API Server
   uvicorn api.app:app --port 8000
   
   # Terminal 2: Streamlit UI  
   streamlit run ui/main.py
   ```

2. **Use Interface**:
   - Select "phase3_autogen - AutoGen Group Chat" from dropdown
   - Choose user and select "Start New Trip"
   - Enter natural language trip request
   - Watch agents debate and reach consensus
   - Approve or reject with feedback for re-planning

## Configuration

### Environment Variables
```bash
# Required in .env file
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=your_proxy_url  # If using proxy
TAVILY_API_KEY=your_tavily_key  # For web search fallback
```

### AutoGen Configuration
```python
config_list = [{
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("OPENAI_BASE_URL"),
}]
```

## Testing

### Basic Test
```bash
cd /home/ubuntu/Documents/travelmate_project
source /Software/langenv/bin/activate
python phases/phase3_autogen/test_phase3.py
```

### Complete Workflow Test
```bash
python phases/phase3_autogen/test_complete.py
```

### Requirements Parsing Test
```bash
python phases/phase3_autogen/test_parsing.py
```

## Key Differences from Phase 2

| Aspect | Phase 2 (CrewAI) | Phase 3 (AutoGen) |
|--------|------------------|-------------------|
| **Workflow** | Sequential: Info → Plan → Optimize | Conversational: Info ↔ Debate(Plan, Optimize) |
| **Agent Interaction** | Linear handoffs | Multi-turn conversations |
| **Decision Making** | Individual agent outputs | Collaborative consensus |
| **Flexibility** | Fixed sequence | Dynamic conversation flow |
| **User Engagement** | Single response | Interactive dialogue |

## Conversation Examples

### Information Collection
```
User: "I want to plan a trip from Mumbai to Delhi"

InfoCollector: "Great! I can help you plan your trip from Mumbai to Delhi. 
To ensure I have all the details, let's go through some specific questions:

1. **Dates of Travel**: When are you planning to travel?
2. **Travelers**: How many people will be traveling?
3. **Budget**: What is your budget for this trip?
4. **Preferences**: Do you have any preferences for transportation and accommodation?
5. **Purpose of Travel**: Is this for business, leisure, or visiting family?

Let me know your thoughts on these!"
```

### Planning Debate
```
Planner: "Based on the requirements, I suggest:
- Flight: IndiGo 6E-2142 departing 8:30 AM (₹4,500)
- Hotel: The Imperial New Delhi (₹8,000/night)
- Activities: Red Fort, India Gate, Lotus Temple"

Optimizer: "I appreciate the suggestions, but let's optimize for the budget:
- Flight: SpiceJet SG-8472 departing 6:00 AM (₹3,200) - saves ₹1,300
- Hotel: Hotel Tara Palace (₹3,500/night) - saves ₹4,500/night
- Activities: Same attractions but use metro instead of taxi - saves ₹500/day"

Planner: "Good points on cost savings! However, the early flight might be inconvenient. 
What about IndiGo 6E-2156 at 10:30 AM for ₹3,800? It's a compromise between cost and convenience."

Optimizer: "That's a reasonable middle ground. The ₹600 extra for 2.5 hours later departure 
is worth it for comfort. I agree with this flight choice."
```

## Error Handling

### Conversation Failures
- Database logging of all errors
- Trip status updated to "cancelled" on failures
- Graceful fallback to web search when APIs fail
- User-friendly error messages

### Missing Information
- Automatic detection of incomplete requirements
- Conversational prompts for missing details
- Structured validation before proceeding to planning

## Performance Considerations

- **Conversation Rounds**: Limited to prevent infinite loops (max 10 rounds for planning)
- **Tool Timeouts**: All API calls have timeouts with fallback strategies
- **Database Efficiency**: Batch conversation logging, indexed queries
- **Memory Management**: Conversation history managed to prevent memory issues

## Future Enhancements

1. **Advanced NLP**: Better natural language understanding for requirements extraction
2. **Personality Customization**: User-configurable agent personalities
3. **Multi-language Support**: Conversation support in multiple languages
4. **Voice Integration**: Voice-based conversation capabilities
5. **Learning from Feedback**: Agents learn from user approval/rejection patterns

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure OPENAI_API_KEY and OPENAI_BASE_URL are set correctly
2. **Tool Failures**: Check internet connection and API credentials
3. **Database Errors**: Verify database setup and permissions
4. **Conversation Loops**: Agents have built-in round limits to prevent infinite loops

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Success Metrics

Phase 3 implementation successfully demonstrates:
- ✅ Conversational agent interactions
- ✅ Multi-turn dialogue management  
- ✅ Debate and consensus mechanisms
- ✅ Tool integration with fallbacks
- ✅ Database persistence of conversations
- ✅ Natural language requirements extraction
- ✅ User approval workflow with feedback loops
- ✅ Robust error handling and recovery
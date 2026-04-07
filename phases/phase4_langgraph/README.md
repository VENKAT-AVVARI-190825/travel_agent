# Phase 4: LangGraph Stateful Workflows

## Overview

Phase 4 implements enterprise-grade AI workflows using LangGraph with stateful execution, checkpoint-based recovery, and human-in-the-loop capabilities. This phase demonstrates production-ready patterns with persistent state management across workflow nodes.

## Key Features

- **Stateful Workflow**: StateGraph with persistent state across all nodes
- **Checkpoint Recovery**: MemorySaver for state persistence and recovery
- **Error Handling**: Dedicated error recovery nodes with graceful fallbacks
- **Human Approval**: Checkpoint-based approval workflow
- **Database Integration**: Complete state transition logging
- **Tool Integration**: LangChain tools with fallback strategies

## Architecture

### Workflow
```
User Input → InfoCollector → Planner → Optimizer → Approval → Completion
                ↓ (error)
           Error Recovery
```

### Components

**TravelState**: Workflow state management
- `messages`: Conversation history
- `requirements`, `travel_plan`, `optimization_results`: Workflow data
- `next_step`, `workflow_complete`: Control flow

**Nodes**:
1. **InfoCollector**: Extract requirements with search_web, get_current_datetime tools
2. **Planner**: Create travel plan with search_flights, search_hotels, search_experiences, get_weather tools
3. **Optimizer**: Cost optimization with web search fallback
4. **Approval**: Prepare human approval with checkpoint persistence
5. **Completion**: Finalize workflow and update database
6. **ErrorRecovery**: Handle failures gracefully

## Files

- `trip_agents.py`: Node implementations with LangChain tools
- `trip_orchestrator.py`: StateGraph orchestrator with MemorySaver
- `../../api/app.py`: FastAPI integration
- `../../ui/main.py`: Streamlit UI

## Usage

### API
```bash
curl -X POST "http://localhost:8000/api/v1/plan_trip" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Trip from Bangalore to Goa, Dec 15-18, 2025, 2 adults, 8000 INR",
    "user_id": 1,
    "phase": "phase4_langgraph"
  }'
```

### UI
```bash
# Terminal 1
uvicorn api.app:app --port 8000

# Terminal 2
streamlit run ui/main.py
```
Select "phase4_langgraph" from dropdown and start planning.

## Configuration

```bash
# .env file
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=your_proxy  # Optional
TAVILY_API_KEY=your_key
```

## Testing

```bash
python phases/phase4_langgraph/trip_orchestrator.py
```

## Comparison with Other Phases

| Feature | Phase 2 (CrewAI) | Phase 3 (AutoGen) | Phase 4 (LangGraph) |
|---------|------------------|-------------------|---------------------|
| Workflow | Sequential | Conversational | Stateful |
| State | None | Context only | Full persistence |
| Recovery | Basic | Retry | Checkpoint-based |
| Approval | Simple | Debate | Stateful workflow |ges=[...],  # Complete conversation history
    user_input="Trip from Mumbai to Delhi",
    user_id=1,
    trip_id=123,
    requirements={...},  # Complete requirements
    travel_plan={...},   # Generated travel plan
    optimization_results={...},  # Cost optimization
    approval_status="pending",
    error_message=None,
    next_step="completion",
    workflow_complete=True
)
```

## Error Handling & Recovery

### State-Based Error Recovery
- **Checkpoint Restoration**: Failed workflows can be resumed from last checkpoint
- **Error Node Routing**: Dedicated error_recovery node handles all failures
- **Graceful Degradation**: Fallback to web search when APIs fail
- **User Feedback Loop**: Missing information handled through state persistence

### Production Patterns
- **Thread Management**: Unique thread IDs for workflow tracking
- **Database Logging**: All state transitions logged for monitoring
- **Exception Handling**: Comprehensive try-catch with meaningful error messages
- **Resource Management**: Proper cleanup and state finalization

## Performance Considerations

- **State Persistence**: Efficient checkpoint storage with MemorySaver
- **Tool Timeouts**: All API calls have timeouts with fallback strategies
- **Memory Management**: State size controlled to prevent memory issues
- **Database Efficiency**: Batch operations and indexed queries

## Future Enhancements

1. **Advanced Checkpointing**: Redis or database-backed checkpoints for production
2. **Workflow Monitoring**: Real-time workflow execution monitoring
3. **Dynamic Routing**: AI-driven conditional edge routing
4. **Parallel Execution**: Concurrent node execution for performance
5. **Custom Recovery**: User-defined error recovery strategies

## Troubleshooting

### Common Issues

1. **State Persistence Errors**: Ensure MemorySaver is properly configured
2. **Tool Binding Issues**: Verify LangChain tool decorators and binding
3. **Workflow Routing**: Check conditional edge functions return valid node names
4. **Database Errors**: Verify database setup and connection handling

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable LangGraph debugging
from langgraph.graph import StateGraph
workflow = StateGraph(TravelState, debug=True)
```

## Success Metrics

Phase 4 implementation successfully demonstrates:
- ✅ Stateful workflow execution with persistent state
- ✅ Enterprise-grade error handling and recovery
- ✅ Human-in-the-loop approval workflows
- ✅ Production-ready monitoring and logging
- ✅ LangChain tool integration with fallbacks
- ✅ Checkpoint-based state persistence
- ✅ Scalable workflow orchestration
- ✅ Complete database integration and validation
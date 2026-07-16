"""
Phase 4: Travel Orchestrator with LangGraph - Stateful Workflow Implementation
"""
import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# Load environment
load_dotenv()

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

from phases.phase4_langgraph.trip_agents import (
    TravelState, collect_travel_info, plan_travel_itinerary, optimize_travel_plan, 
    approval, completion, error_recovery
)
from api.datamodels import Trip, TravelPlan, WorkflowState, CheckpointData
from db import db_utils

class LangGraphTripOrchestrator:
    """
    Orchestrator for Phase 4: LangGraph Stateful Workflow
    - Uses LangGraph to define workflow states and transitions
    - Supports user approval and error recovery
    - Ensures all outputs are validated and persisted
    """
    
    def __init__(self):
        """Initialize StateGraph workflow with enterprise features"""
        self.memory = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.execution_metrics = {}
        self.checkpoints: Dict[str, CheckpointData] = {}

    def _build_workflow(self) -> StateGraph:
        """Build StateGraph with conditional transitions"""
        workflow = StateGraph(TravelState)
        
        # Add nodes with top-level callables
        workflow.add_node("collect_travel_info", collect_travel_info)
        workflow.add_node("plan_travel_itinerary", plan_travel_itinerary)
        workflow.add_node("optimize_travel_plan", optimize_travel_plan)
        workflow.add_node("approval", approval)
        workflow.add_node("completion", completion)
        workflow.add_node("error_recovery", error_recovery)
        
        # Define workflow edges
        workflow.add_edge(START, "collect_travel_info")
        
        # Conditional transitions based on validation
        workflow.add_conditional_edges(
            "collect_travel_info",
            self._route_collect_info,
            {"plan_travel_itinerary": "plan_travel_itinerary", "error_recovery": "error_recovery"}
        )
        
        workflow.add_conditional_edges(
            "plan_travel_itinerary",
            self._route_planner,
            {"optimize_travel_plan": "optimize_travel_plan", "error_recovery": "error_recovery"}
        )
        
        workflow.add_conditional_edges(
            "optimize_travel_plan",
            self._route_optimizer,
            {"approval": "approval", "error_recovery": "error_recovery"}
        )
        
        workflow.add_conditional_edges(
            "approval",
            self._route_approval,
            {"completion": "completion", "error_recovery": "error_recovery"}
        )
        
        workflow.add_edge("completion", END)
        workflow.add_edge("error_recovery", END)
        
        return workflow

    def _route_collect_info(self, state: TravelState) -> str:
        """Route from collect_travel_info based on validation"""
        return state.get("next_step", "error_recovery")

    def _route_planner(self, state: TravelState) -> str:
        """Route from plan_travel_itinerary based on validation"""
        return state.get("next_step", "error_recovery")

    def _route_optimizer(self, state: TravelState) -> str:
        """Route from optimize_travel_plan based on validation"""
        return state.get("next_step", "error_recovery")

    def _route_approval(self, state: TravelState) -> str:
        """Route from approval based on validation"""
        return state.get("next_step", "error_recovery")

    def plan_trip(self, user_input: str, user_id: int, trip_title: str = "My Trip", approval_mode: str = "auto", existing_trip_id: int = None, previous_requirements: Dict = None) -> Dict[str, Any]:
        """Execute StateGraph workflow with enterprise monitoring and multi-turn support"""
        import time
        start_time = time.time()
        thread_id = f"trip_{user_id}_{existing_trip_id or int(time.time())}"
        
        try:
            # Log workflow start
            db_utils.log_action(existing_trip_id, user_id, "workflow_start", {"input": user_input, "thread_id": thread_id}, "phase4_langgraph")
            
            # Create initial state with previous requirements if available
            initial_state = TravelState(
                messages=[],
                user_input=user_input,
                user_id=user_id,
                trip_id=existing_trip_id,
                requirements=previous_requirements,
                travel_plan=None,
                optimization_results=None,
                approval_status=None,
                error_message=None,
                next_step="collect_travel_info",
                workflow_complete=False,
                tool_call_count=0,
                node_visit_count={}
            )
            
            # Execute workflow with checkpoints
            config = RunnableConfig(configurable={"thread_id": thread_id})
            final_state = self.app.invoke(initial_state, config)
            
            # Save checkpoint using Pydantic model
            if final_state.get("trip_id"):
                workflow_state = WorkflowState(
                    trip_id=final_state["trip_id"],
                    user_id=user_id,
                    current_node=final_state.get("next_step", "completion"),
                    requirements=final_state.get("requirements"),
                    travel_plan=final_state.get("travel_plan"),
                    optimization_results=final_state.get("optimization_results"),
                    approval_status=final_state.get("approval_status"),
                    error_message=final_state.get("error_message"),
                    next_step=final_state.get("next_step"),
                    workflow_complete=final_state.get("workflow_complete", False),
                    checkpoint_id=thread_id
                )
                
                checkpoint = CheckpointData(
                    checkpoint_id=thread_id,
                    trip_id=final_state["trip_id"],
                    workflow_state=workflow_state,
                    awaiting_approval=final_state.get("approval_status") == "pending",
                    approval_prompt="Please review and approve the travel plan"
                )
                self.checkpoints[thread_id] = checkpoint
            
            # Track execution metrics
            execution_time = time.time() - start_time
            self.execution_metrics[thread_id] = {
                "execution_time": execution_time,
                "nodes_executed": len([m for m in final_state.get("messages", [])]),
                "success": not final_state.get("error_message"),
                "trip_id": final_state.get("trip_id")
            }
            
            # Log workflow completion
            db_utils.log_action(final_state.get("trip_id"), user_id, "workflow_complete", self.execution_metrics[thread_id], "phase4_langgraph")
            
            # Handle error states
            if final_state.get("error_message"):
                error_msg_lower = final_state.get("error_message", "").lower()
                if "missing" in error_msg_lower or "please provide" in error_msg_lower:
                    return {
                        "success": False,
                        "error": "MISSING_INFO",
                        "message": final_state["error_message"],
                        "trip_id": final_state.get("trip_id"),
                        "thread_id": thread_id,
                        "requirements": final_state.get("requirements"),  # Return partial requirements
                        "state_metadata": {"checkpoint_available": True, "execution_time": execution_time}
                    }
                else:
                    return {
                        "success": False,
                        "error": final_state["error_message"],
                        "message": f"Workflow failed: {final_state['error_message']}",
                        "thread_id": thread_id,
                        "state_metadata": {"checkpoint_available": True, "execution_time": execution_time}
                    }
            
            # Success case with Pydantic model conformance
            trip_id = final_state.get("trip_id")
            plan_id = final_state.get("plan_id")
            
            if approval_mode == "auto" and trip_id:
                db_utils.update_trip_status(trip_id, "confirmed")
                if plan_id:
                    db_utils.update_trip_plan_status(plan_id, "approved")
            elif trip_id:
                db_utils.update_trip_status(trip_id, "pending_approval")
            
            return {
                "success": True,
                "trip_id": trip_id,
                "plan_id": plan_id,
                "message": "Trip planned through LangGraph StateGraph workflow",
                "thread_id": thread_id,
                "requirements": final_state.get("requirements"),
                "plan": final_state.get("travel_plan"),
                "optimization": final_state.get("optimization_results"),
                "workflow_state": {
                    "complete": final_state.get("workflow_complete", False),
                    "approval_status": final_state.get("approval_status")
                },
                "state_metadata": {
                    "execution_time": execution_time,
                    "nodes_executed": len([m for m in final_state.get("messages", [])]),
                    "checkpoint_available": True
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            db_utils.log_action(existing_trip_id, user_id, "workflow_error", {"error": str(e), "execution_time": execution_time}, "phase4_langgraph")
            return {
                "success": False,
                "error": str(e),
                "message": f"StateGraph workflow failed: {str(e)}",
                "thread_id": thread_id,
                "state_metadata": {"execution_time": execution_time, "checkpoint_available": False}
            }

    def continue_trip_approval(self, trip_id: int, approval_decision: str, user_feedback: str = "") -> Dict[str, Any]:
        """Handle human-in-the-loop approval with state persistence"""
        try:
            trip = db_utils.get_trip_by_id(trip_id)
            if not trip:
                return {"success": False, "error": "Trip not found"}
            
            # Log approval decision
            db_utils.log_action(trip_id, trip.user_id, "approval_decision", {"decision": approval_decision, "feedback": user_feedback}, "phase4_langgraph")
            
            if approval_decision == "approved":
                db_utils.update_trip_status(trip_id, "confirmed")
                plan = db_utils.get_trip_plan_by_trip_id(trip_id)
                if plan:
                    db_utils.update_trip_plan_status(plan.id, "approved")
                
                return {
                    "success": True,
                    "message": "Travel plan approved through StateGraph workflow",
                    "trip_id": trip_id,
                    "state_metadata": {"approval_processed": True}
                }
            else:
                db_utils.update_trip_status(trip_id, "rejected")
                return {
                    "success": True,
                    "message": "Travel plan rejected. Checkpoint available for recovery.",
                    "trip_id": trip_id,
                    "state_metadata": {"checkpoint_available": True, "feedback": user_feedback}
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    def resume_from_checkpoint(self, thread_id: str, user_input: str = "") -> Dict[str, Any]:
        """Resume workflow from checkpoint on errors or interruptions"""
        try:
            # Get checkpoint using Pydantic model
            checkpoint = self.checkpoints.get(thread_id)
            if not checkpoint:
                return {"success": False, "error": "No checkpoint found for thread"}
            
            config = RunnableConfig(configurable={"thread_id": thread_id})
            current_state = self.app.get_state(config)
            if not current_state:
                return {"success": False, "error": "No state found for thread"}
            
            # Log recovery using checkpoint data
            db_utils.log_action(
                checkpoint.trip_id, 
                checkpoint.workflow_state.user_id, 
                "checkpoint_recovery", 
                {"thread_id": thread_id, "input": user_input, "checkpoint": checkpoint.checkpoint_id}, 
                "phase4_langgraph"
            )
            
            # Update state with new input if provided
            if user_input:
                updated_state = {**current_state.values, "user_input": user_input}
                final_state = self.app.invoke(updated_state, config)
            else:
                final_state = self.app.invoke(current_state.values, config)
            
            return {
                "success": True,
                "message": "Workflow resumed from checkpoint",
                "thread_id": thread_id,
                "workflow_state": final_state,
                "state_metadata": {"recovered_from_checkpoint": True, "checkpoint_id": checkpoint.checkpoint_id}
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_checkpoint(self, thread_id: str) -> Optional[CheckpointData]:
        """Get checkpoint data using Pydantic model"""
        return self.checkpoints.get(thread_id)
    
    def get_execution_metrics(self, thread_id: str = None) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        if thread_id:
            return self.execution_metrics.get(thread_id, {})
        return self.execution_metrics

    def continue_trip_clarification(self, previous_state: Dict, user_input: str, user_id: int, approval_mode: str = "auto") -> Dict[str, Any]:
        """
        Resume workflow after user provides missing info with multi-turn support
        """
        try:
            thread_id = previous_state.get("thread_id", f"trip_{user_id}_{int(__import__('time').time())}")
            
            # Get previous requirements if any
            previous_requirements = previous_state.get("requirements", {})
            
            # Create state with previous context
            initial_state = TravelState(
                messages=[],
                user_input=user_input,
                user_id=user_id,
                trip_id=previous_state.get("trip_id"),
                requirements=previous_requirements,  # Pass previous requirements
                travel_plan=None,
                optimization_results=None,
                approval_status=None,
                error_message=None,
                next_step="collect_travel_info",
                workflow_complete=False
            )
            
            # Execute workflow with context
            config = RunnableConfig(configurable={"thread_id": thread_id})
            final_state = self.app.invoke(initial_state, config)
            
            # Handle result same as plan_trip
            if final_state.get("error_message"):
                if "missing" in final_state.get("error_message", "").lower():
                    return {
                        "success": False,
                        "error": "MISSING_INFO",
                        "message": final_state["error_message"],
                        "trip_id": final_state.get("trip_id"),
                        "thread_id": thread_id,
                        "requirements": final_state.get("requirements"),
                        "state_metadata": {"checkpoint_available": True}
                    }
                else:
                    return {
                        "success": False,
                        "error": final_state["error_message"],
                        "message": f"Workflow failed: {final_state['error_message']}",
                        "thread_id": thread_id,
                        "state_metadata": {"checkpoint_available": True}
                    }
            
            trip_id = final_state.get("trip_id")
            return {
                "success": True,
                "trip_id": trip_id,
                "message": "Trip planned through multi-turn conversation",
                "thread_id": thread_id,
                "requirements": final_state.get("requirements"),
                "plan": final_state.get("travel_plan"),
                "optimization": final_state.get("optimization_results"),
                "state_metadata": {"multi_turn": True}
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

def test_langgraph_orchestrator():
    """Test the orchestrator with sample input"""
    orchestrator = LangGraphTripOrchestrator()
    
    # Test with sample input
    test_input = "I want to plan a leisure trip from Bangalore to Goa from December 15-18, 2025, for 2 adults with a budget of 8000 INR."
    result = orchestrator.plan_trip(test_input, user_id=1)
    
    print("LangGraph Test Result:", json.dumps(result, indent=2, default=str))
    return result

if __name__ == "__main__":
    test_langgraph_orchestrator()
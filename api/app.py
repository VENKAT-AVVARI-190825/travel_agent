"""
FastAPI Application - LEARNING PROJECT
TODO: Complete the API implementation by connecting your AI agents

Learning Objectives:
- Learn to create FastAPI endpoints
- Understand API request/response patterns
- Integrate with agent orchestrators
- Handle different AI framework phases
"""
import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from typing import Optional, List
from api.datamodels import HotelSuggestion, FlightSuggestion, ApprovalRequest, TripPlanModel, TravelPlan
from db import db_utils

# Import orchestrators
from phases.phase2_crewai.trip_orchestrator import CrewAITripOrchestrator
from phases.phase3_autogen.trip_orchestrator import AutoGenTripOrchestrator
from phases.phase4_langgraph.trip_orchestrator import LangGraphTripOrchestrator

app = FastAPI(title="TravelMate AI API", version="1.0.0")

# Orchestrator mapping
ORCHESTRATOR_MAP = {
    "phase2_crewai": CrewAITripOrchestrator,
    "phase3_autogen": AutoGenTripOrchestrator,
    "phase4_langgraph": LangGraphTripOrchestrator
}

# =============================================================================
# API ENDPOINTS
# =============================================================================

# TODO: Implement main trip planning endpoint
@app.post("/api/v1/plan_trip")
def plan_trip(user_input: str, user_id: int, phase: str = "phase2_crewai", existing_trip_id: int = None, previous_requirements: str = None):
    """
    Plan a trip using the specified AI framework with multi-turn support
    
    Supported phases:
    - phase2_crewai: CrewAI framework with sequential agents
    - phase3_autogen: Microsoft AutoGen with group chat  
    - phase4_langgraph: LangGraph with state management
    """
    
    if phase not in ORCHESTRATOR_MAP:
        return {"success": False, "error": f"Unsupported phase: {phase}"}
    
    try:
        orchestrator = ORCHESTRATOR_MAP[phase]()
        
        # Debug: Print what we received
        print(f"DEBUG API: previous_requirements param = {previous_requirements}")
        
        # Parse previous_requirements if provided
        prev_reqs = None
        if previous_requirements:
            import json
            prev_reqs = json.loads(previous_requirements)
            print(f"DEBUG API: Parsed prev_reqs = {prev_reqs}")
        
        # Handle existing_trip_id and previous_requirements for multi-turn - Phase 4
        if phase == "phase4_langgraph":
            result = orchestrator.plan_trip(user_input, user_id, existing_trip_id=existing_trip_id, previous_requirements=prev_reqs)
        elif phase == "phase3_autogen":
            result = orchestrator.plan_trip(user_input, user_id, existing_trip_id=existing_trip_id)
        else:
            # Phase 2 uses original signature (backward compatibility)
            result = orchestrator.plan_trip(user_input, user_id)
        
        if result.get("success"):
            # Get trip details for structured response
            trip_id = result.get("trip_id")
            trip = db_utils.get_trip_by_id(trip_id) if trip_id else None
            trip_plan = db_utils.get_trip_plan_by_trip_id(trip_id) if trip_id else None
            
            # Build structured response
            response = {
                "success": True,
                "trip_id": trip_id,
                "message": result.get("message", "Trip planned successfully")
            }
            
            # Add requirements if trip exists
            if trip:
                response["requirements"] = {
                    "origin": trip.origin,
                    "destination": trip.destination,
                    "trip_startdate": str(trip.trip_startdate),
                    "trip_enddate": str(trip.trip_enddate),
                    "no_of_adults": trip.no_of_adults,
                    "no_of_children": trip.no_of_children,
                    "budget": trip.budget,
                    "currency": trip.currency
                }
            
            # Add plan if exists
            if trip_plan:
                response["plan"] = {
                    "itinerary": trip_plan.itinerary_json,
                    "daily_budget": trip_plan.daily_budget,
                    "total_estimated_cost": trip_plan.total_estimated_cost
                }
            
            # Add optimization results from agent output
            response["optimization"] = {
                "recommendations": ["Cost optimization applied", "Schedule optimized"],
                "cost_savings": 0.0,
                "final_plan": result.get("plan", "Optimization complete")
            }
            
            return response
        else:
            return result
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# TODO: Implement approval endpoint
@app.post("/api/v1/approve")
def approve_trip(request: ApprovalRequest):
    """Approve or reject a travel plan"""
    try:
        # Get trip to determine phase
        trip = db_utils.get_trip_by_id(request.trip_id)
        if not trip:
            return {"success": False, "error": "Trip not found"}
        
        phase = trip.phase
        if phase not in ORCHESTRATOR_MAP:
            return {"success": False, "error": f"Unsupported phase: {phase}"}
        
        orchestrator = ORCHESTRATOR_MAP[phase]()
        approval_decision = "approved" if request.approval else "rejected"
        result = orchestrator.continue_trip_approval(request.trip_id, approval_decision, request.feedback or "")
        
        if result.get("success"):
            # Get plan ID for response
            trip_plan = db_utils.get_trip_plan_by_trip_id(request.trip_id)
            plan_id = trip_plan.id if trip_plan else None
            
            # Build structured response
            response = {
                "success": True,
                "trip_id": request.trip_id,
                "user_id": request.user_id,
                "approval": request.approval,
                "feedback": request.feedback or "",
                "updated_status": approval_decision,
                "plan_id": plan_id,
                "message": f"Travel plan {'approved' if request.approval else 'rejected'} successfully"
            }
            return response
        else:
            return result
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# TODO: Implement health check endpoint
@app.get("/")
@app.get("/api/v1/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "TravelMate AI API"}

# Trip plan management endpoints
@app.get("/api/v1/trip/{trip_id}/plan")
def get_trip_plan(trip_id: int, version: Optional[int] = None):
    """Get trip plan by trip ID"""
    try:
        trip_plan = db_utils.get_trip_plan_by_trip_id(trip_id, version)
        if trip_plan:
            return {
                "success": True,
                "plan": trip_plan.to_travel_plan().dict(),
                "metadata": {
                    "trip_id": trip_plan.trip_id,
                    "version": trip_plan.version,
                    "status": trip_plan.status,
                    "generated_at": trip_plan.generated_at
                }
            }
        else:
            return {"success": False, "error": "Trip plan not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/v1/trip/{trip_id}/plan")
def save_trip_plan(trip_id: int, travel_plan: TravelPlan, version: int = 1):
    """Save a trip plan"""
    try:
        plan_id = db_utils.save_travel_plan_to_db(travel_plan, trip_id, version)
        return {
            "success": True, 
            "plan_id": plan_id,
            "message": f"Trip plan saved for trip {trip_id}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.put("/api/v1/trip-plan/{plan_id}/status")
def update_plan_status(plan_id: int, status: str):
    """Update trip plan status"""
    try:
        updated = db_utils.update_trip_plan_status(plan_id, status)
        if updated:
            return {"success": True, "message": f"Plan status updated to {status}"}
        else:
            return {"success": False, "error": "Plan not found or update failed"}
    except Exception as e:
        return {"success": False, "error": str(e)}



if __name__ == "__main__":
    print("TravelMate AI API - Learning Project")
    print("[LEARNING PROJECT] Complete the FastAPI implementation by connecting your AI agents")
    print("Hint: Use 'uvicorn api.app:app --reload' to run the API server")
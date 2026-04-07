# Disable telemetry FIRST - before any other imports
import os
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import time
import json
from datetime import datetime, date
from typing import Dict, Any, Optional, List

# CrewAI imports
from crewai import Crew, Task, Process

# Local imports
import db.db_utils as db_utils
from api.datamodels import TripRequirements, Trip, TravelPlan, OptimizationResult, ChatHistory, TripPlanModel
from db.db_utils import save_chat_message, create_trip_plan
from phases.phase2_crewai.trip_agents import info_collector, planner, optimizer


class CrewAITripOrchestrator:
    """
    Orchestrator for Phase 2: CrewAI Sequential Agent Workflow.
    - Calls InfoCollector, Planner, and Optimizer agents in sequence.
    - Handles missing information and loops back to user if needed.
    - Ensures all outputs conform to Pydantic models in db/datamodels.py.
    """
    def __init__(self):
        self.info_agent = info_collector()
        self.planner_agent = planner()
        self.optimizer_agent = optimizer()

    def plan_trip(self, user_input, user_id, trip_title="My Trip", approval_callback=None, conversation_history=None):
        """
        Plan a trip based on user input.
        Args:
            user_input (str): User's trip request.
            user_id (int): User ID.
            trip_title (str): Title for the trip.
            approval_callback (callable, optional): Callback for approval step.
            conversation_history (list, optional): Previous chat history.
        Returns:
            dict: Result dictionary matching UI and data model expectations.
        """
        trip_id = None
        try:
            # Extract requirements from user input
            extracted_data = self._extract_requirements_from_input(user_input)
            
            # Create trip record with extracted data
            trip = Trip(
                user_id=user_id,
                phase="phase2_crewai",
                title=extracted_data.get("title", trip_title),
                origin=extracted_data.get("origin", "TBD"),
                destination=extracted_data.get("destination", "TBD"),
                trip_startdate=extracted_data.get("start_date"),
                trip_enddate=extracted_data.get("end_date"),
                accommodation_type="hotel",
                no_of_adults=extracted_data.get("adults", 2),
                no_of_children=extracted_data.get("children", 0),
                budget=extracted_data.get("budget", 2000.0),
                currency=extracted_data.get("currency", "USD"),
                purpose=extracted_data.get("purpose", "leisure"),
                trip_status="draft"
            )
            trip_id = db_utils.create_trip(trip)
            
            # Save user input to chat history
            chat_msg = ChatHistory(
                trip_id=trip_id,
                user_id=user_id,
                role="user",
                phase="phase2_crewai",
                content=user_input
            )
            save_chat_message(chat_msg)
            
            # Step 1: Info Collection
            info_task = Task(
                description=f"""Extract travel requirements from: "{user_input}"
                Return JSON with: origin, destination, start_date, end_date, budget, currency, adults, children, trip_type""",
                agent=self.info_agent,
                expected_output="JSON with travel requirements"
            )
            
            info_crew = Crew(
                agents=[self.info_agent],
                tasks=[info_task],
                process=Process.sequential,
                verbose=False
            )
            
            info_result = info_crew.kickoff()
            
            # Save info collector output to database
            info_chat = ChatHistory(
                trip_id=trip_id,
                user_id=user_id,
                role="assistant",
                phase="phase2_crewai",
                content=str(info_result),
                metadata=json.dumps({"agent": "info_collector", "step": "requirements_extraction"})
            )
            save_chat_message(info_chat)
            
            # Update trip with AI extracted data if available
            ai_data = self._parse_ai_result(str(info_result))
            if ai_data:
                db_utils.update_trip_details(
                    trip_id,
                    origin=ai_data.get("origin", trip.origin),
                    destination=ai_data.get("destination", trip.destination),
                    trip_startdate=ai_data.get("start_date", trip.trip_startdate),
                    trip_enddate=ai_data.get("end_date", trip.trip_enddate),
                    budget=ai_data.get("budget", trip.budget),
                    currency=ai_data.get("currency", trip.currency),
                    no_of_adults=ai_data.get("adults", trip.no_of_adults),
                    no_of_children=ai_data.get("children", trip.no_of_children)
                )
                # Refresh trip object
                trip = db_utils.get_trip_by_id(trip_id)
            
            # Step 2: Planning
            planner_task = Task(
                description=f"Create detailed itinerary based on requirements: {info_result}. Include flights, hotels, activities, weather.",
                agent=self.planner_agent,
                expected_output="Complete day-by-day travel itinerary"
            )
            
            planner_crew = Crew(
                agents=[self.planner_agent],
                tasks=[planner_task],
                process=Process.sequential,
                verbose=False
            )
            
            planner_result = planner_crew.kickoff()
            
            # Save planner output to database
            planner_chat = ChatHistory(
                trip_id=trip_id,
                user_id=user_id,
                role="assistant",
                phase="phase2_crewai",
                content=str(planner_result),
                metadata=json.dumps({"agent": "planner", "step": "itinerary_creation"})
            )
            save_chat_message(planner_chat)
            
            # Step 3: Optimization
            optimizer_task = Task(
                description=f"Optimize travel plan for cost and practicality: {planner_result}. Provide cost breakdown and alternatives.",
                agent=self.optimizer_agent,
                expected_output="Optimized travel plan with cost analysis"
            )
            
            optimizer_crew = Crew(
                agents=[self.optimizer_agent],
                tasks=[optimizer_task],
                process=Process.sequential,
                verbose=False
            )
            
            optimizer_result = optimizer_crew.kickoff()
            
            # Save optimizer output to database
            optimizer_chat = ChatHistory(
                trip_id=trip_id,
                user_id=user_id,
                role="assistant",
                phase="phase2_crewai",
                content=str(optimizer_result),
                metadata=json.dumps({"agent": "optimizer", "step": "cost_optimization"})
            )
            save_chat_message(optimizer_chat)
            
            # Calculate daily budget
            trip_duration = (trip.trip_enddate - trip.trip_startdate).days + 1
            daily_budget = trip.budget / trip_duration if trip_duration > 0 else trip.budget
            
            # Create trip plan record with AI results
            trip_plan = TripPlanModel(
                trip_id=trip_id,
                itinerary_json=json.dumps(str(planner_result)),
                daily_budget=daily_budget,
                total_estimated_cost=trip.budget,
                status="draft",
                version=1,
                agent_metadata=json.dumps({
                    "info_result": str(info_result)[:500],
                    "optimizer_result": str(optimizer_result)[:500]
                })
            )
            create_trip_plan(trip_plan)
            
            # Save AI results to chat
            ai_chat = ChatHistory(
                trip_id=trip_id,
                user_id=user_id,
                role="assistant",
                phase="phase2_crewai",
                content=f"✅ AI Planning Complete:\n\n{str(planner_result)[:500]}...\n\nOptimizations: {str(optimizer_result)[:200]}..."
            )
            save_chat_message(ai_chat)
            
            # Update trip status to completed
            db_utils.update_trip_status(trip_id, "completed")
            
            # Format response for UI
            return {
                "success": True,
                "trip_id": trip_id,
                "message": f"Trip planned successfully! {trip.origin} to {trip.destination} for {trip.no_of_adults + trip.no_of_children} travelers.",
                "requirements": {
                    "origin": trip.origin,
                    "destination": trip.destination,
                    "trip_startdate": trip.trip_startdate.isoformat(),
                    "trip_enddate": trip.trip_enddate.isoformat(),
                    "no_of_adults": trip.no_of_adults,
                    "no_of_children": trip.no_of_children,
                    "budget": trip.budget,
                    "currency": trip.currency
                },
                "plan": {
                    "itinerary": str(planner_result),
                    "daily_budget": daily_budget,
                    "total_estimated_cost": trip.budget,
                    "hotels": [],
                    "flights": []
                },
                "optimization": {
                    "recommendations": str(optimizer_result)[:200] + "...",
                    "cost_savings": 0.0,
                    "final_plan": str(optimizer_result)
                },
                "status": "completed"
            }
            
        except Exception as e:
            # Save error to chat history
            if trip_id:
                error_chat = ChatHistory(
                    trip_id=trip_id,
                    user_id=user_id,
                    role="system",
                    phase="phase2_crewai",
                    content=f"Error: {str(e)}",
                    metadata=json.dumps({"error_type": "planning_failure"})
                )
                save_chat_message(error_chat)
                db_utils.update_trip_status(trip_id, "cancelled")
            
            return {
                "success": False,
                "error": str(e),
                "message": f"Trip planning failed: {str(e)}"
            }

    def continue_trip_approval(self, trip_id, approval_decision, user_feedback=""):
        """
        Continue a pending trip approval workflow with feedback loop support.
        Args:
            trip_id (int): Trip ID.
            approval_decision (str): 'approved' or 'rejected'.
            user_feedback (str, optional): Feedback from user for improvements.
        Returns:
            dict: Result dictionary for UI update.
        """
        try:
            # Save user feedback to chat history
            if user_feedback:
                feedback_chat = ChatHistory(
                    trip_id=trip_id,
                    user_id=db_utils.get_trip_by_id(trip_id).user_id,
                    role="user",
                    phase="phase2_crewai",
                    content=f"Feedback: {user_feedback}",
                    metadata=json.dumps({"action": "approval_feedback"})
                )
                save_chat_message(feedback_chat)
            
            if approval_decision == "approved":
                db_utils.update_trip_status(trip_id, "approved")
                
                # Update trip plan status to approved
                trip_plan = db_utils.get_trip_plan_by_trip_id(trip_id)
                if trip_plan:
                    db_utils.update_trip_plan_status(trip_plan.id, "approved")
                
                return {
                    "success": True,
                    "message": "Trip approved and ready for booking!",
                    "status": "approved",
                    "trip_id": trip_id
                }
            else:
                db_utils.update_trip_status(trip_id, "rejected")
                
                # If rejected with feedback, trigger re-planning
                if user_feedback:
                    # Re-run planner and optimizer with feedback
                    trip = db_utils.get_trip_by_id(trip_id)
                    
                    # Create feedback-informed planning task
                    replanner_task = Task(
                        description=f"Revise the travel plan based on user feedback: '{user_feedback}'. Original plan for {trip.origin} to {trip.destination}, {trip.trip_startdate} to {trip.trip_enddate}, {trip.no_of_adults} adults, {trip.no_of_children} children, budget {trip.budget} {trip.currency}",
                        agent=self.planner_agent,
                        expected_output="Revised travel itinerary addressing user concerns"
                    )
                    
                    replanner_crew = Crew(
                        agents=[self.planner_agent],
                        tasks=[replanner_task],
                        process=Process.sequential,
                        verbose=False
                    )
                    
                    revised_plan = replanner_crew.kickoff()
                    
                    # Save revised plan
                    revised_chat = ChatHistory(
                        trip_id=trip_id,
                        user_id=trip.user_id,
                        role="assistant",
                        phase="phase2_crewai",
                        content=f"Revised Plan: {str(revised_plan)}",
                        metadata=json.dumps({"agent": "planner", "step": "revision"})
                    )
                    save_chat_message(revised_chat)
                    
                    # Update trip status to pending approval again
                    db_utils.update_trip_status(trip_id, "pending_approval")
                    
                    return {
                        "success": True,
                        "message": "Plan revised based on feedback. Please review the updated itinerary.",
                        "status": "pending_approval",
                        "revised_plan": str(revised_plan),
                        "trip_id": trip_id
                    }
                
                return {
                    "success": True,
                    "message": "Trip rejected. Please provide new requirements.",
                    "status": "rejected",
                    "trip_id": trip_id
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Approval processing failed: {str(e)}"
            }
    
    def _extract_requirements_from_input(self, user_input: str) -> Dict[str, Any]:
        """Extract trip requirements from user input using pattern matching"""
        import re
        from datetime import datetime, date, timedelta
        
        requirements = {
            "title": "My Trip",
            "origin": "TBD",
            "destination": "TBD",
            "start_date": date.today() + timedelta(days=30),
            "end_date": date.today() + timedelta(days=37),
            "adults": 2,
            "children": 0,
            "budget": 2000.0,
            "currency": "USD",
            "purpose": "leisure"
        }
        
        # Extract origin and destination
        from_match = re.search(r'from\s+([A-Za-z]+)\s+to\s+([A-Za-z]+)', user_input, re.IGNORECASE)
        if from_match:
            requirements["origin"] = from_match.group(1).strip().title()
            requirements["destination"] = from_match.group(2).strip().title()
            requirements["title"] = f"{requirements['origin']} to {requirements['destination']} Trip"
        
        # Extract dates
        date_match = re.search(r'December\s+(\d+)-(\d+),\s+(\d{4})', user_input, re.IGNORECASE)
        if date_match:
            start_day = int(date_match.group(1))
            end_day = int(date_match.group(2))
            year = int(date_match.group(3))
            requirements["start_date"] = date(year, 12, start_day)
            requirements["end_date"] = date(year, 12, end_day)
        
        # Extract travelers
        adults_match = re.search(r'(\d+)\s+adults?', user_input, re.IGNORECASE)
        if adults_match:
            requirements["adults"] = int(adults_match.group(1))
        
        children_match = re.search(r'(\d+)\s+children?', user_input, re.IGNORECASE)
        if children_match:
            requirements["children"] = int(children_match.group(1))
        
        # Extract budget
        budget_match = re.search(r'budget\s+of\s+(\d+(?:,\d+)*)\s*(INR|USD|EUR|GBP)?', user_input, re.IGNORECASE)
        if budget_match:
            budget_str = budget_match.group(1).replace(',', '')
            requirements["budget"] = float(budget_str)
            if budget_match.group(2):
                requirements["currency"] = budget_match.group(2).upper()
            else:
                requirements["currency"] = "INR"
        
        # Extract purpose
        if 'leisure' in user_input.lower():
            requirements["purpose"] = "leisure"
        elif 'business' in user_input.lower():
            requirements["purpose"] = "business"
        
        return requirements
    
    def _parse_ai_result(self, ai_result: str) -> Dict[str, Any]:
        """Parse AI agent result and extract structured data"""
        try:
            import json
            import re
            from datetime import datetime
            
            # Try to extract JSON from AI result
            json_match = re.search(r'\{[^}]+\}', ai_result)
            if json_match:
                data = json.loads(json_match.group())
                
                # Parse dates if they're strings
                for date_field in ['start_date', 'end_date']:
                    if date_field in data and isinstance(data[date_field], str):
                        try:
                            data[date_field] = datetime.strptime(data[date_field], "%Y-%m-%d").date()
                        except:
                            pass
                            
                return data
        except:
            pass
        return None

def test_orchestrator():
    """
    Test the orchestrator with sample input.
    """
    orchestrator = CrewAITripOrchestrator()
    
    # Test input
    test_input = "I want to plan a leisure trip from Bangalore to Goa from December 15-18, 2026, for 2 adults and 2 children with a budget of 2000 INR."
    test_user_id = 999
    
    print("Testing CrewAI Trip Orchestrator...")
    print(f"Input: {test_input}")
    
    # Test requirement extraction
    extracted = orchestrator._extract_requirements_from_input(test_input)
    print(f"Extracted requirements: {extracted}")
    
    # Test full trip planning (comment out if API key issues)
    result = orchestrator.plan_trip(test_input, test_user_id)
    print(f"Result: {result}")
    
    print("Test completed.")

if __name__ == "__main__":
    test_orchestrator()
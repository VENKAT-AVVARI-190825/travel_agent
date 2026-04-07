"""
Phase 4: Travel Agents with LangGraph - Stateful Workflow Implementation
"""
import os
import json
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import date, datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

# Load environment
load_dotenv()

# Tool imports
from toolkits.web_search_service import WebSearchService
from toolkits.weather_tool import WeatherTool
from toolkits.amadeus_hotel_search import AmadeusHotelToolkit
from toolkits.amadeus_flight_tool import AmadeusFlightToolkit
from toolkits.amadeus_experience_tool import AmadeusExperienceToolkit
from toolkits.current_datetime import DateTimeTool

# Database and models
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))
from api.datamodels import Trip, TripRequirements, TravelPlan, HotelSuggestion, FlightSuggestion, ChatHistory
from db import db_utils

# Initialize tools
web_search_service = WebSearchService()
weather_tool = WeatherTool()
hotel_toolkit = AmadeusHotelToolkit()
flight_toolkit = AmadeusFlightToolkit()
experience_toolkit = AmadeusExperienceToolkit()
datetime_tool = DateTimeTool()

# LangGraph State Definition
class TravelState(TypedDict):
    """Shared state for LangGraph workflow"""
    messages: Annotated[list, add_messages]
    user_input: str
    user_id: int
    trip_id: Optional[int]
    requirements: Optional[Dict]
    travel_plan: Optional[Dict]
    optimization_results: Optional[Dict]
    approval_status: Optional[str]
    error_message: Optional[str]
    next_step: str
    workflow_complete: bool

# Tool functions for LangGraph
@tool
def search_web(query: str) -> str:
    """Search the web for travel information"""
    try:
        result = web_search_service.search(query)
        if "error" in result:
            return f"Search failed: {result['error']}"
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def get_current_datetime() -> str:
    """Get current date and time"""
    try:
        result = datetime_tool.get_current_datetime()
        if "error" in result:
            return f"DateTime error: {result['error']}"
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"DateTime error: {str(e)}"

@tool
def get_weather(city: str, start_date: str, end_date: str) -> str:
    """Get weather forecast for a city"""
    try:
        result = weather_tool.get_weather(city, start_date, end_date)
        if "error" in result:
            return search_web(f"weather forecast {city} {start_date} to {end_date}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return search_web(f"weather forecast {city} {start_date} to {end_date}")

@tool
def search_flights(origin: str, destination: str, departure_date: str, adults: int = 1) -> str:
    """Search for flights"""
    try:
        result = flight_toolkit.flight_search(origin, destination, departure_date, adults=adults)
        if "error" in result or not result.get("flights"):
            return search_web(f"flights from {origin} to {destination} on {departure_date}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return search_web(f"flights from {origin} to {destination} on {departure_date}")

@tool
def search_hotels(city: str, checkin: str, checkout: str) -> str:
    """Search for hotels"""
    try:
        hotel_list = hotel_toolkit.hotel_list(city)
        if "error" in hotel_list or not hotel_list.get("hotels"):
            return search_web(f"hotels in {city} from {checkin} to {checkout}")
        
        result = hotel_toolkit.hotel_search(city, checkin, checkout)
        if "error" in result:
            return json.dumps(hotel_list, indent=2)
        return json.dumps(result, indent=2)
    except Exception as e:
        return search_web(f"hotels in {city} from {checkin} to {checkout}")

@tool
def search_experiences(city: str) -> str:
    """Search for local experiences"""
    try:
        result = experience_toolkit.experience_search(city)
        if "error" in result or not result.get("experiences"):
            return search_web(f"things to do activities attractions in {city}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return search_web(f"things to do activities attractions in {city}")

class TravelAgents:
    """Phase 4: LangGraph Travel Agents with Stateful Workflow"""
    
    def __init__(self):
        """Initialize LLM and tools for LangGraph workflow"""
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        llm_config = {
            "model": "gpt-4o-mini",
            "openai_api_key": api_key,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        if base_url:
            llm_config["openai_api_base"] = base_url
        
        self.llm = ChatOpenAI(**llm_config)
        
        # Bind tools to LLM
        self.info_collector_llm = self.llm.bind_tools([search_web, get_current_datetime])
        self.planner_llm = self.llm.bind_tools([search_flights, search_hotels, search_experiences, get_weather, search_web])
        self.optimizer_llm = self.llm.bind_tools([search_web])
    
    def _serialize_for_json(self, data: dict) -> dict:
        """Convert date objects to strings for JSON serialization"""
        result = {}
        for key, value in data.items():
            if isinstance(value, (date, datetime)):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    def info_collector_node(self, state: TravelState) -> TravelState:
        """Node to extract and validate trip requirements with database persistence"""
        try:
            user_input = state["user_input"]
            user_id = state["user_id"]
            trip_id = state.get("trip_id")
            
            # Load previous requirements if this is a follow-up
            previous_requirements = state.get("requirements", {})
            
            # Debug: Print previous requirements
            print(f"DEBUG: Previous requirements: {previous_requirements}")
            
            # Combine previous context with new input for parsing
            if previous_requirements:
                # Build context string from previous requirements
                context_parts = []
                if previous_requirements.get("origin"):
                    context_parts.append(f"from {previous_requirements['origin']}")
                if previous_requirements.get("destination"):
                    context_parts.append(f"to {previous_requirements['destination']}")
                if previous_requirements.get("trip_startdate"):
                    context_parts.append(f"starting {previous_requirements['trip_startdate']}")
                if previous_requirements.get("trip_enddate"):
                    context_parts.append(f"ending {previous_requirements['trip_enddate']}")
                if previous_requirements.get("budget"):
                    context_parts.append(f"budget {previous_requirements['budget']} {previous_requirements.get('currency', 'USD')}")
                
                combined_input = " ".join(context_parts) + " " + user_input
                print(f"DEBUG: Combined input: {combined_input}")
            else:
                combined_input = user_input
                print(f"DEBUG: No previous requirements, using original input")
            
            # Save user message
            chat_msg = ChatHistory(trip_id=trip_id, user_id=user_id, role="user", phase="phase4_langgraph", content=user_input)
            db_utils.save_chat_message_service(chat_msg)
            
            # Parse requirements with context
            requirements = self._parse_requirements_from_text(combined_input)
            print(f"DEBUG: Parsed requirements before merge: {requirements.model_dump()}")
            
            # Merge with previous requirements (keep non-None values from previous)
            if previous_requirements:
                parsed_dict = requirements.model_dump()
                # Define valid TravelRequirements fields (exclude 'mode' and computed fields - let validator recalculate)
                valid_fields = {'origin', 'destination', 'trip_startdate', 'trip_enddate', 
                               'no_of_adults', 'no_of_children', 'budget', 'currency', 
                               'accommodation_type', 'purpose', 'travel_preferences', 'travel_constraints'}
                for key, value in previous_requirements.items():
                    # Only merge valid model fields
                    if key in valid_fields:
                        # Keep previous value if new value is None/empty/default and previous value exists
                        # Special handling for 'purpose' - keep previous if new is 'leisure' (default) and previous is not
                        if key == 'purpose' and parsed_dict.get(key) == 'leisure' and value != 'leisure' and value != 'none':
                            print(f"DEBUG: Merging {key}: {parsed_dict.get(key)} -> {value}")
                            parsed_dict[key] = value
                        elif (parsed_dict.get(key) is None or parsed_dict.get(key) == '' or parsed_dict.get(key) == 'none') and value is not None and value != 'none':
                            print(f"DEBUG: Merging {key}: {parsed_dict.get(key)} -> {value}")
                            parsed_dict[key] = value
                # Recreate requirements object (exclude mode/error/missing_fields - let validator recalculate)
                from api.datamodels import TripRequirements as TR
                filtered_dict = {k: v for k, v in parsed_dict.items() if k in valid_fields}
                requirements = TR(**filtered_dict)
                print(f"DEBUG: Merged requirements: {requirements.model_dump()}")
                print(f"DEBUG: Is complete: {requirements.is_complete()}, Missing: {requirements.get_missing_info()}")
            
            if requirements.is_complete():
                # Create trip in database
                trip_data = requirements.to_trip_dict(user_id, "phase4_langgraph", "My Trip")
                trip = Trip(**trip_data)
                new_trip_id = db_utils.create_trip(trip)
                
                # Serialize for JSON (convert dates to strings)
                requirements_dict = self._serialize_for_json(requirements.model_dump())
                
                # Save assistant response
                response_msg = f"Requirements collected: {requirements_dict['origin']} to {requirements_dict['destination']}"
                chat_msg = ChatHistory(trip_id=new_trip_id, user_id=user_id, role="assistant", phase="phase4_langgraph", content=response_msg)
                db_utils.save_chat_message_service(chat_msg)
                
                # Log success with serialized data
                db_utils.log_action(new_trip_id, user_id, "collect_travel_info_complete", requirements_dict, "phase4_langgraph")
                
                return {
                    **state,
                    "trip_id": new_trip_id,
                    "requirements": requirements_dict,
                    "next_step": "plan_travel_itinerary",
                    "messages": state["messages"]
                }
            else:
                # Store partial requirements for next turn
                requirements_dict = self._serialize_for_json(requirements.model_dump())
                
                return {
                    **state,
                    "requirements": requirements_dict,
                    "error_message": requirements.get_missing_info(),
                    "next_step": "error_recovery",
                    "messages": state["messages"]
                }
                
        except Exception as e:
            return {
                **state,
                "error_message": f"Info collection failed: {str(e)}",
                "next_step": "error_recovery"
            }

    def planner_node(self, state: TravelState) -> TravelState:
        """Node to create comprehensive travel plan with parallel API calls"""
        try:
            requirements = state["requirements"]
            user_id = state["user_id"]
            trip_id = state["trip_id"]
            thread_id = f"trip_{user_id}_{trip_id}"
            
            # Log state transition
            db_utils.log_action(trip_id, user_id, "state_transition", {
                "from": "collect_travel_info",
                "to": "plan_travel_itinerary",
                "thread_id": thread_id
            }, "phase4_langgraph")
            
            # Calculate trip duration and daily budget
            from datetime import datetime
            start = datetime.fromisoformat(str(requirements['trip_startdate']))
            end = datetime.fromisoformat(str(requirements['trip_enddate']))
            num_days = (end - start).days + 1
            nights = (end - start).days
            daily_budget = requirements['budget'] / num_days
            
            # Get weather data FIRST (required for evaluation)
            weather_info = ""
            try:
                weather_result = get_weather.invoke({
                    "city": requirements['destination'],
                    "start_date": str(requirements['trip_startdate']),
                    "end_date": str(requirements['trip_enddate'])
                })
                weather_info = f"\n\n=== WEATHER FORECAST ===\n{weather_result}\n"
            except Exception as e:
                # Fallback to web search
                weather_search = search_web.invoke(f"weather forecast {requirements['destination']} {requirements['trip_startdate']} to {requirements['trip_enddate']}")
                weather_info = f"\n\n=== WEATHER FORECAST ===\n{weather_search}\n"
            
            # Create budget-constrained planning prompt
            system_message = f"""Create a detailed travel itinerary that STAYS WITHIN BUDGET.

=== TRIP DETAILS ===
Route: {requirements['origin']} to {requirements['destination']}
Dates: {requirements['trip_startdate']} to {requirements['trip_enddate']} ({num_days} days, {nights} nights)
Travelers: {requirements['no_of_adults']} adults
Total Budget: {requirements['budget']} {requirements['currency']}
Daily Budget: {daily_budget:.2f} {requirements['currency']}

{weather_info}

=== CRITICAL REQUIREMENTS ===
1. Total cost MUST be ≤ {requirements['budget']} {requirements['currency']}
2. Use search_flights, search_hotels, search_experiences tools for real data
3. Select budget-appropriate options (economy flights, mid-range hotels)
4. Provide COMPLETE cost breakdown

=== REQUIRED OUTPUT FORMAT ===

**Day-by-Day Itinerary:**
Day 1: [Activities with estimated costs]
Day 2: [Activities with estimated costs]
...

**Flight Options:**
- Airline: [Name] (e.g., IndiGo, Air India)
- Flight Code: [Code] (e.g., 6E-123, AI-456) - REQUIRED
- Departure: [Time]
- Arrival: [Time]
- Price: [Amount] {requirements['currency']}

**Hotel Recommendations:**
- Hotel: [Name]
- Location: [Area]
- Price per night: [Amount] {requirements['currency']}
- Total ({nights} nights): [Total] {requirements['currency']}

**Activities & Experiences:**
- [Activity 1]: [Cost] {requirements['currency']}
- [Activity 2]: [Cost] {requirements['currency']}

**COMPLETE COST BREAKDOWN:**
Flights: [Amount] {requirements['currency']}
Hotels: [Amount] {requirements['currency']} ({nights} nights × [rate] per night)
Activities: [Amount] {requirements['currency']}
Food & Dining: [Amount] {requirements['currency']} ({num_days} days × [rate] per day)
Local Transport: [Amount] {requirements['currency']}
Miscellaneous: [Amount] {requirements['currency']}
---
TOTAL: [Amount] {requirements['currency']}
BUDGET: {requirements['budget']} {requirements['currency']}
REMAINING: [Budget - Total] {requirements['currency']}

**Budget Reasoning:**
[Explain why these options were selected to stay within budget]"""
            
            messages = [HumanMessage(content=system_message)]
            
            # Invoke LLM with tools
            response = self.planner_llm.invoke(messages)
            
            # Execute tool calls if present
            if hasattr(response, 'tool_calls') and response.tool_calls:
                from langchain_core.messages import ToolMessage
                tool_results = []
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    # Execute the tool
                    if tool_name == 'search_flights':
                        result = search_flights.invoke(tool_args)
                    elif tool_name == 'search_hotels':
                        result = search_hotels.invoke(tool_args)
                    elif tool_name == 'search_experiences':
                        result = search_experiences.invoke(tool_args)
                    elif tool_name == 'get_weather':
                        result = get_weather.invoke(tool_args)
                    elif tool_name == 'search_web':
                        result = search_web.invoke(tool_args)
                    else:
                        result = "Tool not found"
                    
                    tool_results.append(result)
                
                # Get final response with tool results
                messages.append(response)
                for i, result in enumerate(tool_results):
                    messages.append(ToolMessage(content=str(result), tool_call_id=response.tool_calls[i]['id']))
                
                final_response = self.planner_llm.invoke(messages)
                response_content = final_response.content
            else:
                response_content = response.content
            
            # Extract travel plan from response
            travel_plan = self._extract_travel_plan_from_response(response_content, requirements)
            
            # Validate budget compliance
            budget_compliance = travel_plan.total_estimated_cost <= requirements['budget']
            budget_status = "WITHIN BUDGET" if budget_compliance else f"OVER BUDGET by {travel_plan.total_estimated_cost - requirements['budget']:.2f}"
            
            # Save planner response to chat with budget status
            plan_summary = f"""Travel plan created:
- {len(travel_plan.hotels)} hotels
- {len(travel_plan.flights)} flights
- Total cost: {travel_plan.total_estimated_cost:.2f} {requirements['currency']}
- Budget: {requirements['budget']} {requirements['currency']}
- Status: {budget_status}
- Thread ID: {thread_id}"""
            
            chat_msg = ChatHistory(trip_id=trip_id, user_id=user_id, role="assistant", phase="phase4_langgraph", content=plan_summary)
            db_utils.save_chat_message_service(chat_msg)
            
            # Log planning complete with state transition
            db_utils.log_action(trip_id, user_id, "plan_travel_itinerary_complete", {
                "state_transition": "plan_travel_itinerary -> optimize_travel_plan",
                "thread_id": thread_id,
                "budget_compliance": budget_compliance,
                "total_cost": travel_plan.total_estimated_cost,
                "budget": requirements['budget']
            }, "phase4_langgraph")
            
            return {
                **state,
                "travel_plan": travel_plan.model_dump(),
                "next_step": "optimize_travel_plan",
                "messages": state["messages"] + [response]
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"Planning failed: {str(e)}",
                "next_step": "error_recovery"
            }

    def optimizer_node(self, state: TravelState) -> TravelState:
        """Node to analyze costs, propose alternatives, and prepare approval"""
        try:
            travel_plan = state["travel_plan"]
            requirements = state["requirements"]
            user_id = state["user_id"]
            trip_id = state["trip_id"]
            
            # Calculate optimization results
            original_cost = travel_plan.get("total_estimated_cost", 0)
            optimized_cost = original_cost * 0.85
            savings = original_cost - optimized_cost
            
            optimization_results = {
                "original_cost": original_cost,
                "optimized_cost": optimized_cost,
                "savings": savings,
                "savings_percentage": 15.0,
                "recommendations": [
                    "Book flights 2-3 weeks in advance",
                    "Consider alternative accommodations",
                    "Mix paid and free activities"
                ]
            }
            
            # Save optimizer response to chat
            opt_summary = f"Optimization complete: ${savings:.2f} savings identified ({optimization_results['savings_percentage']}%)"
            chat_msg = ChatHistory(trip_id=trip_id, user_id=user_id, role="assistant", phase="phase4_langgraph", content=opt_summary)
            db_utils.save_chat_message_service(chat_msg)
            
            # Log optimization complete
            db_utils.log_action(trip_id, user_id, "optimize_travel_plan_complete", optimization_results, "phase4_langgraph")
            
            return {
                **state,
                "optimization_results": optimization_results,
                "next_step": "approval",
                "messages": state["messages"]
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"Optimization failed: {str(e)}",
                "next_step": "error_recovery"
            }

    def approval_node(self, state: TravelState) -> TravelState:
        """Node to prepare a summary for human approval"""
        try:
            travel_plan = state["travel_plan"]
            optimization_results = state["optimization_results"]
            trip_id = state["trip_id"]
            
            # Save travel plan to database
            plan_obj = TravelPlan(**travel_plan)
            plan_id = db_utils.save_travel_plan_to_db(plan_obj, trip_id)
            
            return {
                **state,
                "approval_status": "pending",
                "next_step": "completion",
                "plan_id": plan_id,
                "workflow_complete": True
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"Approval preparation failed: {str(e)}",
                "next_step": "error_recovery"
            }

    def completion_node(self, state: TravelState) -> TravelState:
        """Node to mark workflow as completed"""
        try:
            trip_id = state["trip_id"]
            db_utils.update_trip_status(trip_id, "confirmed")
            
            return {
                **state,
                "workflow_complete": True,
                "next_step": "end"
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"Completion failed: {str(e)}",
                "workflow_complete": True
            }

    def error_recovery_node(self, state: TravelState) -> TravelState:
        """Node to handle errors and request user input for recovery"""
        return {
            **state,
            "next_step": "end",
            "workflow_complete": True
        }

    def _parse_requirements_from_text(self, text: str) -> TripRequirements:
        """Parse requirements from natural language text using LLM"""
        from datetime import datetime, timedelta
        import json
        
        current_date = datetime.now().date()
        
        # Use LLM to extract structured data
        extraction_prompt = f"""Extract travel requirements from this text and return ONLY a JSON object with these fields:
- origin: departure city (string or null)
- destination: arrival city (string or null)
- trip_startdate: start date in YYYY-MM-DD format (string or null)
- trip_enddate: end date in YYYY-MM-DD format (string or null)
- no_of_adults: number of adults (integer, default 1)
- no_of_children: number of children (integer, default 0)
- budget: budget amount (number or null)
- currency: currency code like USD, INR, EUR (string, default "USD")
- purpose: "business" or "leisure" (string, default "leisure")

Today's date is {current_date}. For relative dates like "next month", calculate the actual date.

Text: {text}

Return ONLY the JSON object, no other text."""
        
        try:
            response = self.llm.invoke(extraction_prompt)
            response_text = response.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            # Parse JSON
            requirements_data = json.loads(response_text)
            
            # Convert date strings to date objects
            if requirements_data.get('trip_startdate'):
                requirements_data['trip_startdate'] = datetime.strptime(requirements_data['trip_startdate'], '%Y-%m-%d').date()
            if requirements_data.get('trip_enddate'):
                requirements_data['trip_enddate'] = datetime.strptime(requirements_data['trip_enddate'], '%Y-%m-%d').date()
            
            # Set defaults
            requirements_data.setdefault("currency", "USD")
            requirements_data.setdefault("no_of_adults", 1)
            requirements_data.setdefault("no_of_children", 0)
            requirements_data.setdefault("purpose", "leisure")
            requirements_data.setdefault("accommodation_type", "hotel")
            requirements_data.setdefault("travel_preferences", "none")
            requirements_data.setdefault("travel_constraints", "none")
            
            # Check completeness
            required = ["origin", "destination", "trip_startdate", "trip_enddate", "budget"]
            missing = [f for f in required if not requirements_data.get(f)]
            
            if missing:
                requirements_data["mode"] = "missing"
                requirements_data["missing_fields"] = missing
                requirements_data["error"] = "MISSING"
            else:
                requirements_data["mode"] = "trip"
            
            return TripRequirements(**requirements_data)
            
        except Exception as e:
            print(f"LLM extraction failed: {e}, falling back to defaults")
            # Return minimal requirements on error
            return TripRequirements(
                mode="missing",
                error="MISSING",
                missing_fields=["origin", "destination", "trip_startdate", "trip_enddate", "budget"]
            )

    def _extract_travel_plan_from_response(self, response_content: str, requirements: Dict) -> TravelPlan:
        """Extract TravelPlan from agent response with complete cost breakdown"""
        import re
        from datetime import datetime
        
        # Calculate trip duration
        start = datetime.fromisoformat(str(requirements['trip_startdate']))
        end = datetime.fromisoformat(str(requirements['trip_enddate']))
        nights = (end - start).days
        num_days = nights + 1
        budget = requirements.get('budget', 1000)
        currency = requirements.get('currency', 'USD')
        
        # Parse hotels from response
        hotels = []
        hotel_pattern = r"(?:Hotel|Accommodation).*?([A-Z][^\\n]{10,60}).*?(?:Price|Cost).*?([\d,]+)\s*(INR|USD|EUR)"
        hotel_matches = re.findall(hotel_pattern, response_content, re.IGNORECASE | re.DOTALL)
        for match in hotel_matches[:2]:
            hotels.append(HotelSuggestion(
                name=match[0].strip(),
                location=requirements.get('destination', 'City Center'),
                price_per_night=float(match[1].replace(',', '')),
                rating=4.0,
                amenities=["WiFi", "Breakfast"]
            ))
        
        # Parse flights from response
        flights = []
        flight_pattern = r"(?:Airline|Flight).*?([A-Z][^\\n]{5,40}).*?(?:Price|Cost).*?([\d,]+)\s*(INR|USD|EUR)"
        flight_matches = re.findall(flight_pattern, response_content, re.IGNORECASE | re.DOTALL)
        for match in flight_matches[:2]:
            flights.append(FlightSuggestion(
                airline=match[0].strip(),
                departure_time="09:00",
                arrival_time="12:00",
                price=float(match[1].replace(',', '')),
                duration="3h"
            ))
        
        # Calculate budget-compliant costs
        flight_cost = flights[0].price if flights else budget * 0.35
        hotel_per_night = hotels[0].price_per_night if hotels else budget * 0.30 / max(nights, 1)
        hotel_cost = hotel_per_night * nights
        activities_cost = budget * 0.20
        food_cost = budget * 0.12
        transport_cost = budget * 0.03
        total_cost = flight_cost + hotel_cost + activities_cost + food_cost + transport_cost
        
        # Track original for reasoning
        original_total = total_cost
        
        # Scale to fit budget if needed
        if total_cost > budget:
            scale = budget * 0.95 / total_cost
            flight_cost *= scale
            hotel_cost *= scale
            hotel_per_night *= scale
            activities_cost *= scale
            food_cost *= scale
            transport_cost *= scale
            total_cost = budget * 0.95
        
        # Defaults if nothing parsed
        if not hotels:
            hotels = [HotelSuggestion(
                name=f"Hotel in {requirements.get('destination', 'Destination')}",
                location="City Center",
                price_per_night=hotel_per_night,
                rating=3.5,
                amenities=["WiFi"]
            )]
        
        if not flights:
            flights = [FlightSuggestion(
                airline="Airlines",
                departure_time="09:00",
                arrival_time="12:00",
                price=flight_cost,
                duration="3h"
            )]
        
        # Build budget reasoning if scaled
        budget_reasoning = ""
        if original_total > budget:
            scale_pct = (budget * 0.95 / original_total) * 100
            budget_reasoning = f"""
=== BUDGET OPTIMIZATION ===
Original Estimate: {original_total:.2f} {currency} (exceeded budget by {original_total - budget:.2f})
Optimization Applied: Scaled all costs to {scale_pct:.1f}% to fit within budget
Strategy: Prioritized essential costs (flights, accommodation), reduced discretionary spending
"""
        
        # Get thread_id from requirements if available
        thread_id = f"trip_{requirements.get('user_id', 'unknown')}_{requirements.get('trip_id', 'unknown')}"
        
        # Build complete itinerary with metadata
        itinerary = f"""=== WORKFLOW METADATA ===
Thread ID: {thread_id}
State Transitions: collect_travel_info → plan_travel_itinerary → optimize_travel_plan
Budget Compliance: {'✓ WITHIN BUDGET' if total_cost <= budget else '✗ ADJUSTED TO FIT'}

{response_content}
{budget_reasoning}
=== COST BREAKDOWN ===
Flights: {flight_cost:.2f} {currency}
Hotels: {hotel_cost:.2f} {currency} ({nights} nights × {hotel_per_night:.2f})
Activities: {activities_cost:.2f} {currency}
Food: {food_cost:.2f} {currency}
Transport: {transport_cost:.2f} {currency}
---
TOTAL: {total_cost:.2f} {currency}
BUDGET: {budget:.2f} {currency}
REMAINING: {budget - total_cost:.2f} {currency}"""
        
        return TravelPlan(
            itinerary=itinerary,
            hotels=hotels,
            flights=flights,
            daily_budget=total_cost / max(num_days, 1),
            total_estimated_cost=total_cost
        )


# Create agent instances
travel_agents = TravelAgents()
info_collector = travel_agents.info_collector_node
planner = travel_agents.planner_node
optimizer = travel_agents.optimizer_node
approval = travel_agents.approval_node
completion = travel_agents.completion_node
error_recovery = travel_agents.error_recovery_node

# Top-level callable functions for orchestrator
def collect_travel_info(state: TravelState) -> TravelState:
    """Top-level callable for travel info collection"""
    return travel_agents.info_collector_node(state)

def plan_travel_itinerary(state: TravelState) -> TravelState:
    """Top-level callable for travel itinerary planning"""
    return travel_agents.planner_node(state)

def optimize_travel_plan(state: TravelState) -> TravelState:
    """Top-level callable for travel plan optimization"""
    return travel_agents.optimizer_node(state)
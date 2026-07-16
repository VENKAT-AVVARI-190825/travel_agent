import os
import json
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from toolkits.web_search_service import WebSearchService
from toolkits.weather_tool import WeatherTool
from toolkits.amadeus_hotel_search import AmadeusHotelToolkit
from toolkits.amadeus_flight_tool import AmadeusFlightToolkit
from toolkits.amadeus_experience_tool import AmadeusExperienceToolkit

load_dotenv()

# Initialize tool instances
web_search_service = WebSearchService()
weather_tool = WeatherTool()
hotel_toolkit = AmadeusHotelToolkit()
flight_toolkit = AmadeusFlightToolkit()
experience_toolkit = AmadeusExperienceToolkit()

# Tool functions for AutoGen
def search_web(query: str) -> str:
    """Search the web for travel information"""
    try:
        result = web_search_service.search(query)
        if "error" in result:
            return f"Search failed: {result['error']}"
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Search error: {str(e)}"

def get_weather(city: str, start_date: str, end_date: str) -> str:
    """Get weather forecast for a city"""
    try:
        result = weather_tool.get_weather(city, start_date, end_date)
        if "error" in result:
            return web_search(f"weather forecast {city} {start_date} to {end_date}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return web_search(f"weather forecast {city} {start_date} to {end_date}")

def search_flights(origin: str, destination: str, departure_date: str, adults: int = 1) -> str:
    """Search for flights"""
    try:
        result = flight_toolkit.flight_search(origin, destination, departure_date, adults=adults)
        if "error" in result or not result.get("flights"):
            return web_search(f"flights from {origin} to {destination} on {departure_date}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return web_search(f"flights from {origin} to {destination} on {departure_date}")

def search_hotels(city: str, checkin: str, checkout: str) -> str:
    """Search for hotels"""
    try:
        # First get hotel list
        hotel_list = hotel_toolkit.hotel_list(city)
        if "error" in hotel_list or not hotel_list.get("hotels"):
            return web_search(f"hotels in {city} from {checkin} to {checkout}")
        
        # Then search with filters
        result = hotel_toolkit.hotel_search(city, checkin, checkout)
        if "error" in result:
            return json.dumps(hotel_list, indent=2)
        return json.dumps(result, indent=2)
    except Exception as e:
        return web_search(f"hotels in {city} from {checkin} to {checkout}")

def search_experiences(city: str) -> str:
    """Search for local experiences"""
    try:
        result = experience_toolkit.experience_search(city)
        if "error" in result or not result.get("experiences"):
            return web_search(f"things to do activities attractions in {city}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return web_search(f"things to do activities attractions in {city}")

# AutoGen configuration
config_list = [{
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("OPENAI_BASE_URL"),
}]

# Create agents
def create_planner():
    return AssistantAgent(
        name="Planner",
        system_message="""You are a Travel Itinerary Specialist. Your role is to create comprehensive travel itineraries through creative discussion.
        
Personality: Creative, enthusiastic, and collaborative. Propose multiple options and engage in brainstorming.
        
Your goal: Create detailed travel itineraries with flights, hotels, activities, and daily schedules.
        
Propose creative options, defend planning decisions, and adapt based on team feedback. Always use tools to find real options.
        
If any tool fails, immediately use web search to find alternative information. Never return empty results.

Tool usage rules:
- Call each tool at most ONCE per response
- Do not repeat the same tool call with the same arguments
- If a tool returns results, use them directly — do not call the same tool again""",
        llm_config={"config_list": config_list, "temperature": 0.8},
        function_map={
            "search_flights": search_flights,
            "search_hotels": search_hotels,
            "search_experiences": search_experiences,
            "get_weather": get_weather,
            "web_search": web_search
        }
    )

def create_optimizer():
    return AssistantAgent(
        name="Optimizer",
        system_message="""You are a Travel Cost Optimizer. Your role is to optimize travel plans through analytical debate and cost-conscious discussion.
        
Personality: Analytical, cost-conscious, and pragmatic. Challenge expensive suggestions and propose alternatives.
        
Your goal: Optimize travel plans for cost, value, and efficiency while maintaining quality.
        
Challenge costly suggestions, debate trade-offs, and build consensus on optimizations. Use web search extensively to find cheaper alternatives and verify pricing.
        
Always propose specific cost-saving alternatives and justify your recommendations. If primary tools fail, immediately use web_search as fallback.
        
Tool usage priority: web_search (primary for cost comparison and alternatives)""",
        llm_config={"config_list": config_list, "temperature": 0.6},
        function_map={
            "web_search": web_search,
            "search_flights": search_flights,
            "search_hotels": search_hotels,
            "get_weather": get_weather
        }
    )

def create_user_proxy():
    return UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False
    )
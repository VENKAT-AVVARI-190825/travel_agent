from crewai import Agent
from crewai.tools import tool, BaseTool
import openai
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field
import os

# Import tools
from toolkits.web_search_service import WebSearchService
from toolkits.weather_tool import WeatherTool
from toolkits.amadeus_hotel_search import AmadeusHotelToolkit
from toolkits.amadeus_flight_tool import AmadeusFlightToolkit
from toolkits.amadeus_experience_tool import AmadeusExperienceToolkit
from toolkits.current_datetime import DateTimeTool
from api.datamodels import TripRequirements, TravelPlan, OptimizationResult

from crewai.llm import LLM

# LLM Configuration
from config import OPENAI_API_KEY, OPENAI_BASE_URL, get_openai_config

client = openai.OpenAI(**get_openai_config())

llm = LLM(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model="openai/gpt-4o-mini"
)

# Input schemas for tools
class SearchWebInput(BaseModel):
    query: str = Field(description="Search query string")

class WeatherInput(BaseModel):
    city: str = Field(description="City name")
    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: str = Field(description="End date (YYYY-MM-DD)")

class HotelInput(BaseModel):
    city: str = Field(description="City name")
    checkin: str = Field(description="Check-in date (YYYY-MM-DD)")
    checkout: str = Field(description="Check-out date (YYYY-MM-DD)")
    adults: int = Field(default=1, description="Number of adults")

class FlightInput(BaseModel):
    origin: str = Field(description="Origin city")
    destination: str = Field(description="Destination city")
    departure_date: str = Field(description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = Field(default=None, description="Return date (YYYY-MM-DD)")

class ExperienceInput(BaseModel):
    city: str = Field(description="City name")

# BaseTool implementations
class SearchWebTool(BaseTool):
    name: str = "search_web"
    description: str = "Search the web for travel information"
    args_schema: Type[BaseModel] = SearchWebInput
    
    def _run(self, query: str) -> str:
        try:
            web_search = WebSearchService()
            return web_search.search(query)
        except:
            return f"Web search results for: {query}"

class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get weather forecast for a city and date range"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, city: str, start_date: str, end_date: str) -> str:
        try:
            weather_tool = WeatherTool()
            return weather_tool.get_weather_range(city, start_date, end_date)
        except:
            try:
                web_search = WebSearchService()
                return web_search.search(f"weather forecast {city} {start_date} {end_date}")
            except:
                return f"Weather forecast for {city} from {start_date} to {end_date}"

class SearchHotelsTool(BaseTool):
    name: str = "search_hotels"
    description: str = "Search for hotels in a city"
    args_schema: Type[BaseModel] = HotelInput
    
    def _run(self, city: str, checkin: str, checkout: str, adults: int = 1) -> str:
        try:
            hotel_toolkit = AmadeusHotelToolkit()
            hotel_ids, hotels = hotel_toolkit.hotel_list(city)
            if hotel_ids:
                return hotel_toolkit.hotel_search(hotel_ids[:5], hotels[:5], checkin, checkout, adults)
            return []
        except:
            try:
                web_search = WebSearchService()
                return web_search.search(f"hotels {city} {checkin} {checkout} {adults} adults")
            except:
                return f"Hotel options in {city} for {adults} adults"

class SearchFlightsTool(BaseTool):
    name: str = "search_flights"
    description: str = "Search for flights between cities"
    args_schema: Type[BaseModel] = FlightInput
    
    def _run(self, origin: str, destination: str, departure_date: str, return_date: str = None) -> str:
        try:
            flight_toolkit = AmadeusFlightToolkit()
            return flight_toolkit.flight_search(origin, destination, departure_date, return_date, adults=1)
        except:
            try:
                web_search = WebSearchService()
                query = f"flights {origin} to {destination} {departure_date}"
                if return_date:
                    query += f" return {return_date}"
                return web_search.search(query)
            except:
                return f"Flight options from {origin} to {destination}"

class SearchExperiencesTool(BaseTool):
    name: str = "search_experiences"
    description: str = "Search for local activities and experiences"
    args_schema: Type[BaseModel] = ExperienceInput
    
    def _run(self, city: str) -> str:
        try:
            experience_toolkit = AmadeusExperienceToolkit()
            return experience_toolkit.experience_search(city)
        except:
            try:
                web_search = WebSearchService()
                return web_search.search(f"activities experiences things to do {city}")
            except:
                return f"Local activities and experiences in {city}"

class GetCurrentDateTool(BaseTool):
    name: str = "get_current_date"
    description: str = "Get today's date"
    
    def _run(self) -> str:
        try:
            datetime_tool = DateTimeTool()
            return datetime_tool.get_current_date()
        except:
            return "2024-01-15"

def info_collector():
    """Agent to extract trip requirements from user input."""
    return Agent(
        role="Travel Requirements Specialist",
        goal="Extract and validate complete travel requirements from user requests and return structured JSON",
        backstory="""You are an experienced travel consultant who specializes in understanding customer needs and gathering comprehensive travel requirements. 
        You always return properly formatted JSON with all required fields. You are excellent at parsing natural language travel requests and extracting specific details like dates, locations, budgets, and traveler counts.""",
        tools=[SearchWebTool(), GetCurrentDateTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def planner():
    """Agent to create a travel plan using real data."""
    return Agent(
        role="Travel Itinerary Specialist",
        goal="Create comprehensive travel itineraries with flights, hotels, and activities",
        backstory="You are a skilled travel planner with extensive knowledge of destinations worldwide and access to the best travel booking systems.",
        tools=[SearchFlightsTool(), SearchHotelsTool(), SearchExperiencesTool(), GetWeatherTool(), SearchWebTool(), GetCurrentDateTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def optimizer():
    """Agent to optimize travel plan for cost and value."""
    return Agent(
        role="Travel Cost Optimizer",
        goal="Optimize travel plans for cost, timing, and customer satisfaction",
        backstory="You are a financial analyst specializing in travel cost optimization and finding the best value propositions for customers.",
        tools=[SearchWebTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

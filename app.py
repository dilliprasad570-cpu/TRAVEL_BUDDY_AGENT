# =====================================================================
# TRAVEL BUDDY AGENT — Main Application File
# =====================================================================
# This file builds an AI-powered travel assistant that can:
#   1. Research any travel destination (attractions, culture, weather, etc.)
#   2. Search for real flights between cities with prices
#   3. Chat with the user through a simple web interface
# =====================================================================


# ----- STEP 1: IMPORT REQUIRED LIBRARIES -----
# Think of "importing" as bringing in pre-built tools from a toolbox.
# Each line below loads a specific capability we need.

from langchain_nvidia_ai_endpoints import ChatNVIDIA  # Connects us to NVIDIA's AI brain (the language model)
from langchain.tools import tool                       # Lets us create custom "tools" the AI agent can use
import os                                              # Lets us access system-level things like environment variables
from langchain.agents import create_agent              # Builds the AI agent that can reason and pick tools
from langchain_tavily import TavilySearch              # A powerful web search engine for destination research
from dotenv import load_dotenv                         # Loads secret API keys from a .env file (keeps them safe)
from langgraph.checkpoint.memory import InMemorySaver  # Gives the agent "memory" so it remembers past messages


# ----- STEP 2: SET UP MEMORY AND LOAD SECRETS -----

checkpointer = InMemorySaver()  # Create a memory store — the agent will remember the conversation history
load_dotenv()  # Read the .env file and load all the secret API keys into the system
               # (API keys are like passwords that let us use external services)

# This config tells the agent which conversation thread to use.
# Think of "thread_id" as a chat room ID — all messages in thread "1" belong to the same conversation.
config = {"configurable": {"thread_id": "1"}}


# ----- STEP 3: CONNECT TO THE AI MODEL (THE "BRAIN") -----
# This sets up the connection to NVIDIA's AI model, which powers the agent's intelligence.
# It's like hiring a very smart assistant and giving them their access badge.

client = ChatNVIDIA(
  model="openai/gpt-oss-120b",          # The specific AI model to use (a 120-billion parameter model)
  api_key=os.getenv("OPEN_AI_API"),      # Fetch the API key from the .env file (the "password" to use the model)
  temperature=1,                          # Controls creativity: 0 = very predictable, 1 = more creative responses
  top_p=1,                                # Another creativity control: 1 = consider all possible word choices
  max_tokens=4096,                        # Maximum length of the AI's response (4096 tokens ≈ ~3000 words)
)


# =====================================================================
# STEP 4: DEFINE THE TOOLS (Skills the agent can use)
# =====================================================================
# Tools are like special abilities we give to the agent.
# The agent reads each tool's description and decides WHEN to use it
# based on what the user asks.
# =====================================================================


# ----- TOOL 1: DESTINATION RESEARCH -----
# This tool searches the internet for travel information about any place.

@tool  # This decorator tells LangChain: "Register this function as a tool the agent can call"
def destination_research(place:str)-> str:
    """"description": "Research travel destinations. Use this tool to retrieve comprehensive, up-to-date information about a specific city, region, or country, including local attractions, culture, weather conditions, safety advisories, and general travel guidelines."""

    # Set up the Tavily search engine with our preferences
    searcher = TavilySearch(
        max_results=5,                              # Return the top 5 most relevant search results
        search_depth="advanced",                    # Use deep search (more thorough than a basic search)
        tavily_api_key=os.getenv("TAVILY_API"),     # Our Tavily API key from the .env file
    )
    result = searcher.invoke(place)  # Actually perform the search for the given place
    # If we got results, convert them to text and return; otherwise say "No results found"
    return str(result) if result else "No results found."


# ----- TOOL 2: FLIGHT SEARCH -----
# This tool searches Google for real flight information between two cities.

@tool  # Register this function as another tool the agent can use
def custom_flight_search_tool(origin: str, destination: str, date: str) -> str:
    """Search for flights between two cities using Serper Google Search.

    Args:
        origin: Departure city or airport code (e.g., 'Delhi' or 'DEL').
        destination: Arrival city or airport code (e.g., 'Goa' or 'GOI').
        date: Travel date in YYYY-MM-DD format (e.g., '2026-12-15').

    Returns:
        Flight search results from Google.
    """
    import requests  # Library to make HTTP requests (like visiting a webpage programmatically)

    # Build a search query string — just like what you'd type into Google
    query = f"flights from {origin} to {destination} on {date} one way price in INR"

    # Send the search query to Serper's API (which searches Google for us)
    response = requests.post(
        "https://google.serper.dev/search",          # The Serper API endpoint (the "address" we send our request to)
        headers={
            "X-API-KEY": os.getenv("SERP_API"),      # Our Serper API key for authentication
            "Content-Type": "application/json",       # Tell the server we're sending JSON data
        },
        json={"q": query, "gl": "in", "hl": "en"},  # q = query, gl = country (India), hl = language (English)
    )

    # If the request was successful (status code 200 = "OK")
    if response.status_code == 200:
        data = response.json()   # Convert the response into a Python dictionary (structured data)
        results = []              # An empty list to collect all the flight info we find

        # Check for answer box (Google's direct answer at the top of search results)
        if "answerBox" in data:
            results.append(f"Answer: {data['answerBox'].get('answer') or data['answerBox'].get('snippet', '')}")

        # Check for knowledge graph (Google's info panel on the right side)
        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]
            results.append(f"{kg.get('title', '')}: {kg.get('description', '')}")

        # Get top organic results (the regular search results — we take the first 5)
        for item in data.get("organic", [])[:5]:
            results.append(f"- {item.get('title', '')}\n  {item.get('snippet', '')}\n  {item.get('link', '')}")

        # Combine all results into one string, separated by blank lines
        return "\n\n".join(results) if results else "No flight information found."
    else:
        # If the API request failed, tell the user what went wrong
        return f"Serper API request failed with status code {response.status_code}."


# =====================================================================
# STEP 5: DEFINE THE SYSTEM PROMPT (Agent's personality & instructions)
# =====================================================================
# The system prompt is like a job description for the AI agent.
# It tells the agent WHO it is, WHAT tools it has, and HOW to respond.

system_prompt="""You are a TravelBuddy assistant that helps travelers plan trips.
You have access to these tools:
- destination_research_tool: Research attractions, culture, and travel tips
- search_flights: Find flight options (use IATA codes: HYD=Hyderabad, GOI=Goa, BOM=Mumbai, DEL=Delhi, BLR=Bangalore)
Help the traveler by researching destinations and finding flights.
Present results in a clean, readable format with clear sections.
Don't use markdown format."""


# =====================================================================
# STEP 6: CREATE THE AGENT (Putting it all together)
# =====================================================================
# This is where we assemble all the pieces:
#   - The AI brain (client)
#   - The tools it can use (destination_research, custom_flight_search_tool)
#   - Its personality/instructions (system_prompt)
#   - Its memory (checkpointer)

agent=create_agent(
    model=client,                                              # The AI model that powers the agent's thinking
    tools=[destination_research,custom_flight_search_tool],    # The list of tools the agent can choose from
    system_prompt=system_prompt,                                # The agent's instructions and personality
    checkpointer=checkpointer                                  # Memory — so the agent remembers earlier messages
)


# =====================================================================
# STEP 7: BUILD THE WEB INTERFACE (What the user sees and interacts with)
# =====================================================================
# Gradio is a library that creates a simple, clean web page with
# an input box and an output box — no HTML/CSS needed!

import gradio as gr  # Import the Gradio library for building the web interface


# This function is called every time a user submits a query.
# It sends the user's question to the agent and returns the agent's answer.
def travel_buddy_agent(query):
    response = agent.invoke({
            "messages": [{"role": "user", "content": query}]},  # Package the user's query as a message
            config=config  # Pass the conversation config (thread ID)
        )
    # The agent returns a list of messages; we grab the LAST one (the agent's reply)
    # and extract just the text content from it
    return response["messages"][-1].content


# --- Set up the Gradio web interface ---
demo = gr.Interface(
    fn=travel_buddy_agent,  # The function to call when user submits a query
    inputs=gr.Textbox(lines=4, placeholder="Ask about a destination...", label="Your Query"),   # The INPUT box (4 lines tall, with hint text)
    outputs=gr.Textbox(lines=15, label="Agent Response"),  # The OUTPUT box (15 lines tall for detailed responses)
    title="Travel_Buddy_Agent",                                              # Title shown at the top of the page
    description="Ask about destination and find related information and flights."  # Subtitle / description text
)

# --- LAUNCH THE APP ---
# This starts a local web server and opens the interface in your browser.
# debug=True means it will show detailed error messages if something goes wrong (helpful during development).
demo.launch(debug=True)

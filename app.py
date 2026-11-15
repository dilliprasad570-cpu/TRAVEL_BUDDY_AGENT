# =====================================================================
# TRAVEL BUDDY AGENT
# =====================================================================

# ----- STEP 1: IMPORT LIBRARIES -----

import os
import requests
import gradio as gr

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver


# ----- STEP 2: LOAD ENVIRONMENT VARIABLES -----

load_dotenv()

checkpointer = InMemorySaver()

config = {
    "configurable": {
        "thread_id": "1"
    }
}


# ----- STEP 3: DEFINE TOOLS -----


# TOOL 1: DESTINATION RESEARCH

@tool
def destination_research(place: str) -> str:
    """Research travel destinations, attractions, culture, weather,
    safety advisories and general travel guidelines."""

    searcher = TavilySearch(
        max_results=5,
        search_depth="advanced",
        tavily_api_key=os.getenv("TAVILY_API")
    )

    result = searcher.invoke(place)

    return str(result) if result else "No results found."


# TOOL 2: FLIGHT SEARCH

@tool
def custom_flight_search_tool(
    origin: str,
    destination: str,
    date: str
) -> str:
    """Search for flights between two cities using Serper Google Search.

    Args:
        origin: Departure city or airport code.
        destination: Arrival city or airport code.
        date: Travel date in YYYY-MM-DD format.

    Returns:
        Flight search results from Google.
    """

    query = (
        f"flights from {origin} to {destination} "
        f"on {date} one way price in INR"
    )

    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": os.getenv("SERP_API"),
            "Content-Type": "application/json"
        },
        json={
            "q": query,
            "gl": "in",
            "hl": "en"
        },
        timeout=30
    )

    if response.status_code == 200:

        data = response.json()
        results = []

        if "answerBox" in data:
            results.append(
                f"Answer: {data['answerBox'].get('answer') or data['answerBox'].get('snippet', '')}"
            )

        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]

            results.append(
                f"{kg.get('title', '')}: {kg.get('description', '')}"
            )

        for item in data.get("organic", [])[:5]:

            results.append(
                f"- {item.get('title', '')}\n"
                f"  {item.get('snippet', '')}\n"
                f"  {item.get('link', '')}"
            )

        return "\n\n".join(results) if results else "No flight information found."

    return f"Serper API request failed: {response.status_code}"


# ----- STEP 4: SYSTEM PROMPT -----

system_prompt = """
You are a TravelBuddy assistant that helps travelers plan trips.

You have access to these tools:

1. destination_research:
   Research attractions, culture, weather and travel tips.

2. custom_flight_search_tool:
   Find flight options using origin, destination and travel date.

Help travelers by researching destinations and finding flights.

Present results in a clean, readable format with clear sections.
Use simple and understandable language.
Do not invent flight prices or availability.
"""


# ----- STEP 5: LAZY INITIALIZATION OF AI AGENT -----

# The agent is NOT initialized when the application starts.
# It will be initialized only when the first user submits a query.

agent = None


def get_agent():

    global agent

    if agent is None:

        print("Initializing NVIDIA AI agent...", flush=True)

        api_key = os.getenv("OPEN_AI_API")

        if not api_key:
            raise ValueError("OPEN_AI_API environment variable is missing.")

        client = ChatNVIDIA(
            model="openai/gpt-oss-120b",
            api_key=api_key,
            temperature=1,
            top_p=1,
            max_completion_tokens=4096
        )

        agent = create_agent(
            model=client,
            tools=[
                destination_research,
                custom_flight_search_tool
            ],
            system_prompt=system_prompt,
            checkpointer=checkpointer
        )

        print("NVIDIA AI agent initialized successfully!", flush=True)

    return agent


# ----- STEP 6: USER QUERY HANDLER -----


def travel_buddy_agent(query):

    try:

        current_agent = get_agent()

        response = current_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            },
            config=config
        )

        return response["messages"][-1].content

    except Exception as e:

        print(f"Agent Error: {str(e)}", flush=True)

        return f"Something went wrong: {str(e)}"


# ----- STEP 7: BUILD GRADIO INTERFACE -----


demo = gr.Interface(
    fn=travel_buddy_agent,

    inputs=gr.Textbox(
        lines=4,
        placeholder="Ask about a destination...",
        label="Your Query"
    ),

    outputs=gr.Textbox(
        lines=15,
        label="Agent Response"
    ),

    title="Travel_Buddy_Agent",

    description="Ask about destinations and find related information and flights."
)


# ----- STEP 8: LAUNCH APPLICATION ON RENDER -----


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    print("=" * 50, flush=True)
    print("Starting Travel Buddy Agent...", flush=True)
    print(f"PORT: {port}", flush=True)
    print("Starting Gradio server...", flush=True)
    print("=" * 50, flush=True)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        debug=False
    )

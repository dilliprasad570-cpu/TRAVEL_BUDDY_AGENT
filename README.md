# 🌍 Travel Buddy Agent

> An AI-powered travel assistant that researches destinations and searches flights in real time — built with LangGraph, NVIDIA AI Endpoints, and Gradio.

---

## 🗺️ Project Roadmap

Below is the end-to-end journey of how this project was built — from idea to working product.

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                        TRAVEL BUDDY AGENT — ROADMAP                        │
 └─────────────────────────────────────────────────────────────────────────────┘

  PHASE 1               PHASE 2               PHASE 3               PHASE 4
  Define the            Build the             Wire the              Launch the
  Problem               Tools                 Agent Brain           Interface
  ─────────             ─────────             ──────────            ──────────
  │                     │                     │                     │
  ├─ What do            ├─ Destination        ├─ LLM backbone       ├─ Gradio UI
  │  travelers need?    │  Research Tool      │  (NVIDIA endpoint)  │
  │                     │  (Tavily Search)    │                     ├─ Text input
  ├─ Real-time info     │                     ├─ System prompt      │  + response
  │  on destinations    ├─ Flight Search      │  engineering        │
  │                     │  Tool               │                     ├─ One-click
  ├─ Flight options     │  (Serper API)       ├─ Tool binding       │  launch
  │  between cities     │                     │                     │
  │                     │                     ├─ Memory / state     │
  └─ A single chat      └─ Wrapped with       │  (InMemorySaver)    └─ debug=True
     interface              @tool decorator    │                        for dev
                                               └─ create_agent()
                                                  orchestration
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         GRADIO UI                            │
│                  (User types a travel query)                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                     LANGGRAPH AGENT                          │
│               (Decides which tool to call)                   │
│                                                              │
│   ┌─────────────────────┐    ┌────────────────────────────┐  │
│   │   System Prompt      │    │   InMemorySaver            │  │
│   │   (Travel expert     │    │   (Conversation memory)    │  │
│   │    persona)          │    │                            │  │
│   └─────────────────────┘    └────────────────────────────┘  │
└──────────┬──────────────────────────────┬────────────────────┘
           │                              │
           ▼                              ▼
┌────────────────────────┐   ┌──────────────────────────────┐
│  🔍 Destination        │   │  ✈️  Flight Search Tool       │
│     Research Tool      │   │                              │
│                        │   │  Serper Google Search API    │
│  Tavily Search API     │   │  → Flights from A to B      │
│  → Attractions         │   │  → Prices in INR            │
│  → Culture & tips      │   │  → One-way results          │
│  → Weather & safety    │   │                              │
└────────────────────────┘   └──────────────────────────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
              ┌───────────────────────┐
              │  NVIDIA AI Endpoint   │
              │  (openai/gpt-oss-120b)│
              │  via ChatNVIDIA       │
              └───────────────────────┘
```

---

## 🧰 Tech Stack

| Layer            | Technology                        | Purpose                                      |
|------------------|-----------------------------------|----------------------------------------------|
| **LLM**          | NVIDIA AI Endpoints (GPT-OSS-120B)| Language understanding & response generation  |
| **Agent Framework** | LangGraph + LangChain          | Tool orchestration, agent loop, memory        |
| **Search – Destinations** | Tavily Search API        | Deep web search for travel information        |
| **Search – Flights**      | Serper Google Search API | Real-time flight search via Google            |
| **UI**           | Gradio                            | Web-based chat interface                      |
| **Config**       | python-dotenv                     | Secure API key management via `.env`          |

---

## 🔧 Tools Deep Dive

### 1. `destination_research` 🔍
| Property     | Detail                                                                 |
|-------------|------------------------------------------------------------------------|
| **Input**    | A place name (city, region, or country)                                |
| **What it does** | Searches the web via Tavily (advanced depth, top 5 results) for attractions, culture, weather, safety, and travel tips |
| **Output**   | Consolidated string of search results                                  |

### 2. `custom_flight_search_tool` ✈️
| Property     | Detail                                                                 |
|-------------|------------------------------------------------------------------------|
| **Input**    | Origin city/code, Destination city/code, Date (YYYY-MM-DD)            |
| **What it does** | Queries Serper Google Search for one-way flights with INR pricing. Extracts answer boxes, knowledge graphs, and top 5 organic results |
| **Output**   | Formatted flight information with titles, snippets, and links          |

**Supported IATA Codes:** `HYD` (Hyderabad) · `GOI` (Goa) · `BOM` (Mumbai) · `DEL` (Delhi) · `BLR` (Bangalore) — and more

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- API keys for: **NVIDIA AI**, **Tavily**, **Serper**

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Travel_Buddy_Agent.git
cd Travel_Buddy_Agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langchain langchain-nvidia-ai-endpoints langchain-tavily langgraph gradio python-dotenv requests
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
OPEN_AI_API=your_nvidia_api_key_here
TAVILY_API=your_tavily_api_key_here
SERP_API=your_serper_api_key_here
```

### 5. Run the app

```bash
python app.py
```

The Gradio interface will launch at **http://localhost:7860** 🚀

---

## 💬 Example Queries

| Query | What Happens |
|-------|-------------|
| *"Tell me about Goa as a travel destination"* | Agent calls `destination_research` → returns attractions, culture, weather, tips |
| *"Find flights from Delhi to Goa on 2026-12-15"* | Agent calls `custom_flight_search_tool` → returns flight prices & links |
| *"I want to visit Manali. Also find flights from Hyderabad to Delhi for 2026-12-20"* | Agent calls **both** tools → combined destination info + flight results |

---

## 📁 Project Structure

```
Travel_Buddy_Agent/
├── app.py              # Main application — agent, tools, and Gradio UI
├── .env                # API keys (not committed to git)
├── .venv/              # Python virtual environment
└── README.md           # You are here
```

---

## 🧠 How the Agent Thinks

```
User Query
    │
    ▼
┌──────────────────────────────────────────┐
│  Agent receives query + system prompt    │
│  "You are a TravelBuddy assistant..."    │
└──────────────────┬───────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Does the user want │
        │  destination info?  │──── YES ──→ Call destination_research()
        └─────────┬───────────┘
                  │ NO
                  ▼
        ┌─────────────────────┐
        │  Does the user want │
        │  flight info?       │──── YES ──→ Call custom_flight_search_tool()
        └─────────┬───────────┘
                  │ NO
                  ▼
        ┌─────────────────────┐
        │  Answer from LLM's  │
        │  general knowledge   │
        └─────────┬───────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  Format & return     │
        │  response to user    │
        └──────────────────────┘
```

---

## 🛣️ Future Roadmap

- [ ] **Hotel Search Tool** — Add real-time hotel/stay options via API
- [ ] **Itinerary Generator** — Auto-build day-by-day travel plans
- [ ] **Budget Estimator** — Calculate total trip cost (flights + stays + activities)
- [ ] **Multi-language Support** — Respond in the user's preferred language
- [ ] **Persistent Memory** — Save conversation history across sessions (database-backed)
- [ ] **Deploy to Cloud** — Host on Hugging Face Spaces / AWS / GCP

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/hotel-search`)
3. Commit your changes (`git commit -m 'Add hotel search tool'`)
4. Push to the branch (`git push origin feature/hotel-search`)
5. Open a Pull Request

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using LangGraph · NVIDIA AI · Tavily · Serper · Gradio
</p>

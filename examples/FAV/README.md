
# 🔍 Agentic Provenance Tracing Demo

## Overview

This project demonstrates **Agentic Provenance Tracing** - a system that tracks and traces the reasoning, decision-making processes, and data sources used by AI agents in multi-agent systems. By logging every thought, action, and data interaction, the system provides complete transparency into how agents arrive at their conclusions, enabling users to understand, verify, and trust AI-generated advice.

The demo focuses on a **financial advisory use case** where multiple specialized agents collaborate to provide investment advice by gathering market data, performing analysis, and generating portfolio recommendations. Users can interact with the system through intuitive web interfaces and see exactly how each piece of advice was derived.

### Key Features

- **Multi-Agent Financial Advisory System**: Specialized agents for data gathering, web search, analysis, and portfolio advice
- **Complete Provenance Tracking**: Every agent thought, action, and data source is logged at each chat turn
- **Interactive Traceability**: Click on any sentence in the response to see supporting evidence and reasoning
- **Claim Verification**: Automatic fact-checking and source attribution using the domynclaimalign library
- **Real-time Market Data**: Integration with Yahoo Finance API and web search for current market information
- **Multiple Interface Options**: Streamlit for development/testing and Flask for demo-ready UI

## Architecture

### Agent System Overview

The system implements a **ReAct (Reasoning + Acting) pattern** with multiple specialized agents orchestrated to provide comprehensive financial advice:

#### Simple Agent System (`main/agent_system.py`)
A streamlined version with three core agents:

1. **InternetAgent**: Fetches real-time financial data from Yahoo Finance API
2. **AnalysisAgent**: Performs financial analysis using LLM reasoning
3. **PortfolioAgent**: Generates investment advice based on analysis
4. **ReActOrchestrator**: Coordinates agent interactions and maintains conversation flow

#### Complex Agent System (`main/agent_system_complex.py`)
An extended system with additional specialized agents:

1. **YHFinanceAPIAgent**: Advanced Yahoo Finance integration with comprehensive financial metrics
2. **WebSearchAgent**: Searches the web for recent news and market sentiment using DuckDuckGo
3. **WebContentScraperAgent**: Extracts and processes content from financial news websites
4. **JSONExtractorAgent**: Processes and extracts relevant information from structured data
5. **SummarizerAgent**: Condenses large amounts of information into digestible summaries
6. **InternetAgent**: Orchestrates data gathering from multiple sources
7. **AnalysisAgent**: Performs deep financial analysis incorporating multiple data sources
8. **PortfolioAgent**: Provides comprehensive investment recommendations
9. **ReActOrchestrator**: Advanced orchestration with error handling and retry logic

### Provenance Tracing System

The tracing system captures:

- **Agent Thoughts**: LLM reasoning at each decision point
- **Actions Taken**: What each agent does and why
- **Data Sources**: All external data retrieved (APIs, web scraping, etc.)
- **Conversation Context**: Full chat history and user interactions
- **Timestamps**: Precise timing of all operations
- **Error Handling**: Failed attempts and recovery strategies

All logs are stored in structured JSONL format in the `agent_logs/` directory for later analysis and tracing.

## Installation and Setup

### Prerequisites

- Python 3.8+
- Access to an OpenAI-compatible LLM endpoint
- Internet connection for real-time data fetching

### Step 1: Environment Setup

```bash
# Clone the repository
git clone git@github.com:kratisaxena-domyn/Agentic_Provenance_Tracing_Demo.git
cd AgentClaimDemo

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install additional NLP models
python -m spacy download en_core_web_sm
```

### Step 3: Install domynclaimalign Library

The system requires the `domynclaimalign` library (in packages folder) for claim verification and provenance tracing:

```bash
# Install from its folder (adjust path as needed)
cd /path/to/domynclaimalign
pip install -e .
cd /path/to/AgentClaimDemo
```

### Step 4: Configure LLM Endpoint

Update the OpenAI client configuration in the agent system files to point to your LLM endpoint:

```python
client = OpenAI(
    base_url="your-llm-endpoint-url",
    api_key="your-api-key"
)
```

## Running the Applications

Demo Interface (`app/flask_app.py`) 
A Flask-based web application with advanced traceability features:

```bash
cd app
python flask_app.py
```

**Features:**
- **Clickable sentence-level tracing**: Click any sentence to see supporting evidence
- Interactive provenance visualization
- Fact verification and source attribution
- Optimized for production use

## 🧪 Running Experiments

To run comprehensive experiments and evaluations, follow these steps:

### 1. Create Experiment Dataset
Generate prompts and evaluation datasets for experiments:
```bash
cd experiments
python create_prompt_dataset.py
```

### 2. Run Experiments
Execute the main experiment pipeline:
```bash
python comprehensive_experiment_runner.py
```

### 3. Calculate and Save Metrics
Run the metrics calculation and save results:
```bash
chmod +x run.sh
./run.sh
```

The experiments will:
- Generate evaluation prompts for Wikipedia topics
- Run fact-checking and claim verification
- Calculate various metrics (coverage, hit@5, hallucination detection, etc.)
- Save results and metrics for analysis

Results will be saved in the `experiment_data/` directory.


## Usage Examples

### Sample Questions to Try

The system works best with specific financial questions about major stocks:

**Investment Decisions:**
- "Should I buy AAPL stock now?"
- "Is MSFT a good investment for long-term growth?"
- "Should I sell my AMZN shares?"
- How did AAPL perform last year in terms of stock price and dividend payouts?
- Is now a good time to invest in AAPL looking at its forward P/E and analyst ratings?

**Market Analysis:**
- "What is the outlook for TSLA in the next quarter?"
- "How did AAPL perform last year?"
- "What are the risks of investing in TSLA?"

**Portfolio Management:**
- "Compare GOOG and AMZN for portfolio diversification."
- "What advice do you have for a beginner investor interested in MSFT?"

**Current Market Data:**
- "What is the current price of GOOG?"
- "Is TSLA undervalued or overvalued?"

### Understanding the Traces

When you ask a question, the system:

1. **Orchestrator** receives your question and plans the approach
2. **Data Agents** gather information from Yahoo Finance and web sources
3. **Analysis Agent** processes the data and provides insights
4. **Portfolio Agent** generates specific investment advice
5. **Tracing System** links every claim in the response to supporting evidence

In the Flask interface, you can click on any sentence in the response to see:
- Which agent generated that claim
- What data sources were used
- The reasoning process behind the statement
- Confidence scores and verification status

## Project Structure

```
AgentClaimDemo/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── agent_logs/                  # Generated agent logs (JSONL format)
│   ├── orchestrator_log.jsonl
│   ├── InternetAgent_log.jsonl
│   ├── AnalysisAgent_log.jsonl
│   ├── PortfolioAgent_log.jsonl
│   ├── WebSearchAgent_log.jsonl
│   ├── YHFinanceAPIAgent_log.jsonl
│   ├── JSONExtractorAgent_log.jsonl
│   └── SummarizerAgent_log.jsonl
├── app/                         # Web applications
│   └── flask_app.py             # Production Flask interface
└── main/                        # Core agent systems
    ├── agent_system_complex.py  # Full 9-agent system
    └── show_traces.py           # Traceability utilities
```

## Log Format and Analysis

### Agent Log Structure

Each agent generates logs in JSONL format with the following structure:

```json
{
  "timestamp": "2025-09-23T10:30:45.123456",
  "turn": 1,
  "agent": "AnalysisAgent",
  "step": "thought|action|error|finalize",
  "thought": "Reasoning text from the agent",
  "data": "Actual data or results"
}
```

## Technical Details

### ReAct Pattern Implementation

The system implements the ReAct (Reasoning + Acting) pattern where agents:

1. **Reason**: Use LLM to understand the situation and plan next steps
2. **Act**: Execute specific actions (API calls, web search, data processing)
3. **Observe**: Process results and update understanding
4. **Repeat**: Continue until the goal is achieved

### Error Handling and Resilience

- **Retry Logic**: Automatic retries for failed API calls or empty responses
- **Graceful Degradation**: System continues with available data if some sources fail
- **Error Logging**: All failures are logged for debugging and analysis
- **Timeout Protection**: Prevents hanging on slow external services

### Supported Stock Tickers

Currently optimized for major stocks: `AAPL`, `GOOG`, `MSFT`, `TSLA`, `AMZN`

The system can be extended to support additional tickers by updating the ticker extraction logic.

## Troubleshooting

### Common Issues

1. **Missing domynclaimalign library**: Ensure the library is properly installed from its source directory
2. **LLM endpoint errors**: Verify the endpoint URL and API key configuration
3. **Web search failures**: Check internet connectivity and DuckDuckGo access
4. **Yahoo Finance API issues**: Some data may be temporarily unavailable


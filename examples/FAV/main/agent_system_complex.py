
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import os
import json
from datetime import datetime
from openai import OpenAI
import yfinance as yf
import time
from icecream import ic
import re
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
from utils.config_loader import get_llm_client_config
llm_config = get_llm_client_config()

# --- SummarizerAgent ---
class SummarizerAgent:
    def __init__(self):
        self.name = "SummarizerAgent"

    def act(self, text, user_query, turn_number, history):
        start_time = time.time()
        # ic(f"[SummarizerAgent] act called with text length: {len(str(text))}")
        prompt = (
            f"Conversation history: {json.dumps(history)}\n"
            f"User query: {user_query}\n"
            f"Summarize the following content in no more than 500 words, focusing on the most relevant information for the user query.\nContent: {text}"
        )
        messages = [
            {"role": "system", "content": "You are an expert summarizer for financial and news data."},
            {"role": "user", "content": prompt}
        ]
        result = client.chat.completions.create(
            model=llm_config["model_name"],
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True}
            }
        )
        summary = result.choices[0].message.content
        # ic(f"[SummarizerAgent] summary: {summary[:200]}...")
        log_turn(turn_number, self.name, "action", "Summarized JSONExtractorAgent output.", summary)
        summarizer_time = time.time() - start_time
        ic(summarizer_time)
        return summary
    
# --- WebContentScraperAgent ---

class WebContentScraperAgent:
    def __init__(self):
        self.name = "WebContentScraperAgent"

    def act(self, links, turn_number, history):
        start_time = time.time()
        # ic(f"[WebContentScraperAgent] act called with links: {links}")
        contents = []
        for link in links:
            try:
                resp = requests.get(link, timeout=10)
                soup = BeautifulSoup(resp.text, 'html.parser')
                # Get visible text only
                text = soup.get_text(separator=' ', strip=True)
                contents.append({'url': link, 'content': text[:2000]})  # Limit to 2000 chars per page
            except Exception as e:
                # ic(f"[WebContentScraperAgent] Exception for {link}: {e}")
                contents.append({'url': link, 'content': f"Error: {e}"})
        # ic(f"[WebContentScraperAgent] scraped contents: {contents}")
        log_turn(turn_number, self.name, "action", "Scraped web page contents.", contents)
        webscraper_time = time.time() - start_time
        ic(webscraper_time)
        return contents


# --- WebSearchAgent using ddgs ---
class WebSearchAgent:
    def __init__(self):
        self.name = "WebSearchAgent"

    def act(self, query, turn_number, history, max_results=10):
        start_time = time.time()
        # ic(f"[WebSearchAgent] act called with query: {query}")
        thought = llm_thought(history, self.name, observation=f"Searching web for: {query}")
        log_turn(turn_number, self.name, "thought", thought, {"query": query})
        ddgs = DDGS()
        results = []
        try:
            for r in ddgs.news(query):
                results.append(r)
                if len(results) >= max_results:
                    break
        except Exception as e:
            # Handle DDGSException or any other error gracefully
            log_turn(turn_number, self.name, "error", f"Web search failed: {e}", {"query": query})
            results = []
        # ic(f"[WebSearchAgent] results: {results}")
        log_turn(turn_number, self.name, "action", f"Web search results for {query}", results)
        websearch_time = time.time() - start_time
        ic(websearch_time)
        return results

class OptimizedJSONExtractorAgent:
    """Optimized version that batches multiple extractions"""
    def __init__(self):
        self.name = "OptimizedJSONExtractorAgent"

    def batch_extract(self, data_dict, user_query, turn_number, history):
        """Extract from multiple data sources in one LLM call"""
        start_time = time.time()
        
        # Combine all data into one prompt
        combined_prompt = f"""
        Conversation history: {json.dumps(history)}
        User query: {user_query}
        
        Extract and summarize the most relevant information from each data source below for the user query.
        Return a JSON object with keys matching the data source names.
        
        Data sources:
        {json.dumps(data_dict, indent=2)}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at extracting and summarizing structured information. Return only a valid JSON object."},
            {"role": "user", "content": combined_prompt}
        ]
        
        result = client.chat.completions.create(
            model=llm_config["model_name"],
            messages=messages,
            max_tokens=4000,  # Increased for batch processing
            temperature=0.1,  # Lower temperature for more focused results
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True}
            }
        )
        
        extracted = result.choices[0].message.content
        batch_extract_time = time.time() - start_time
        ic({"batch_extract_time": batch_extract_time})
        
        try:
            # Try to parse as JSON, fallback to string if it fails
            return json.loads(extracted)
        except:
            return {"combined_summary": extracted}


class JSONExtractorAgent:
    def __init__(self):
        self.name = "JSONExtractorAgent"

    def act(self, json_data, extraction_instructions, turn_number, history):
        start_time = time.time()
        # ic(f"[JSONExtractorAgent] act called with instructions: {extraction_instructions}")
        thought = llm_thought(history, self.name, observation="Extracting info from JSON")
        log_turn(turn_number, self.name, "thought", thought, {"instructions": extraction_instructions})
        messages = [
            {"role": "system", "content": "You are an expert at extracting structured information from JSON objects."},
            {"role": "user", "content": f"Conversation history: {json.dumps(history)}\nExtract the following from this JSON: {extraction_instructions}\nJSON: {json.dumps(json_data)}"}
        ]
        result = client.chat.completions.create(
            model=llm_config["model_name"],
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True}
            }
        )
        extracted = result.choices[0].message.content
        # ic(f"[JSONExtractorAgent] extracted: {extracted}")
        log_turn(turn_number, self.name, "action", "Extracted info from JSON.", extracted)
        jsonextractor_time = time.time() - start_time
        ic(jsonextractor_time)
        return extracted


# --- YHFinanceAPI Agent ---
class YHFinanceAPIAgent:
    def __init__(self):
        self.name = "YHFinanceAPIAgent"
        self.headers = {
            "x-rapidapi-key": llm_config["rapidapi_key"], 
            "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
        }

    async def _fetch_async(self, session, url, params, endpoint_name):
        """Async version of _fetch for parallel API calls"""
        try:
            async with session.get(url, headers=self.headers, params=params) as response:
                return await response.json()
        except Exception as e:
            return None
        
    # def get_news_v1(self, tickers, turn_number, history):
    #     url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/news"
    #     params = {"tickers": tickers, "type": "ALL"}
    #     return self._fetch(url, params, turn_number, history, "news_v1")

    def get_news_v2(self, tickers, turn_number, history):
        url = "https://yahoo-finance15.p.rapidapi.com/api/v2/markets/news"
        params = {"tickers": tickers, "type": "ALL"}
        return self._fetch(url, params, turn_number, history, "news_v2")

    # def get_stock_history_v1(self, symbol, interval, turn_number, history):
    #     url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/history"
    #     params = {"symbol": symbol, "interval": interval, "diffandsplits": "false"}
    #     return self._fetch(url, params, turn_number, history, "stock_history_v1")

    # def get_stock_history_v2(self, symbol, interval, limit, turn_number, history):
    #     url = "https://yahoo-finance15.p.rapidapi.com/api/v2/markets/stock/history"
    #     params = {"symbol": symbol, "interval": interval, "limit": limit}
    #     return self._fetch(url, params, turn_number, history, "stock_history_v2")

    def search(self, search_term, turn_number, history):
        url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/search"
        params = {"search": search_term}
        return self._fetch(url, params, turn_number, history, "search")

    def get_statistics(self, ticker, turn_number, history):
        url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/modules"
        params = {"ticker": ticker, "module": "statistics"}
        return self._fetch(url, params, turn_number, history, "statistics")

    def get_financial_data(self, ticker, turn_number, history):
        url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/modules"
        params = {"ticker": ticker, "module": "financial-data"}
        return self._fetch(url, params, turn_number, history, "financial_data")

    # def get_sec_filings(self, ticker, turn_number, history):
    #     url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/modules"
    #     params = {"ticker": ticker, "module": "sec-filings"}
    #     return self._fetch(url, params, turn_number, history, "sec_filings")

    def _fetch(self, url, params, turn_number, history, endpoint_name):
        start_time = time.time()
        # ic(f"[YHFinanceAPIAgent] Fetching {endpoint_name} with params: {params}")
        thought = llm_thought(history, self.name, observation=f"Endpoint: {endpoint_name}, Params: {params}")
        log_turn(turn_number, self.name, "thought", thought, {"endpoint": endpoint_name, "params": params})
        try:
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()
            # ic(f"[YHFinanceAPIAgent] {endpoint_name} data: {data}")
            log_turn(turn_number, self.name, "action", f"Fetched data from {endpoint_name}", data)
            yhfinanceapi_time = time.time() - start_time
            ic(yhfinanceapi_time)
            return data
        except Exception as e:
            # ic(f"[YHFinanceAPIAgent] Exception in {endpoint_name}: {e}")
            log_turn(turn_number, self.name, "error", f"API request failed: {e}", None)
            yhfinanceapi_time = time.time() - start_time
            ic(yhfinanceapi_time)
            return None

    def fetch_all_endpoints_parallel(self, ticker, turn_number, history):
        """Fetch all endpoints in parallel using asyncio"""
        start_time = time.time()
        
        async def fetch_all():
            async with aiohttp.ClientSession() as session:
                endpoints = [
                    ("news_v2", "https://yahoo-finance15.p.rapidapi.com/api/v2/markets/news", {"tickers": ticker, "type": "ALL"}),
                    ("search", "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/search", {"search": ticker}),
                    ("statistics", "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/modules", {"ticker": ticker, "module": "statistics"}),
                    ("financial_data", "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/modules", {"ticker": ticker, "module": "financial-data"}),
                ]
                
                tasks = [
                    self._fetch_async(session, url, params, endpoint_name)
                    for endpoint_name, url, params in endpoints
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Return as dict with endpoint names as keys
                return {endpoints[i][0]: results[i] for i in range(len(endpoints))}
        
        # Run the async function
        try:
            import nest_asyncio
            nest_asyncio.apply()  # Allow nested event loops
        except ImportError:
            pass
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(fetch_all())
        finally:
            loop.close()
            
        yhfinance_parallel_time = time.time() - start_time
        ic({"YHFinance_parallel_time": yhfinance_parallel_time})
        return results

# Set up the LLM client
client = OpenAI(
    base_url=llm_config["base_url"],
    api_key=llm_config["api_key"]
)

# if the logs already exist from previous runs, remove them and create new ones
LOG_DIR = "../agent_logs"
if os.path.exists(LOG_DIR):
    for file in os.listdir(LOG_DIR):
        file_path = os.path.join(LOG_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
os.makedirs(LOG_DIR, exist_ok=True)

def log_turn(turn_number, agent_name, step, thought, data=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "turn": turn_number,
        "agent": agent_name,
        "step": step,
        "thought": thought,
        "data": data
    }
    log_file = os.path.join(LOG_DIR, f"{agent_name}_log.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def llm_thought(history, agent_role, observation=None):
    messages = [
        {"role": "system", "content": f"You are a {agent_role}. Think step by step and explain your reasoning."},
        {"role": "user", "content": f"History: {json.dumps(history)}. Observation: {observation if observation else ''} What should you think/do next?"}
    ]
    result = client.chat.completions.create(
        model=llm_config["model_name"],
        messages=messages,
        max_tokens=1000,
        temperature=0.3,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True}
        }
    )
    return result.choices[0].message.content

# --- Agent Definitions ---
class InternetAgent:
    def __init__(self):
        self.name = "InternetAgent"

    def act(self, query, turn_number, history):
        start_time = time.time()
        # ic(f"[InternetAgent] act called with query: {query}")
        thought = llm_thought(history, self.name)
        log_turn(turn_number, self.name, "thought", thought)
        ticker = yf.Ticker(query)
        data = ticker.info
        # ic(f"[InternetAgent] data: {data}")
        if not data or 'regularMarketPrice' not in data:
            log_turn(turn_number, self.name, "error", f"No valid data found for {query}", data)
        else:
            log_turn(turn_number, self.name, "action", f"Fetched data for {query}", data)
        internetagent_time = time.time() - start_time
        ic(internetagent_time)
        return data

class AnalysisAgent:
    def __init__(self):
        self.name = "AnalysisAgent"

    def act(self, financial_data, turn_number, history):
        start_time = time.time()
        # ic(f"[AnalysisAgent] act called with data: {financial_data}")
        thought = llm_thought(history, self.name, observation=financial_data)
        log_turn(turn_number, self.name, "thought", thought, financial_data)
        messages = [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": f"Conversation history: {json.dumps(history)}\nAnalyze this financial data: {json.dumps(financial_data)}"}
        ]
        result = client.chat.completions.create(
            model=llm_config["model_name"],
            messages=messages,
            max_tokens=16000,
            temperature=0.3,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True}
            }
        )
        analysis = result.choices[0].message.content
        # ic(f"[AnalysisAgent] analysis: {analysis}")
        log_turn(turn_number, self.name, "action", "LLM analysis complete.", analysis)
        analysisagent_time = time.time() - start_time
        ic(analysisagent_time)
        return analysis

class PortfolioAgent:
    def __init__(self):
        self.name = "PortfolioAgent"

    def act(self, analysis, user_question, turn_number, history):
        start_time = time.time()
        # ic(f"[PortfolioAgent] act called with analysis: {analysis}, user_question: {user_question}")
        thought = llm_thought(history, self.name, observation=analysis)
        log_turn(turn_number, self.name, "thought", thought, {"analysis": analysis, "question": user_question})
        # Explicitly include chat history in the prompt
        messages = [
            {"role": "system", "content": "You are a portfolio advisor."},
            {"role": "user", "content": f"Conversation history: {json.dumps(history)}\nBased on this analysis: {analysis}, answer this question: {user_question}"}
        ]
        result = client.chat.completions.create(
            model=llm_config["model_name"],
            messages=messages,
            max_tokens=16000,
            temperature=0.3,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True}
            }
        )
        advice = result.choices[0].message.content
        # ic(f"[PortfolioAgent] advice: {advice}")
        log_turn(turn_number, self.name, "action", "Portfolio advice generated.", advice)
        portfolioagent_time = time.time() - start_time
        ic(portfolioagent_time)
        return advice

# --- ReAct Orchestrator ---
class ReActOrchestrator:
    def extract_tickers(self, question):
        
        # Regex: 1-5 uppercase letters/numbers (common for tickers)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        # Find tickers in question
        found = set(re.findall(ticker_pattern, question))
        # Search chat history for more tickers
        for msg in reversed(self.chat_history):
            found.update(re.findall(ticker_pattern, msg.get("content", "")))
        # Optionally filter out common English words (to reduce false positives)
        common_words = {"AND", "OR", "THE", "FOR", "WITH", "BY", "ON", "IN", "TO", "OF", "A", "AN", "IS", "IT", "AS", "AT"}
        tickers = [t for t in found if t not in common_words]
        if tickers:
            return tickers
        raise ValueError("No ticker found in user query or chat history.")
    def log(self, step, thought, data=None):
        log_entry = {
            "turn": self.turn_number,
            "step": step,
            "thought": thought,
            "data": data
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    def __init__(self):
        self.agents = {
            "internet": InternetAgent(),
            "analysis": AnalysisAgent(),
            "portfolio": PortfolioAgent(),
            "yhfinance": YHFinanceAPIAgent(),
            "jsonextractor": OptimizedJSONExtractorAgent(), #JSONExtractorAgent(),
            "summarizer": SummarizerAgent(),
            "websearch": WebSearchAgent(),
            "webscraper": WebContentScraperAgent()
        }
        self.log_file = os.path.join(LOG_DIR, "orchestrator_log.jsonl")
        self.chat_history = []
        self.turn_number = 1
    
    def run(self, user_question):
        """Optimized version of run with parallel processing and batched LLM calls"""
        start_time = time.time()
        self.chat_history.append({"role": "user", "content": user_question})
        self.log("start", f"Received user question: {user_question}")
        
        # Extract ticker once
        ticker = None
        try:
            tickers = self.extract_tickers(user_question)
            ticker = tickers[0]
        except Exception as e:
            self.log("error", f"No ticker found: {e}")
        
        # Use ThreadPoolExecutor for parallel execution of different agent types
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            # 1. YHFinance API calls (parallel)
            if ticker:
                yhfinance_future = executor.submit(
                    self.agents["yhfinance"].fetch_all_endpoints_parallel,
                    ticker, self.turn_number, self.chat_history
                )
                futures['yhfinance'] = yhfinance_future
            
            # 2. Internet Agent (yfinance)
            if ticker:
                internet_future = executor.submit(
                    self.agents["internet"].act,
                    ticker, self.turn_number, self.chat_history
                )
                futures['internet'] = internet_future
            
            # 3. Web search (parallel for both user query and ticker)
            websearch_futures = []
            websearch_user_future = executor.submit(
                self.agents["websearch"].act,
                user_question, self.turn_number, self.chat_history
            )
            websearch_futures.append(('user', websearch_user_future))
            
            if ticker:
                websearch_ticker_future = executor.submit(
                    self.agents["websearch"].act,
                    ticker, self.turn_number, self.chat_history
                )
                websearch_futures.append(('ticker', websearch_ticker_future))
            
            # Collect results
            yhfinance_results = futures.get('yhfinance', None)
            yhfinance_data = yhfinance_results.result() if yhfinance_results else {}
            
            internet_results = futures.get('internet', None)
            financial_data = internet_results.result() if internet_results else None
            
            # Collect web search results
            all_web_results = []
            for search_type, future in websearch_futures:
                try:
                    results = future.result()
                    all_web_results.extend(results)
                except Exception as e:
                    self.log("error", f"Web search failed for {search_type}: {e}")
            
            # Web scraping (if needed)
            web_links = [r['href'] for r in all_web_results if 'href' in r][:5]  # Limit to 5 links for speed
            scraped_contents = []
            if web_links:
                webscraper_start_time = time.time()
                scraped_contents = self.agents["webscraper"].act(web_links, self.turn_number, self.chat_history)
                ic({"WebContentScraperAgent_time": time.time() - webscraper_start_time})
        
        # Summarize/truncate each data source before batch extraction
        summarizer = self.agents["summarizer"]
        summarized_data = {}
        # Summarize YHFinance data
        if yhfinance_data:
            try:
                summarized_data["yhfinance"] = summarizer.act(str(yhfinance_data)[:4000], user_question, self.turn_number, self.chat_history)
            except Exception as e:
                summarized_data["yhfinance"] = str(yhfinance_data)[:1000]
        else:
            summarized_data["yhfinance"] = ""

        # Summarize InternetAgent data
        if financial_data:
            try:
                summarized_data["internet"] = summarizer.act(str(financial_data)[:4000], user_question, self.turn_number, self.chat_history)
            except Exception as e:
                summarized_data["internet"] = str(financial_data)[:1000]
        else:
            summarized_data["internet"] = ""

        # Summarize websearch results (as a list)
        if all_web_results:
            try:
                websearch_text = json.dumps(all_web_results[:10])
                summarized_data["websearch"] = summarizer.act(websearch_text[:4000], user_question, self.turn_number, self.chat_history)
            except Exception as e:
                summarized_data["websearch"] = websearch_text[:1000]
        else:
            summarized_data["websearch"] = ""

        # Summarize scraped contents (as a list)
        if scraped_contents:
            try:
                scraped_text = json.dumps(scraped_contents[:3])
                summarized_data["webscraped"] = summarizer.act(scraped_text[:4000], user_question, self.turn_number, self.chat_history)
            except Exception as e:
                summarized_data["webscraped"] = scraped_text[:1000]
        else:
            summarized_data["webscraped"] = ""

        # Batch extraction and summarization
        batch_start_time = time.time()
        extracted_data = self.agents["jsonextractor"].batch_extract(
            summarized_data, user_question, self.turn_number, self.chat_history
        )
        ic({"batch_processing_time": time.time() - batch_start_time})

        # Final analysis and portfolio advice (these are inherently sequential)
        analysisagent_start_time = time.time()
        analysis = self.agents["analysis"].act(extracted_data, self.turn_number, self.chat_history)
        ic({"AnalysisAgent_time": time.time() - analysisagent_start_time})

        portfolioagent_start_time = time.time()
        advice = self.agents["portfolio"].act(analysis, user_question, self.turn_number, self.chat_history)
        ic({"PortfolioAgent_time": time.time() - portfolioagent_start_time})

        self.chat_history.append({"role": "agent", "content": advice})
        self.turn_number += 1

        total_time = time.time() - start_time
        ic({"Orchestrator_optimized_run_time": total_time})

        return advice

def main():
    print("Welcome to the Financial ReAct Agent Chat System!")
    orchestrator = ReActOrchestrator()
    while True:
        user_question = input("You: ")
        if user_question.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        advice = orchestrator.run(user_question)
        print(f"Agent: {advice}\n")

# if __name__ == "__main__":
#     main()

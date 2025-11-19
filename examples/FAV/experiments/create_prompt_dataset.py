import json, os
import random

save_folder = "experiment_data/prompt_datasets/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
# Ticker categories with representative companies
TICKER_CATEGORIES = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META"],
    "Financial": ["JPM", "BAC", "WFC", "USB", "PNC", "BLK"],
    # "Healthcare": ["JNJ", "PFE", "LLY", "UNH", "GILD"],
    # "Consumer Discretionary": ["MCD", "NKE", "SBUX", "DIS", "NFLX"] #, "TGT", "COST", "BKNG"],
    # "Consumer Staples": ["PEP", "KMB", "GIS", "HSY", "CAG"],
    # "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "DVN"],
    # "Industrials": ["HON", "CAT", "UPS", "FDX", "LMT", "RTX"],
    # "Communication Services": ["TMUS", "CHTR", "PARA", "FOX", "LYV", "SNAP"],
    # "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "PSA", "SPG"],
    # "Materials": ["LIN", "APD", "ECL", "SHW", "NEM"],
    # "Utilities": ["NEE", "DUK", "AEP", "XEL", "SRE", "DTE"]
}

# Helper to get category for a ticker
def ticker_to_category(ticker):
    for category, tickers in TICKER_CATEGORIES.items():
        if ticker in tickers:
            return category
    return None

# Core financial query categories separated by numeric / non-numeric queries
QUERY_CATEGORIES = {
    "investment_decision": {
        "numeric": [
            "Should I buy {ticker} stock now based on its P/E ratio and recent earnings?",
            "Is {ticker} a good long-term investment considering its dividend yield and revenue growth?",
            "Should I sell my {ticker} shares given the current market capitalization and profit margins?",
            "Is now a good time to invest in {ticker} looking at its forward P/E and analyst ratings?"
        ],
        "non_numeric": [
            "What risks should I consider before investing in {ticker}?",
            "How does {ticker}'s business model support long-term growth?",
            "What recent news might impact {ticker}'s stock price?"
        ],
        # Additional single-ticker queries substituting for pairwise prompts
        "additional": [
            "How does {ticker}'s recent earnings report impact its stock outlook?",
            "What are the key competitive advantages of {ticker} in its sector?",
            "How is {ticker} positioned for growth in the next fiscal year?"
        ]
    },
    "market_analysis": {
        "numeric": [
            "What is the outlook for {ticker} in the next quarter based on EPS forecasts and revenue estimates?",
            "How did {ticker} perform last year in terms of stock price and dividend payouts?",
            "What are the financial risks of investing in {ticker} based on debt-to-equity ratio and cash flow?",
            "Is {ticker} undervalued or overvalued based on its book value and market price?"
        ],
        "non_numeric": [
            "What market trends are influencing {ticker}'s sector?",
            "How is {ticker} positioned against competitors?",
            "What are analyst opinions on {ticker} outside of raw financial metrics?"
        ],
        "additional": [
            "What regulatory changes might affect {ticker} in the coming year?",
            "What qualitative factors are important for assessing {ticker}'s growth potential?"
        ]
    },
    "portfolio_management": {
        "numeric": [
            "How does including {ticker} influence portfolio expected return and risk?",
            "What is {ticker}'s beta and volatility over the past year?",
            "What Sharpe ratio does {ticker} present compared to its sector average?"
        ],
        "non_numeric": [
            "What advice do you have for a beginner investor interested in {ticker}?",
            "How does {ticker} align with ESG investing principles?",
            "What qualitative factors make {ticker} a good fit for a balanced portfolio?"
        ],
        "additional": [
            "How resilient is {ticker} to macroeconomic downturns?",
            "What diversification benefits does {ticker} provide for growth-focused portfolios?"
        ]
    },
    "current_market": {
        "numeric": [
            "What is the current price of {ticker} and how does it compare to its 52-week high?",
            "What's driving {ticker}'s recent performance using volume and price momentum?",
            "What are analysts saying about {ticker} price targets and earnings revisions?"
        ],
        "non_numeric": [
            "What recent events are affecting {ticker}'s stock price?",
            "How is investor sentiment impacting {ticker} today?",
            "What are the key challenges facing {ticker} this quarter?"
        ],
        "additional": [
            "How does {ticker} perform relative to its sector benchmarks today?",
            "What are the latest management comments or guidance for {ticker}?"
        ]
    }
}

def generate_prompt_dataset(include_numeric=True, include_non_numeric=True):
    prompts = []
    prompt_id = 0
    
    for category, subtypes in QUERY_CATEGORIES.items():
        for prompt_type in ["numeric", "non_numeric", "additional"]:
            if (prompt_type == "numeric" and not include_numeric) or \
               (prompt_type == "non_numeric" and not include_non_numeric):
                continue
            
            templates = subtypes.get(prompt_type, [])
            for template in templates:
                for category_name, tickers in TICKER_CATEGORIES.items():
                    for ticker in tickers:
                        prompts.append({
                            "id": prompt_id,
                            "financial_category": category_name,
                            "query_category": category,
                            "prompt_type": prompt_type,
                            "template": template,
                            "prompt": template.format(ticker=ticker),
                            "tickers": [ticker]
                        })
                        prompt_id += 1

    random.shuffle(prompts)  # Mix prompts for evaluation diversity
    return prompts

if __name__ == "__main__":
    prompts = generate_prompt_dataset(include_numeric=True, include_non_numeric=True)
    with open(os.path.join(save_folder,"evaluation_prompts_with_categories_single_ticker.json"), "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Generated {len(prompts)} single-ticker evaluation prompts with category metadata.")

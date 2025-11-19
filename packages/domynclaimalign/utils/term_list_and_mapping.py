

financial_terms = [
    'ratio', 'pe', 'margin', 'rate', 'cap', 'volume', 'revenue', 
    'profit', 'earnings', 'price', 'value', 'percent', '%',

    # --- accounting & financial statements ---
    'income statement', 'balance sheet', 'cash flow statement', 'statement of changes in equity',
    'revenue', 'cost of goods sold', 'cogs', 'gross profit', 'operating income',
    'ebit', 'ebitda', 'net income', 'net profit margin', 'operating margin', 'gross margin',
    'accruals', 'deferred revenue', 'deferred tax', 'amortization', 'depreciation',
    'accounts receivable', 'ar', 'accounts payable', 'ap', 'inventory', 'working capital',
    'free cash flow', 'fcf', 'operating cash flow', 'ocf', 'investing cash flow', 'financing cash flow',
    'capital expenditure', 'capex', 'operating expenditure', 'opex',

    # --- equities & markets ---
    'initial public offering', 'ipo', 'follow on public offering', 'fpo',
    'market capitalization', 'market cap', 'price to earnings ratio', 'p e ratio', 'pe ratio',
    'price to book ratio', 'p b ratio', 'pb ratio', 'price to sales ratio', 'p s ratio', 'ps ratio',
    'earnings per share', 'eps', 'forward eps', 'trailing eps', 'ttm eps', 'diluted eps',
    'dividend', 'dividend yield', 'dividend payout ratio', 'ex dividend date',
    'stock split', 'reverse split', 'buyback', 'float', 'shares outstanding',

    # --- trading & execution ---
    'bid', 'ask', 'bid ask spread', 'limit order', 'market order', 'stop order',
    'stop limit order', 'fill', 'partial fill', 'slippage', 'liquidity',
    'volume', 'average daily volume', 'adv', 'open', 'close', 'high', 'low',
    'volatility', 'historical volatility', 'hv', 'implied volatility', 'iv',

    # --- portfolio management ---
    'portfolio', 'asset allocation', 'diversification', 'sharpe ratio', 'sortino ratio',
    'treynor ratio', 'jensens alpha', 'beta', 'correlation', 'covariance',
    'drawdown', 'maximum drawdown', 'mdd', 'value at risk', 'var',
    'conditional var', 'cvar', 'risk adjusted return', 'return on investment', 'roi',
    'internal rate of return', 'irr', 'net present value', 'npv',
    'discount rate', 'hurdle rate',

    # --- derivatives ---
    'option', 'call option', 'put option', 'strike price', 'expiration date',
    'itm', 'otm', 'atm', 'delta', 'gamma', 'theta', 'vega', 'rho', 'greeks',
    'futures', 'forwards', 'swaps', 'interest rate swap', 'irs', 'credit default swap', 'cds',
    'notional value', 'margin requirement', 'initial margin', 'maintenance margin',

    # --- banking & credit ---
    'loan to value ratio', 'ltv', 'debt service coverage ratio', 'dscr',
    'non performing asset', 'npa', 'gross npa', 'net npa',
    'credit rating', 'credit score', 'credit spread', 'yield spread',
    'interest rate', 'base rate', 'prime rate', 'libor', 'sofr',
    'amortization schedule', 'mortgage', 'refinancing',

    # --- fixed income ---
    'bond', 'coupon rate', 'yield to maturity', 'ytm', 'yield to call', 'ytc',
    'current yield', 'duration', 'modified duration', 'convexity',
    'treasury bond', 'corporate bond', 'municipal bond', 'muni bond',
    'zero coupon bond', 'high yield bond', 'investment grade bond',

    # --- corporate finance ---
    'capital structure', 'debt to equity ratio', 'd e', 'leverage ratio',
    'interest coverage ratio', 'return on equity', 'roe', 'return on assets', 'roa',
    'return on capital employed', 'roce', 'weighted average cost of capital', 'wacc',
    'cost of equity', 'cost of debt',
    'enterprise value', 'ev', 'ev ebitda', 'ev revenue',
    'book value', 'shareholder equity', 'minority interest', 'goodwill', 'intangible assets',

    # --- financial analysis ---
    'fundamental analysis', 'technical analysis', 'quantitative analysis',
    'discounted cash flow', 'dcf', 'comparable company analysis', 'comps',
    'precedent transactions', 'sensitivity analysis', 'scenario analysis',
    'forecasting', 'guidance', 'consensus estimate',

    # --- risk management ---
    'market risk', 'credit risk', 'liquidity risk', 'operational risk',
    'systemic risk', 'tail risk', 'model risk', 'counterparty risk',
    'hedge', 'hedging strategy', 'stop loss', 'risk parity',

    # --- macroeconomic indicators ---
    'gross domestic product', 'gdp', 'consumer price index', 'cpi',
    'producer price index', 'ppi', 'unemployment rate', 'labor force participation rate',
    'purchasing managers index', 'pmi', 'inflation rate', 'deflation', 'stagflation',
    'fiscal policy', 'monetary policy', 'federal reserve', 'fed', 'european central bank', 'ecb',
    'quantitative easing', 'qe', 'balance of payments', 'current account deficit',

    # --- regulations ---
    'know your customer', 'kyc', 'anti money laundering', 'aml', 'counter terrorist financing', 'ctf',
    'basel iii', 'mifid ii', 'dodd frank act', 'sarbanes oxley act', 'sox',
    'sec filing', '10k', '10q', '8k', 's1 filing', 'proxy statement', 'def 14a',

    # --- alternative investments ---
    'private equity', 'pe', 'venture capital', 'vc', 'hedge fund',
    'real estate investment trust', 'reit', 'commodities', 'crypto assets',
    'blockchain', 'tokenization', 'exchange traded fund', 'etf', 'index fund',
    'mutual fund', 'nav',

    # --- trading strategies ---
    'arbitrage', 'pairs trading', 'market making', 'trend following',
    'mean reversion', 'high frequency trading', 'hft', 'statistical arbitrage', 'statarb',
    'algorithmic trading', 'backtesting', 'paper trading', 'order book',

    # --- financial ratios ---
    'quick ratio', 'current ratio', 'debt ratio', 'cash ratio',
    'asset turnover ratio', 'inventory turnover', 'receivables turnover',
    'profit margin', 'operating margin', 'roic',

    # --- market vocabulary ---
    'market index', 'sp500', 'nasdaq', 'dow jones industrial average', 'djia',
    'bull market', 'bear market', 'market correction', 'recession', 'expansion',
    'benchmark', 'alpha', 'tracking error', 'smart beta'
]


financial_mapping = {
    # ---------------------------
    # Core ratios / valuation ratios
    # ---------------------------
    'pe': [
        'p/e', 'price earnings', 'price_earnings', 'pe_ratio',
        'p e', 'price to earnings', 'price-to-earnings'
    ],
    'pb': [
        'p/b', 'price book', 'price_book', 'pb_ratio',
        'price_to_book', 'price-to-book', 'p b'
    ],
    'ps': [
        'p/s', 'price sales', 'price_sales', 'ps_ratio',
        'price_to_sales', 'price-to-sales', 'p s'
    ],
    'ev_ebitda': [
        'ev ebitda', 'ev/ebitda', 'enterprise_value_ebitda',
        'enterprise value to ebitda', 'enterprise_value_to_ebitda'
    ],
    'ev_sales': [
        'ev_sales', 'ev/sales', 'enterprise_value_sales'
    ],
    'peg': [
        'price earnings growth', 'price_earnings_growth',
        'peg_ratio'
    ],

    # ---------------------------
    # Profitability ratios
    # ---------------------------
    'roe': [
        'return on equity', 'return_on_equity', 'roe_ratio'
    ],
    'roa': [
        'return on assets', 'return_on_assets', 'roa_ratio'
    ],
    'roce': [
        'return on capital employed', 'return_on_capital_employed'
    ],
    'roic': [
        'return on invested capital', 'return_on_invested_capital'
    ],
    'operating_margin': [
        'op_margin', 'operating margin', 'operating_margin_ratio'
    ],
    'gross_margin': [
        'gross profit margin', 'gross_profit_margin'
    ],
    'net_margin': [
        'net profit margin', 'net_profit_margin'
    ],

    # ---------------------------
    # Liquidity ratios
    # ---------------------------
    'current_ratio': [
        'current ratio', 'liquidity ratio'
    ],
    'quick_ratio': [
        'acid test ratio', 'quick ratio'
    ],
    'cash_ratio': [
        'cash ratio', 'cash_reserves_ratio'
    ],

    # ---------------------------
    # Leverage ratios
    # ---------------------------
    'de': [
        'de_ratio', 'debt equity ratio', 'debt_to_equity',
        'debt-equity', 'd/e'
    ],
    'debt_ratio': [
        'total_debt_ratio', 'debt_to_assets'
    ],
    'interest_coverage': [
        'interest coverage ratio', 'icr', 'ebit_interest'
    ],

    # ---------------------------
    # Earnings metrics
    # ---------------------------
    'eps': [
        'earnings per share', 'earnings_per_share',
        'eps diluted', 'diluted_eps'
    ],
    'forward_eps': [
        'fwd_eps', 'projected_eps'
    ],
    'ttm_eps': [
        'trailing_eps', 'eps_ttm', 'trailing_twelve_month_eps'
    ],

    # ---------------------------
    # Revenue / income terms
    # ---------------------------
    'revenue': [
        'sales', 'turnover', 'income', 'top_line'
    ],
    'gross_profit': [
        'gp', 'gross_profit_value'
    ],
    'operating_income': [
        'ebit', 'operating profit', 'operating_profit'
    ],
    'ebitda': [
        'earnings before interest tax depreciation amortization',
        'ebitda_value'
    ],
    'net_income': [
        'bottom_line', 'net_profit'
    ],

    # ---------------------------
    # Cash flow metrics
    # ---------------------------
    'fcf': [
        'free_cash_flow', 'free cash flow'
    ],
    'ocf': [
        'operating_cash_flow', 'operating cash flow'
    ],
    'capex': [
        'capital_expenditure', 'capital expenditure', 'cap ex'
    ],

    # ---------------------------
    # Market terms
    # ---------------------------
    'market_cap': [
        'market capitalization', 'marketcap', 'mcap'
    ],
    'volume': [
        'vol', 'trading_volume', 'trade_volume'
    ],
    'volatility': [
        'vol', 'price_volatility', 'market_volatility'
    ],
    'beta': [
        'systematic_risk', 'beta_value'
    ],

    # ---------------------------
    # Stock-related terms
    # ---------------------------
    'dividend_yield': [
        'div_yield', 'yield', 'dy'
    ],
    'dividend_payout': [
        'payout_ratio', 'dividend_payout_ratio'
    ],
    'buyback': [
        'share_repurchase', 'stock_buyback'
    ],

    # ---------------------------
    # Balance sheet terms
    # ---------------------------
    'assets': [
        'total_assets', 'asset_value'
    ],
    'liabilities': [
        'total_liabilities', 'liability_value'
    ],
    'equity': [
        'shareholder_equity', 'equity_value'
    ],
    'working_capital': [
        'wc', 'net_working_capital'
    ],

    # ---------------------------
    # Cash metrics
    # ---------------------------
    'cash': [
        'cash_on_hand', 'cash_balance'
    ],
    'cash_equivalents': [
        'liquid_assets', 'short_term_investments'
    ],

    # ---------------------------
    # Trading / execution
    # ---------------------------
    'bid': [
        'bid_price'
    ],
    'ask': [
        'ask_price', 'offer_price'
    ],
    'bid_ask_spread': [
        'spread'
    ],
    'market_order': [
        'mo', 'market_order_execution'
    ],
    'limit_order': [
        'lo', 'limit_order_execution'
    ],

    # ---------------------------
    # Macro terms
    # ---------------------------
    'inflation': [
        'cpi', 'consumer_price_index'
    ],
    'gdp': [
        'gross domestic product', 'economic_output'
    ],
    'interest_rate': [
        'base_rate', 'policy_rate'
    ],

    # ---------------------------
    # Risk / volatility
    # ---------------------------
    'var': [
        'value at risk', 'value_at_risk'
    ],
    'cvar': [
        'conditional var', 'conditional_value_at_risk'
    ],
    'drawdown': [
        'dd', 'max_drawdown', 'mdd'
    ],

    # ---------------------------
    # Corporate finance
    # ---------------------------
    'wacc': [
        'weighted average cost of capital', 'cost_of_capital'
    ],
    'npv': [
        'net_present_value'
    ],
    'irr': [
        'internal_rate_of_return'
    ],

    # ---------------------------
    # Regulations / compliance
    # ---------------------------
    'kyc': [
        'know_your_customer'
    ],
    'aml': [
        'anti_money_laundering'
    ],
    'sox': [
        'sarbanes oxley', 'sarbanes_oxley'
    ],
}

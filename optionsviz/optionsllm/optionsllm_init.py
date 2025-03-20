"""
Options Analysis strategies package.

This package provides tools for analyzing stock options, including:
- Retrieving options data from yfinance
- Computing options metrics like max pain and IV skew
- Formatting options data for analysis
- LLM-powered options strategy recommendations
"""

from optionsllm.options_data import (
    get_fundamental_data,
    get_historical_data,
    compute_historical_volatility,
    get_option_expirations,
    fetch_option_chain,
    calculate_greeks,
    compute_put_call_ratio
)

from optionsllm.advanced_options import (
    calculate_max_pain,
    compute_iv_skew
)

from optionsllm.json_packaging import (
    build_compact_options_json
)

from optionsllm.main import (
    summarize_options_data,
    ask_llm_about_options
)

__all__ = [
    'get_fundamental_data',
    'get_historical_data',
    'compute_historical_volatility',
    'get_option_expirations',
    'fetch_option_chain',
    'calculate_greeks',
    'compute_put_call_ratio',
    'calculate_max_pain',
    'compute_iv_skew',
    'build_compact_options_json',
    'summarize_options_data',
    'ask_llm_about_options'
]

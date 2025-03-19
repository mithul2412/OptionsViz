# main.py
"""
Main orchestration module for the Options Analysis Tool.

This module provides:
1. A command-line interface for analyzing stock options
2. Functions to summarize options data in text format
3. Integration with Language Models (LLMs) for AI-powered analysis
   and trading recommendations

The module can be run directly to analyze a stock from the command line
or its functions can be imported for use in the Streamlit dashboard.
"""

import argparse
import json
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Import from other modules
from strategies.json_packaging import build_compact_options_json

# Conditional import for LLM support
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


def summarize_options_data(data: Dict[str, Any]) -> str:
    """
    Create a human-readable text summary of options data.
    
    Converts the JSON options data structure into a formatted text summary
    that can be displayed in a console or used as context for LLM queries.
    
    Args:
        data: Options data dictionary from build_compact_options_json
        
    Returns:
        Multi-line string with a formatted summary of the options data
        
    Example:
        >>> data = build_compact_options_json("AAPL")
        >>> summary = summarize_options_data(data)
        >>> print(summary[:100])  # First 100 chars
        Ticker: AAPL  Date: 2023-10-15
        Price: $150.0   Market Cap: 2500000000   Beta: 1.2
    """
    if "error" in data:
        return data["error"]

    lines = []
    lines.append(f"Ticker: {data['ticker']}  Date: {data['analysis_date']}")
    lines.append(f"Price: ${data['price']}   Market Cap: {data['market_cap']}   Beta: {data['beta']}")
    lines.append(f"52W High: {data['fifty_two_week_high']}  52W Low: {data['fifty_two_week_low']}")
    lines.append(f"Dividend Yield: {data['dividend_yield']}  Next Earnings: {data['next_earnings']}")
    lines.append(f"Put/Call Ratio: {data['put_call_ratio']}")

    # Add historical volatility if available
    if "historical_volatility" in data:
        hv = data["historical_volatility"]
        if hv is not None:
            lines.append(f"Historical Volatility (1y): {hv:.2f}%")

    lines.append("")

    # Add details for each expiration
    for exp, exp_data in data["expirations"].items():
        lines.append(f"EXPIRATION: {exp}  (DTE = {exp_data['days_to_expiry']})")
        lines.append(f"   Max Pain: {exp_data['max_pain_strike']}")
        
        # Add IV skew information if available
        if exp_data["iv_skew_calls"]:
            skew = exp_data["iv_skew_calls"]
            lines.append(f"   IV Skew (Calls) - ITM: {skew['itm']}%, ATM: {skew['atm']}%, OTM: {skew['otm']}%")

        # Add call options details
        lines.append("   CALLS near ATM:")
        for call in exp_data["calls"]:
            lines.append(f"     Strike={call['strike']} ({call['moneyness']}) IV={call['iv']}%, "
                         f"Delta={call['delta']}, Gamma={call['gamma']}, Theta={call['theta']}, Vega={call['vega']}, "
                         f"OI={call['open_interest']}, Vol={call['volume']}")
        
        # Add put options details
        lines.append("   PUTS near ATM:")
        for put in exp_data["puts"]:
            lines.append(f"     Strike={put['strike']} ({put['moneyness']}) IV={put['iv']}%, "
                         f"Delta={put['delta']}, Gamma={put['gamma']}, Theta={put['theta']}, Vega={put['vega']}, "
                         f"OI={put['open_interest']}, Vol={put['volume']}")
        
        lines.append("")

    return "\n".join(lines)


def ask_llm_about_options(
    summary_text: str,
    user_query: str,
    max_tokens: int = 2000
) -> str:
    """
    Use an LLM to analyze options data and answer questions.
    
    This function sends the options data summary along with a user question
    to an LLM (via OpenRouter or OpenAI) and returns the analysis. When the
    analysis includes trading recommendations, it is formatted with a structured
    JSON block for potential automated execution.
    
    Args:
        summary_text: Text summary of options data (from summarize_options_data)
        user_query: User's question or request for analysis
        max_tokens: Maximum tokens in the LLM response (default: 2000)
        
    Returns:
        LLM-generated analysis or error message if LLM is unavailable
        
    Note:
        Requires valid API credentials in environment variables:
        - OPENROUTER_API_KEY or OPENAI_API_KEY
    
    Example:
        >>> data = build_compact_options_json("AAPL")
        >>> summary = summarize_options_data(data)
        >>> analysis = ask_llm_about_options(summary, "What strategy would you recommend?")
    """
    if not LLM_AVAILABLE:
        return "Error: OpenAI library not installed. Please install it with: pip install openai"

    # Create system prompt with instructions
    system_prompt = (
        "You are an expert options strategist. "
        f"Answer the user's question based on the following options data, "
        f"and keep your response under {max_tokens} tokens.\n\n"
        "IMPORTANT: If your response suggests any options trades or trading strategies, you MUST include "
        "a structured JSON block at the end of your message with the following format:\n\n"
        "```json\n"
        "{\n"
        "  \"orders\": [\n"
        "    {\n"
        "      \"symbol\": \"TICKER\",\n"
        "      \"option_type\": \"call\",\n" 
        "      \"direction\": \"buy\",\n"
        "      \"strike\": 180.0,\n"
        "      \"expiration\": \"2023-12-15\",\n"
        "      \"quantity\": 1,\n"
        "      \"reason\": \"Short explanation of this trade\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n\n"
        "Make sure the JSON is valid and properly formatted with the exact fields shown above. "
        "For option_type, use 'call' or 'put'. For direction, use 'buy' or 'sell'. "
        "Use the YYYY-MM-DD format for expiration dates. "
        "The options chain data will tell you the available strikes and expirations - "
        "ONLY use strikes and expirations that are actually available in the data."
    )

    # Format user content with context and question
    user_content = f"OPTIONS DATA:\n\n{summary_text}\n\nQUESTION: {user_query}"

    try:
        # Load environment variables for API keys
        load_dotenv()
        
        # Try OpenRouter first, then OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: No API key found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY."

        # Determine which API to use based on available keys
        if os.getenv("OPENROUTER_API_KEY"):
            # Use OpenRouter API
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            extra_headers = {
                "HTTP-Referer": "https://options-analysis-tool.com",
                "X-Title": "OptionsAnalysisLLM",
            }
            # Use Claude for better JSON capabilities if available
            model = "deepseek/deepseek-r1-zero:free"
        else:
            # Use OpenAI API
            client = OpenAI(api_key=api_key)
            extra_headers = {}
            # Try to use GPT-4 for better JSON capabilities
            model = "gpt-4" if "gpt-4" in client.models.list() else "gpt-3.5-turbo"

        # Make the API call
        completion = client.chat.completions.create(
            extra_headers=extra_headers,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,  # Lower temperature for more deterministic responses
            max_tokens=max_tokens
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"Error making LLM request: {str(e)}"


def main():
    """
    Main function for command-line interface.
    
    Parses command-line arguments, builds options data, and provides
    interactive analysis mode if requested.
    """
    parser = argparse.ArgumentParser(description="Modular Options Analysis with Additional Metrics")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g. AAPL, MSFT)")
    parser.add_argument("--expirations", type=int, default=3, help="Number of expiration dates to include")
    parser.add_argument("--hv", action="store_true", help="Include historical volatility in the data")
    parser.add_argument("--tokens", type=int, default=1000, help="Max tokens for LLM responses")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive LLM query mode")
    args = parser.parse_args()

    # Build the options data
    data = build_compact_options_json(args.ticker, expirations_limit=args.expirations, include_hv=args.hv)
    summary = summarize_options_data(data)

    # Print the options summary
    print("=== OPTIONS SUMMARY ===\n")
    print(summary)

    # Run interactive LLM queries if requested
    if args.interactive and "error" not in data:
        if not LLM_AVAILABLE:
            print("OpenAI library not installed. LLM query mode unavailable.")
            print("Install it with: pip install openai")
            return

        # Start interactive loop
        print("Enter your questions about the above data (type 'quit' to exit):\n")
        while True:
            user_q = input("> ")
            if user_q.lower() in ('quit', 'exit', 'q'):
                break
            if not user_q.strip():
                print("Please enter a question or type 'quit' to exit.")
                continue

            # Get and display LLM response
            print("\nAnalyzing with LLM...")
            ans = ask_llm_about_options(summary, user_q, max_tokens=args.tokens)
            print("\n=== LLM RESPONSE ===\n")
            print(ans)
            print("\n---------------------\n")


if __name__ == "__main__":
    main()

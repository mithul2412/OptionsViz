# json_packaging.py
"""
Package options data into standardized JSON format for analysis and visualization.

This module combines data from multiple sources (fundamental data, option chains,
volatility calculations) into a structured JSON format that's optimized for
analysis and display in the Options Analysis Dashboard.

The main function is build_compact_options_json which orchestrates the data
gathering and formatting process.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import from other modules
from strategies.options_data import (
    get_fundamental_data,
    get_historical_data,
    compute_historical_volatility,
    get_option_expirations,
    fetch_option_chain,
    calculate_greeks,
    compute_put_call_ratio
)
from strategies.advanced_options import calculate_max_pain, compute_iv_skew


def build_compact_options_json(
    ticker_symbol: str,
    expirations_limit: int = 3,
    include_hv: bool = False
) -> Dict[str, Any]:
    """
    Build a comprehensive JSON object with options and stock data.
    
    This function orchestrates the collection and formatting of stock and options data
    into a structured JSON format optimized for analysis. It includes fundamental data,
    options chain information, derived metrics like max pain and IV skew, and
    optionally historical volatility.
    
    Args:
        ticker_symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        expirations_limit: Maximum number of expiration dates to include (default: 3)
        include_hv: Whether to include historical volatility calculation (default: False)
        
    Returns:
        Structured dictionary containing all options and stock data
        
    Example:
        >>> data = build_compact_options_json('AAPL', expirations_limit=2, include_hv=True)
        >>> data['ticker']
        'AAPL'
        >>> list(data['expirations'].keys())[:2]  # First two expiration dates
        ['2023-10-20', '2023-10-27']
    """
    # 1. Fetch fundamental data
    f_data = get_fundamental_data(ticker_symbol)
    current_price = f_data["current_price"]
    if not current_price:
        return {"error": f"Unable to retrieve price for {ticker_symbol}"}

    # 2. Get option expirations
    all_exps = get_option_expirations(ticker_symbol)
    if not all_exps:
        return {"error": f"No options data available for {ticker_symbol}"}
    exps_used = all_exps[:expirations_limit]

    # 3. Calculate overall put/call ratio
    pcr = compute_put_call_ratio(ticker_symbol, exps_used)

    # 4. Build base JSON structure
    data = {
        "ticker": f_data["ticker"],
        "analysis_date": f_data["analysis_date"],
        "price": current_price,
        "market_cap": f_data["market_cap"],
        "beta": f_data["beta"],
        "fifty_two_week_high": f_data["fifty_two_week_high"],
        "fifty_two_week_low": f_data["fifty_two_week_low"],
        "dividend_yield": f_data["dividend_yield"],
        "next_earnings": f_data["next_earnings"],
        "put_call_ratio": pcr,
        "expirations": {}
    }

    # 5. Process each expiration date
    for exp in exps_used:
        # Fetch option chain
        calls_df, puts_df = fetch_option_chain(ticker_symbol, exp)
        if calls_df.empty and puts_df.empty:
            continue  # Skip empty chains

        # Calculate days to expiry
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - datetime.now().date()).days

        # Calculate max pain
        mp_strike = calculate_max_pain(calls_df, puts_df)

        # Calculate IV skew for calls
        skew = compute_iv_skew(calls_df, current_price)

        # Extract near ATM options to keep data size manageable
        calls_compact = _extract_near_atm(calls_df, 'c', current_price, dte)
        puts_compact = _extract_near_atm(puts_df, 'p', current_price, dte)

        # Add to data structure
        data["expirations"][exp] = {
            "days_to_expiry": dte,
            "max_pain_strike": mp_strike,
            "iv_skew_calls": skew,
            "calls": calls_compact,
            "puts": puts_compact
        }

    # 6. Add historical volatility if requested
    if include_hv:
        hist_df = get_historical_data(ticker_symbol, period="1y")
        hv = compute_historical_volatility(hist_df)
        if hv is not None:
            data["historical_volatility"] = round(hv * 100, 2)  # as a percentage
        else:
            data["historical_volatility"] = None

    return data


def _extract_near_atm(
    df: pd.DataFrame,
    option_type: str,
    current_price: float,
    dte: int
) -> List[Dict[str, Any]]:
    """
    Extract a subset of options closest to at-the-money (ATM).
    
    This helper function selects a small number of strikes near the current price
    and formats them for the JSON output, including calculating Greeks and
    determining moneyness (ITM/ATM/OTM).
    
    Args:
        df: DataFrame containing options data
        option_type: Option type ('c' for call, 'p' for put)
        current_price: Current price of the underlying
        dte: Days to expiry
        
    Returns:
        List of dictionaries with formatted option data for each selected strike
        
    Note:
        This is an internal helper function not intended for direct use.
    """
    if df.empty:
        return []

    # Find the ATM strike
    df["distance"] = (df["strike"] - current_price).abs()
    atm_idx = df["distance"].idxmin()

    # If no valid ATM index found, return empty list
    if pd.isna(atm_idx):
        return []

    # Sort by strike price
    df_sorted = df.sort_values("strike")
    
    # Get the ATM strike
    atm_strike = df.loc[atm_idx, "strike"]
    
    # Get rows below ATM (for ITM calls or OTM puts)
    below = df_sorted[df_sorted["strike"] < atm_strike].iloc[::-1]  # descending
    
    # Get rows at or above ATM (for ATM and OTM calls or ITM puts)
    above = df_sorted[df_sorted["strike"] >= atm_strike]

    # Create a subset with 2 strikes below and 3 at/above ATM
    subset = pd.concat([
        below.head(2).iloc[::-1],  # re-sort ascending
        above.head(3)  # ATM plus next 2
    ]).drop_duplicates()

    # Format the results
    results = []
    for _, row in subset.iterrows():
        s = float(row["strike"])
        iv = float(row["impliedVolatility"])
        
        # Calculate Greeks
        delta, gamma, theta, vega = calculate_greeks(option_type, current_price, s, dte, iv)

        # Determine moneyness (ITM/ATM/OTM)
        if option_type == 'c':
            # For calls
            if s < current_price * 0.98:
                m = "ITM"  # In-the-money
            elif s > current_price * 1.02:
                m = "OTM"  # Out-of-the-money
            else:
                m = "ATM"  # At-the-money
        else:
            # For puts
            if s > current_price * 1.02:
                m = "ITM"  # In-the-money
            elif s < current_price * 0.98:
                m = "OTM"  # Out-of-the-money
            else:
                m = "ATM"  # At-the-money

        # Format and add the option data
        results.append({
            "strike": s,
            "moneyness": m,
            "last": round(float(row["lastPrice"]), 2),
            "bid": round(float(row["bid"]), 2),
            "ask": round(float(row["ask"]), 2),
            "iv": round(iv * 100, 2),
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "open_interest": int(row["openInterest"]),
            "volume": int(row["volume"])
        })

    return results

# json_packaging.py
"""
Package options data into standardized JSON format for analysis and visualization.

This module combines data from multiple sources (fundamental data, option chains,
volatility calculations) into a structured JSON format that's optimized for
analysis and display in the Options Analysis Dashboard.

The main function is build_compact_options_json which orchestrates the data
gathering and formatting process.
"""

from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

# Import from other modules
from options_data import (
    get_fundamental_data,
    get_historical_data,
    compute_historical_volatility,
    get_option_expirations,
    fetch_option_chain,
    calculate_greeks,
    compute_put_call_ratio
)
from advanced_options import calculate_max_pain, compute_iv_skew


def build_compact_options_json(
    ticker_symbol: str,
    expirations_limit: int = 3,
    include_hv: bool = False
) -> Dict[str, Any]:
    """
    Build a comprehensive JSON object with options and stock data.

    This function orchestrates the collection and formatting of stock and options
    data into a structured JSON format optimized for analysis. It includes
    fundamental data, options chain information, derived metrics like max pain
    and IV skew, and optionally historical volatility.

    Args:
        ticker_symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        expirations_limit: Maximum number of expiration dates to include
        include_hv: Whether to include historical volatility calculation

    Returns:
        Structured dictionary containing all options and stock data
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

    # 3. Calculate overall put/call ratio and prepare data structure
    data = _prepare_base_data(ticker_symbol, f_data, all_exps[:expirations_limit])

    # 4. Process each expiration date
    _process_expirations(data, ticker_symbol, all_exps[:expirations_limit], current_price)

    # 5. Add historical volatility if requested
    if include_hv:
        _add_historical_volatility(data, ticker_symbol)

    return data


def _prepare_base_data(ticker_symbol: str, f_data: Dict[str, Any],
                        exps_used: List[str]) -> Dict[str, Any]:
    """Helper function to prepare the base JSON structure."""
    return {
        "ticker": f_data["ticker"],
        "analysis_date": f_data["analysis_date"],
        "price": f_data["current_price"],
        "market_cap": f_data["market_cap"],
        "beta": f_data["beta"],
        "fifty_two_week_high": f_data["fifty_two_week_high"],
        "fifty_two_week_low": f_data["fifty_two_week_low"],
        "dividend_yield": f_data["dividend_yield"],
        "next_earnings": f_data["next_earnings"],
        "put_call_ratio": compute_put_call_ratio(ticker_symbol, exps_used),
        "expirations": {}
    }


def _process_expirations(data: Dict[str, Any], ticker_symbol: str,
                         exps_used: List[str], current_price: float) -> None:
    """Process each expiration date and add to data structure."""
    for exp in exps_used:
        # Fetch option chain
        calls_df, puts_df = fetch_option_chain(ticker_symbol, exp)
        if calls_df.empty and puts_df.empty:
            continue  # Skip empty chains

        # Calculate days to expiry
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - datetime.now().date()).days

        # Add to data structure
        data["expirations"][exp] = {
            "days_to_expiry": dte,
            "max_pain_strike": calculate_max_pain(calls_df, puts_df),
            "iv_skew_calls": compute_iv_skew(calls_df, current_price),
            "calls": _extract_near_atm(calls_df, 'c', current_price, dte),
            "puts": _extract_near_atm(puts_df, 'p', current_price, dte)
        }


def _add_historical_volatility(data: Dict[str, Any], ticker_symbol: str) -> None:
    """Add historical volatility data if requested."""
    hist_df = get_historical_data(ticker_symbol, period="1y")
    hv = compute_historical_volatility(hist_df)
    data["historical_volatility"] = round(hv * 100, 2) if hv is not None else None


def _extract_near_atm(
    df: pd.DataFrame,
    option_type: str,
    current_price: float,
    dte: int
) -> List[Dict[str, Any]]:
    """
    Extract a subset of options closest to at-the-money (ATM).

    This helper function selects a small number of strikes near current price
    and formats them for the JSON output, including calculating Greeks and
    determining moneyness (ITM/ATM/OTM).

    Args:
        df: DataFrame containing options data
        option_type: Option type ('c' for call, 'p' for put)
        current_price: Current price of the underlying
        dte: Days to expiry

    Returns:
        List of dictionaries with formatted option data for each strike

    Note:
        This is an internal helper function not intended for direct use.
    """
    if df.empty:
        return []

    # Get ATM subset
    subset = _get_atm_subset(df, current_price)
    if subset.empty:
        return []

    # Format the results
    return [_format_option_data(row, option_type, current_price, dte)
            for _, row in subset.iterrows()]


def _get_atm_subset(df: pd.DataFrame, current_price: float) -> pd.DataFrame:
    """Get subset of dataframe with strikes near the money."""
    # Find the ATM strike
    df["distance"] = (df["strike"] - current_price).abs()
    atm_idx = df["distance"].idxmin()

    # If no valid ATM index found, return empty DataFrame
    if pd.isna(atm_idx):
        return pd.DataFrame()

    # Sort by strike price
    df_sorted = df.sort_values("strike")

    # Get the ATM strike
    atm_strike = df.loc[atm_idx, "strike"]

    # Get rows below ATM (for ITM calls or OTM puts)
    below = df_sorted[df_sorted["strike"] < atm_strike].iloc[::-1]

    # Get rows at or above ATM (for ATM and OTM calls or ITM puts)
    above = df_sorted[df_sorted["strike"] >= atm_strike]

    # Create a subset with 2 strikes below and 3 at/above ATM
    return pd.concat([
        below.head(2).iloc[::-1],  # re-sort ascending
        above.head(3)  # ATM plus next 2
    ]).drop_duplicates()


def _format_option_data(row: pd.Series, option_type: str,
                        current_price: float, dte: int) -> Dict[str, Any]:
    """Format option data for a single strike."""
    s = float(row["strike"])
    iv = float(row["impliedVolatility"])

    # Calculate Greeks
    delta, gamma, theta, vega = calculate_greeks(
        option_type, current_price, s, dte, iv)

    # Determine moneyness (ITM/ATM/OTM)
    m = _determine_moneyness(option_type, s, current_price)

    # Format and return the option data
    return {
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
    }

def _determine_moneyness(option_type: str, strike: float,
                         current_price: float) -> str:
    """Determine if the option is ITM, ATM, or OTM."""
    if option_type == 'c':
        # For calls
        if strike < current_price * 0.98:
            return "ITM"  # In-the-money
        if strike > current_price * 1.02:
            return "OTM"  # Out-of-the-money
        return "ATM"  # At-the-money

    # For puts
    if strike > current_price * 1.02:
        return "ITM"  # In-the-money
    if strike < current_price * 0.98:
        return "OTM"  # Out-of-the-money
    return "ATM"  # At-the-money

# advanced_options.py
"""
Advanced options analysis calculations module.

This module handles specialized calculations for options analysis including:
- Max pain calculation (point of maximum financial pain for option buyers)
- Implied volatility skew analysis across strikes
- Other advanced metrics used by options traders

Functions in this module expect pandas DataFrames with standard yfinance
option chain structure.
"""

import pandas as pd


def calculate_max_pain(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> float:
    """
    Calculate the max pain point for option contracts.

    Max pain is the strike price where option buyers (collectively) experience
    the most financial pain (loss) at expiration. It's calculated by determining
    the strike price where the total value of all out-of-the-money options
    is minimized.

    Args:
        calls_df: DataFrame containing call options data with 'strike' and
        'openInterest' columns
        puts_df: DataFrame containing put options data with 'strike' and
        'openInterest' columns

    Returns:
        float: The max pain strike price, or None if calculation fails

    Example:
        >>> calls_df = pd.DataFrame({'strike': [100, 105, 110],
        'openInterest': [500, 300, 200]})
        >>> puts_df = pd.DataFrame({'strike': [90, 95, 100],
        'openInterest': [200, 400, 300]})
        >>> calculate_max_pain(calls_df, puts_df)
        100.0
    """
    if calls_df.empty or puts_df.empty:
        return None

    all_strikes = sorted(set(calls_df['strike']) | set(puts_df['strike']))
    results = []

    for strike in all_strikes:
        # calls in-the-money if strike < underlying
        # for a naive approach, sum openInterest for calls with strike < this strike
        # Similarly for puts with strike > this strike
        # This is not the exact net payoff, but a approximation for demonstration
        in_money_calls = calls_df[calls_df['strike'] < strike]['openInterest'].sum()
        in_money_puts = puts_df[puts_df['strike'] > strike]['openInterest'].sum()

        total_pain = in_money_calls + in_money_puts
        results.append((strike, total_pain))

    if not results:
        return None

    results.sort(key=lambda x: x[1])  # sort by total_pain ascending
    return results[0][0]  # the strike with the minimal combined "pain"


def compute_iv_skew(calls_df: pd.DataFrame, current_price: float) -> dict:
    """
    Compute implied volatility skew for calls (ITM, ATM, OTM).

    Implied volatility often varies across different strike prices,
    creating a "skew" or "smile" pattern. This function identifies
    IV at three key points relative to the current price.

    Args:
        calls_df: DataFrame containing call options with 'strike' and
        'impliedVolatility' columns
        current_price: Current price of the underlying asset

    Returns:
        dict: Dictionary with keys 'itm', 'atm', 'otm' and corresponding
        IV values as percentages.
              Returns empty dict if calls_df is empty.

    Example:
        >>> calls_df = pd.DataFrame({
        ...     'strike': [95, 100, 105],
        ...     'impliedVolatility': [0.35, 0.30, 0.32]
        ... })
        >>> compute_iv_skew(calls_df, 100.0)
        {'itm': 35.0, 'atm': 30.0, 'otm': 32.0}
    """
    if calls_df.empty:
        return {}

    sorted_calls = calls_df.sort_values('strike')

    # Find ATM option (closest strike to current price)
    atm_idx = (sorted_calls['strike'] - current_price).abs().idxmin()
    atm_iv = round(sorted_calls.loc[atm_idx, 'impliedVolatility'] * 100, 2)

    # ITM: calls with strike < current_price
    itm_calls = sorted_calls[sorted_calls['strike'] < current_price]
    itm_iv = round(itm_calls
                   ['impliedVolatility'].iloc[-1]
                     * 100, 2) if not itm_calls.empty else None

    # OTM: calls with strike > current_price
    otm_calls = sorted_calls[sorted_calls['strike'] > current_price]
    otm_iv = round(otm_calls
                   ['impliedVolatility'].iloc[0]
                     * 100, 2) if not otm_calls.empty else None

    return {"itm": itm_iv, "atm": atm_iv, "otm": otm_iv}

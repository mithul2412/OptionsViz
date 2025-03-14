# options_data.py
"""
Core module for retrieving and processing stock and options data.

This module provides functions to retrieve fundamental data, historical price data,
and options chain information using the yfinance library. It also includes utility
functions for common options-related calculations.

The module is organized into:
- Fundamental data functions
- Historical data functions
- Options chain functions
- Options metrics calculations
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

# Constants
RISK_FREE_RATE = 0.05  # Standard risk-free rate for options calculations
TRADING_DAYS_PER_YEAR = 252  # Standard number of trading days in a year

# =====================================================================
# FUNDAMENTAL DATA FUNCTIONS
# =====================================================================

def get_fundamental_data(ticker_symbol: str) -> Dict[str, Any]:
    """
    Retrieve fundamental stock data for the given ticker.
    
    Fetches key financial metrics including current price, market cap,
    beta, 52-week high/low, dividend yield, and upcoming earnings date.
    Uses fallback mechanisms if primary data retrieval fails.
    
    Args:
        ticker_symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        Dictionary containing fundamental stock data
        
    Example:
        >>> data = get_fundamental_data('AAPL')
        >>> data['current_price']
        150.25
    """
    ticker = yf.Ticker(ticker_symbol)
    today = datetime.now()

    # Set default/fallback values
    current_price = 0
    market_cap = 0
    beta = 'N/A'
    next_earnings = 'Unknown'
    fifty_two_week_high = 'N/A'
    fifty_two_week_low = 'N/A'
    dividend_yield = 'None'

    try:
        stock_info = ticker.info
        current_price = stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 0))
        market_cap = stock_info.get('marketCap', 0)
        beta = stock_info.get('beta', 'N/A')
        fifty_two_week_high = stock_info.get('fiftyTwoWeekHigh', 'N/A')
        fifty_two_week_low = stock_info.get('fiftyTwoWeekLow', 'N/A')

        dy_raw = stock_info.get('dividendYield', 0)
        if dy_raw:
            dividend_yield = f"{round(dy_raw * 100, 2)}%"
        else:
            dividend_yield = 'None'

        # Attempt to find next earnings
        try:
            earnings_dates = ticker.earnings_dates
            if not earnings_dates.empty:
                next_earnings = earnings_dates.index[0].strftime('%Y-%m-%d')
        except (AttributeError, IndexError):
            # Handle cases where earnings dates are not available
            pass
    except Exception:
        # Fallback if .info is incomplete or fails
        hist = ticker.history(period='1d')
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]

    return {
        "ticker": ticker_symbol.upper(),
        "analysis_date": today.strftime('%Y-%m-%d'),
        "current_price": current_price,
        "market_cap": market_cap,
        "beta": beta,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_low": fifty_two_week_low,
        "dividend_yield": dividend_yield,
        "next_earnings": next_earnings
    }


def get_historical_data(ticker_symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Retrieve historical price data for the specified ticker.
    
    Fetches OHLCV (Open, High, Low, Close, Volume) data for the given
    time period using yfinance.
    
    Args:
        ticker_symbol: Stock ticker symbol
        period: Time period for historical data (e.g., '1d', '1mo', '1y')
        
    Returns:
        DataFrame with historical price data or empty DataFrame if retrieval fails
        
    Example:
        >>> hist_data = get_historical_data('AAPL', '3mo')
        >>> hist_data.head()
                   Open   High    Low  Close    Volume  Dividends  Stock Splits
        Date                                                                  
        2023-01-03  130.28  130.90  124.17  125.07  112117500        0.0           0.0
    """
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period)
    if hist.empty:
        return pd.DataFrame()
    return hist


def compute_historical_volatility(hist_df: pd.DataFrame) -> Optional[float]:
    """
    Calculate annualized historical volatility from daily price data.
    
    Computes the standard deviation of log returns and annualizes
    by multiplying by the square root of trading days per year.
    
    Args:
        hist_df: DataFrame containing historical price data with 'Close' column
        
    Returns:
        Annualized historical volatility as a decimal (e.g., 0.25 for 25%),
        or None if calculation fails
        
    Example:
        >>> hist_data = get_historical_data('SPY', '1y')
        >>> volatility = compute_historical_volatility(hist_data)
        >>> print(f"{volatility:.2%}")
        '15.42%'
    """
    if hist_df.empty:
        return None

    # Calculate daily log returns
    hist_df['log_ret'] = (hist_df['Close'] / hist_df['Close'].shift(1)).apply(
        lambda x: 0 if x <= 0 else np.log(x)
    )
    
    # Calculate standard deviation and annualize
    stdev = hist_df['log_ret'].std(skipna=True)
    annual_hv = stdev * np.sqrt(TRADING_DAYS_PER_YEAR)
    return annual_hv

# =====================================================================
# OPTIONS CHAIN FUNCTIONS
# =====================================================================

def get_option_expirations(ticker_symbol: str) -> List[str]:
    """
    Get all available option expiration dates for a ticker.
    
    Args:
        ticker_symbol: Stock ticker symbol
        
    Returns:
        List of expiration dates in YYYY-MM-DD format, sorted chronologically
        
    Example:
        >>> expirations = get_option_expirations('AAPL')
        >>> expirations[:3]  # First three expiration dates
        ['2023-10-20', '2023-10-27', '2023-11-03']
    """
    ticker = yf.Ticker(ticker_symbol)
    if not ticker.options:
        return []
    
    # Convert to datetime objects for sorting, then back to strings
    raw_exps = ticker.options
    dt_exps = [datetime.strptime(e, "%Y-%m-%d") for e in raw_exps]
    dt_exps.sort()
    return [d.strftime("%Y-%m-%d") for d in dt_exps]


def fetch_option_chain(ticker_symbol: str, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch the full option chain for a specific expiration date.
    
    Args:
        ticker_symbol: Stock ticker symbol
        expiration: Option expiration date in YYYY-MM-DD format
        
    Returns:
        Tuple of (calls_dataframe, puts_dataframe)
        Returns empty DataFrames if fetching fails
        
    Example:
        >>> calls_df, puts_df = fetch_option_chain('AAPL', '2023-12-15')
        >>> calls_df.columns
        Index(['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask',
               'change', 'percentChange', 'volume', 'openInterest', 'impliedVolatility',
               'inTheMoney', 'contractSize', 'currency'],
              dtype='object')
    """
    ticker = yf.Ticker(ticker_symbol)
    try:
        chain = ticker.option_chain(expiration)
        return chain.calls, chain.puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def calculate_greeks(
    option_type: str,
    underlying_price: float,
    strike: float,
    days_to_expiry: int,
    implied_vol: float
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Calculate option Greeks using the Black-Scholes model.
    
    Computes Delta, Gamma, Theta, and Vega for an option using py_vollib
    if available. If py_vollib is not installed, returns None for all Greeks.
    
    Args:
        option_type: Option type, 'c' for call or 'p' for put
        underlying_price: Current price of the underlying asset
        strike: Strike price of the option
        days_to_expiry: Number of days until expiration
        implied_vol: Implied volatility as a decimal
        
    Returns:
        Tuple of (delta, gamma, theta, vega), with None values if calculation fails
        
    Example:
        >>> delta, gamma, theta, vega = calculate_greeks('c', 150.0, 155.0, 30, 0.25)
        >>> delta
        0.424
    """
    # Check if py_vollib is available
    try:
        from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
    except ImportError:
        return None, None, None, None

    # Convert days to years (minimum of 0.1% of a year to avoid division issues)
    time_to_expiry = max(days_to_expiry / 365.0, 0.001)
    flag = option_type.lower()[0]  # 'c' or 'p'

    try:
        # Calculate the Greeks
        d = delta(flag, underlying_price, strike, time_to_expiry, RISK_FREE_RATE, implied_vol)
        g = gamma(flag, underlying_price, strike, time_to_expiry, RISK_FREE_RATE, implied_vol)
        
        # Theta is daily, so divide by 365
        t = theta(flag, underlying_price, strike, time_to_expiry, RISK_FREE_RATE, implied_vol) / 365.0
        
        # Vega is traditionally for 1% volatility change, so divide by 100
        v = vega(flag, underlying_price, strike, time_to_expiry, RISK_FREE_RATE, implied_vol) / 100.0
        
        return round(d, 3), round(g, 4), round(t, 3), round(v, 3)
    except Exception:
        return None, None, None, None


def compute_put_call_ratio(ticker_symbol: str, expirations: List[str]) -> float:
    """
    Calculate the put/call ratio based on open interest.
    
    The put/call ratio is the sum of put open interest divided by
    the sum of call open interest across all specified expirations.
    
    Args:
        ticker_symbol: Stock ticker symbol
        expirations: List of expiration dates to include in the calculation
        
    Returns:
        Put/call ratio as a float, or 0.0 if there is no call open interest
        
    Example:
        >>> expirations = get_option_expirations('SPY')[:3]  # First three expirations
        >>> compute_put_call_ratio('SPY', expirations)
        1.25  # Indicates more put open interest than call open interest
    """
    ticker = yf.Ticker(ticker_symbol)
    total_call_oi = 0
    total_put_oi = 0
    
    for expiration in expirations:
        try:
            chain = ticker.option_chain(expiration)
            total_call_oi += chain.calls['openInterest'].sum()
            total_put_oi += chain.puts['openInterest'].sum()
        except Exception:
            # Skip this expiration if there's an error
            continue

    if total_call_oi == 0:
        return 0.0
    
    return round(total_put_oi / total_call_oi, 2)

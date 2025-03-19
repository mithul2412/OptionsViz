"""
Options Strategy Visualization Wrapper Module

This module wraps the functionality from options_viz.py to be used in other Streamlit applications
without causing conflicts with Streamlit's page configuration.

Author:
    Julian Sanders 

Created:
    March 2025

License:
    MIT
"""

import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import streamlit as st

@st.cache_data()
def get_stock_data(stock_ticker, period='1y'):
    """
    Fetch historical stock data for a given ticker and period.

    Args:
        stock_ticker (str): The stock ticker symbol.
        period (str, optional): The historical period. Defaults to '1y'.

    Returns:
        pandas.DataFrame: Historical stock data.
    
    Raises:
        TypeError: If stock_ticker is not a string
        ValueError: If period is invalid
    """
    if not isinstance(stock_ticker, str):
        raise TypeError("stock_ticker must be a string")
        
    if period not in ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']:
        raise ValueError("period must be one of: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'")
        
    stock = yf.Ticker(stock_ticker)
    return stock.history(period=period)

@st.cache_data()
def get_option_data(stock_ticker):
    """
    Retrieve option chain data for a given ticker.

    Args:
        stock_ticker (str): The stock ticker symbol.

    Returns:
        tuple: (atm_call_strike, call_premium, atm_put_strike, put_premium) 
        or (None, None, None, None)
    
    Raises:
        TypeError: If stock_ticker is not a string
    """
    if not isinstance(stock_ticker, str):
        raise TypeError("stock_ticker must be a string")
        
    stock = yf.Ticker(stock_ticker)
    options = stock.options
    if not options:
        return None, None, None, None

    expiry = options[0]  # Get nearest expiration date
    opt_chain = stock.option_chain(expiry)
    calls, puts = opt_chain.calls, opt_chain.puts

    if calls.empty or puts.empty:
        return None, None, None, None

    current_price = stock.history(period='1d')['Close'].iloc[-1]
    atm_call = calls.iloc[(calls['strike'] - current_price).abs().idxmin()]
    atm_put = puts.iloc[(puts['strike'] - current_price).abs().idxmin()]

    return atm_call['strike'], atm_call['lastPrice'], atm_put['strike'], atm_put['lastPrice']

def long_call(stock_prices, strike, premium):
    """
    Calculate profit/loss for a long call option.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Strike price of the call option.
        premium (float): Premium paid for the call option.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    
    Raises:
        TypeError: If inputs are not of the expected types
    """
    if not isinstance(stock_prices, np.ndarray):
        stock_prices = np.array(stock_prices)
    
    if not isinstance(strike, (int, float)):
        raise TypeError("strike must be a number")
        
    if not isinstance(premium, (int, float)):
        raise TypeError("premium must be a number")
        
    result = []
    for stock_price in stock_prices:
        if stock_price < strike:
            result.append(-premium)  # Loss is the premium paid
        else:
            result.append((stock_price - strike) - premium)
    return np.array(result)

@st.cache_data()
def long_straddle(stock_prices, strike, premium_call, premium_put):
    """
    Calculate profit/loss for a long straddle strategy.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Strike price of the options.
        premium_call (float): Premium paid for the call option.
        premium_put (float): Premium paid for the put option.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    
    Raises:
        TypeError: If inputs are not of the expected types
    """
    if not isinstance(stock_prices, np.ndarray):
        stock_prices = np.array(stock_prices)
        
    if not isinstance(strike, (int, float)):
        raise TypeError("strike must be a number")
        
    if not isinstance(premium_call, (int, float)):
        raise TypeError("premium_call must be a number")
        
    if not isinstance(premium_put, (int, float)):
        raise TypeError("premium_put must be a number")
        
    return (long_call(stock_prices, strike, premium_call) +
        long_put(stock_prices, strike, premium_put))

@st.cache_data()
def short_straddle(stock_prices, strike, premium_call, premium_put):
    """
    Calculate profit/loss for a short straddle strategy.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Strike price of the options.
        premium_call (float): Premium received for the call option.
        premium_put (float): Premium received for the put option.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    
    Raises:
        TypeError: If inputs are not of the expected types
    """
    if not isinstance(stock_prices, np.ndarray):
        stock_prices = np.array(stock_prices)
        
    if not isinstance(strike, (int, float)):
        raise TypeError("strike must be a number")
        
    if not isinstance(premium_call, (int, float)):
        raise TypeError("premium_call must be a number")
        
    if not isinstance(premium_put, (int, float)):
        raise TypeError("premium_put must be a number")
        
    return (short_call(stock_prices, strike, premium_call) +
        short_put(stock_prices, strike, premium_put))

@st.cache_data()
def bull_call_spread(stock_prices, strike, premium, strike_high, premium_high):
    """
    Calculate profit/loss for a bull call spread strategy.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Lower strike price.
        premium (float): Premium paid for the lower strike call.
        strike_high (float): Higher strike price.
        premium_high (float): Premium received for the higher strike call.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    
    Raises:
        TypeError: If inputs are not of the expected types
        ValueError: If strike_high <= strike
    """
    if not isinstance(stock_prices, np.ndarray):
        stock_prices = np.array(stock_prices)
        
    if not isinstance(strike, (int, float)):
        raise TypeError("strike must be a number")
        
    if not isinstance(premium, (int, float)):
        raise TypeError("premium must be a number")
        
    if not isinstance(strike_high, (int, float)):
        raise TypeError("strike_high must be a number")
        
    if not isinstance(premium_high, (int, float)):
        raise TypeError("premium_high must be a number")
        
    if strike_high <= strike:
        raise ValueError("strike_high must be greater than strike for a bull call spread")
        
    return (long_call(stock_prices, strike, premium) +
        short_call(stock_prices, strike_high, premium_high))

@st.cache_data()
def bear_call_spread(stock_prices, strike, premium, strike_high, premium_high):
    """
    Calculate profit/loss for a bear call spread strategy.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Lower strike price.
        premium (float): Premium received for the lower strike call.
        strike_high (float): Higher strike price.
        premium_high (float): Premium paid for the higher strike call.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    
    Raises:
        TypeError: If inputs are not of the expected types
        ValueError: If strike_high <= strike
    """
    if not isinstance(stock_prices, np.ndarray):
        stock_prices = np.array(stock_prices)
        
    if not isinstance(strike, (int, float)):
        raise TypeError("strike must be a number")
        
    if not isinstance(premium, (int, float)):
        raise TypeError("premium must be a number")
        
    if not isinstance(strike_high, (int, float)):
        raise TypeError("strike_high must be a number")
        
    if not isinstance(premium_high, (int, float)):
        raise TypeError("premium_high must be a number")
        
    if strike_high <= strike:
        raise ValueError("strike_high must be greater than strike for a bear call spread")
        
    return (short_call(stock_prices, strike, premium) +
        long_call(stock_prices, strike_high, premium_high))

@st.cache_data()
def bull_put_spread(stock_prices, strike, premium, strike_high, premium_high):
    """
    Calculate profit/loss for a bull put spread strategy.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Higher strike price.
        premium (float): Premium received for the higher strike put.
        strike_high (float): Lower strike price.
        premium_high (float): Premium paid for the lower strike put.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    
    Raises:
        TypeError: If inputs are not of the expected types
        ValueError: If strike <= strike_high
    """
    if not isinstance(stock_prices, np.ndarray):
        stock_prices = np.array(stock_prices)
        
    if not isinstance(strike, (int, float)):
        raise TypeError("strike must be a number")
        
    if not isinstance(premium, (int, float)):
        raise TypeError("premium must be a number")
        
    if not isinstance(strike_high, (int, float)):
        raise TypeError("strike_high must be a number")
        
    if not isinstance(premium_high, (int, float)):
        raise TypeError("premium_high must be a number")
        
    if strike <= strike_high:
        raise ValueError("strike must be greater than strike_high for a bull put spread")
        
    return (short_put(stock_prices, strike, premium) +
        long_put(stock_prices, strike_high, premium_high))

@st.cache_data()
def bear_put_spread(stock_prices, strike, premium, strike_high, premium_high):
    """
    Calculate profit/loss for a bear put spread strategy.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Higher strike price.
        premium (float): Premium paid for the higher strike put.
        strike_high (float): Lower strike price.
        premium_high (float): Premium received for the lower strike put.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    
    Raises:
        TypeError: If inputs are not of the expected types
        ValueError: If strike <= strike_high
    """
    if not isinstance(stock_prices, np.ndarray):
        stock_prices = np.array(stock_prices)
        
    if not isinstance(strike, (int, float)):
        raise TypeError("strike must be a number")
        
    if not isinstance(premium, (int, float)):
        raise TypeError("premium must be a number")
        
    if not isinstance(strike_high, (int, float)):
        raise TypeError("strike_high must be a number")
        
    if not isinstance(premium_high, (int, float)):
        raise TypeError("premium_high must be a number")
        
    if strike <= strike_high:
        raise ValueError("strike must be greater than strike_high for a bear put spread")
        
    return (long_put(stock_prices, strike, premium) +
        short_put(stock_prices, strike_high, premium_high))

@st.cache_data()
def plot_strategy(stock_ticker, strategy, strike):
    """
    Plot the profit/loss diagram for the selected option strategy.

    Args:
        stock_ticker (str): The stock ticker symbol.
        strategy (str): The option strategy to plot.
        strike (float): The strike price for the strategy.
    
    Returns:
        fig: The Plotly figure object, or None if there was an error
    
    Raises:
        TypeError: If inputs are not of the expected types
        ValueError: If strategy is not recognized
    """
    if not isinstance(stock_ticker, str):
        raise TypeError("stock_ticker must be a string")
        
    if not isinstance(strategy, str):
        raise TypeError("strategy must be a string")
        
    if not isinstance(strike, (int, float)):
        raise TypeError("strike must be a number")
    
    strategy_list = get_available_strategies()
    if strategy not in strategy_list:
        raise ValueError(f"strategy must be one of: {', '.join(strategy_list)}")
    
    data = get_stock_data(stock_ticker)
    if data.empty:
        return None

    atm_call_strike, call_premium, atm_put_strike, put_premium = get_option_data(stock_ticker)
    if atm_call_strike is None:
        return None

    current_price = data['Close'].iloc[-1]
    stock_prices = np.linspace(strike * 0.8, strike * 1.2, 200)

    strategy_funcs = {
        "Long Call": long_call,
        "Short Call": short_call,
        "Long Put": long_put,
        "Short Put": short_put,
        "Long Straddle": long_straddle,
        "Short Straddle": short_straddle,
        "Bull Call Spread": bull_call_spread,
        "Bear Call Spread": bear_call_spread,
        "Bull Put Spread": bull_put_spread,
        "Bear Put Spread": bear_put_spread
    }

    if strategy in ["Long Straddle", "Short Straddle"]:
        profit_loss = strategy_funcs[strategy](stock_prices, strike, call_premium, put_premium)
    elif "Spread" in strategy:
        profit_loss = (strategy_funcs[strategy](stock_prices, strike,
            call_premium, atm_put_strike, put_premium))
    else:
        profit_loss = strategy_funcs[strategy](stock_prices, strike, call_premium)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_prices, y=profit_loss, mode='lines', name=strategy))
    fig.add_hline(y=0, line={"color": "black", "dash": "dash"})
    fig.add_vline(x=current_price, line={"color": "blue", "dash": "dot"},
                  annotation_text=f"Current Price: {current_price:.2f}")
    min_loss = min(profit_loss)
    max_profit = max(profit_loss)
    fig.update_yaxes(range=[min_loss * 1.2, max_profit * 1.2])
    fig.add_trace(go.Scatter(x=stock_prices, y=np.maximum(profit_loss, 0),
                            fill='tozeroy', fillcolor='rgba(0,255,0,0.3)',
                            line={"color": "rgba(0,0,0,0)"}, showlegend=False))
    fig.add_trace(go.Scatter(x=stock_prices, y=np.minimum(profit_loss, 0),
                            fill='tozeroy', fillcolor='rgba(255,0,0,0.3)',
                            line={"color": "rgba(0,0,0,0)"}, showlegend=False))
    fig.update_layout(
        title=f"{strategy} Strategy for {stock_ticker}",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit / Loss",
        template="plotly_dark",
        hovermode="x"
    )
    return fig

def get_available_strategies():
    """
    Returns a list of available options strategies
    
    Returns:
        list: List of strategy names
    """
    return [
        "Long Call", "Short Call", "Long Put", "Short Put", 
        "Long Straddle", "Short Straddle", "Bull Call Spread", "Bear Call Spread",
        "Bull Put Spread", "Bear Put Spread"
    ]

def get_strategy_description(strategy):
    """
    Get a detailed description of the selected strategy
    
    Args:
        strategy (str): Name of the strategy
        
    Returns:
        str: Markdown-formatted description of the strategy
    
    Raises:
        ValueError: If strategy is not recognized
    """
    if not isinstance(strategy, str):
        raise TypeError("strategy must be a string")
        
    if strategy not in get_available_strategies():
        raise ValueError(f"strategy must be one of: {', '.join(get_available_strategies())}")
        
    descriptions = {
        "Long Call": """
        **Long Call Strategy**: Buying a call option gives you the right to purchase the underlying stock at the strike price. 
        This strategy is bullish - you profit when the stock price rises above the strike price plus the premium paid.
        - **Maximum Loss**: Limited to the premium paid
        - **Maximum Gain**: Unlimited as the stock price rises
        - **Breakeven Point**: Strike price + premium paid
        """,
        
        "Short Call": """
        **Short Call Strategy**: Selling a call option obligates you to sell the underlying stock at the strike price if assigned.
        This strategy is bearish or neutral - you profit when the stock stays below the strike price.
        - **Maximum Gain**: Limited to the premium received
        - **Maximum Loss**: Unlimited as the stock price rises
        - **Breakeven Point**: Strike price + premium received
        """,
        
        "Long Put": """
        **Long Put Strategy**: Buying a put option gives you the right to sell the underlying stock at the strike price.
        This strategy is bearish - you profit when the stock price falls below the strike price minus the premium paid.
        - **Maximum Loss**: Limited to the premium paid
        - **Maximum Gain**: Limited to the strike price minus premium (if stock goes to zero)
        - **Breakeven Point**: Strike price - premium paid
        """,
        
        "Short Put": """
        **Short Put Strategy**: Selling a put option obligates you to buy the underlying stock at the strike price if assigned.
        This strategy is bullish or neutral - you profit when the stock stays above the strike price.
        - **Maximum Gain**: Limited to the premium received
        - **Maximum Loss**: Strike price - premium received (if stock goes to zero)
        - **Breakeven Point**: Strike price - premium received
        """,
        
        "Long Straddle": """
        **Long Straddle Strategy**: Buying both a call and a put at the same strike price.
        This strategy profits from significant price movement in either direction.
        - **Maximum Loss**: Limited to the combined premium paid for both options
        - **Maximum Gain**: Unlimited to the upside, limited to strike price minus premium to the downside
        - **Breakeven Points**: Strike price ± combined premium paid
        """,
        
        "Short Straddle": """
        **Short Straddle Strategy**: Selling both a call and a put at the same strike price.
        This strategy profits from minimal price movement in either direction.
        - **Maximum Gain**: Limited to the combined premium received
        - **Maximum Loss**: Unlimited to the upside, limited to strike price minus premium to the downside
        - **Breakeven Points**: Strike price ± combined premium received
        """,
        
        "Bull Call Spread": """
        **Bull Call Spread Strategy**: Buying a call at a lower strike price and selling a call at a higher strike price.
        This strategy is moderately bullish with defined risk and reward.
        - **Maximum Loss**: Limited to the net premium paid
        - **Maximum Gain**: Difference between strikes minus net premium paid
        - **Breakeven Point**: Lower strike price + net premium paid
        """,
        
        "Bear Call Spread": """
        **Bear Call Spread Strategy**: Selling a call at a lower strike price and buying a call at a higher strike price.
        This strategy is moderately bearish with defined risk and reward.
        - **Maximum Loss**: Difference between strikes minus net premium received
        - **Maximum Gain**: Limited to the net premium received
        - **Breakeven Point**: Lower strike price + net premium received
        """,
        
        "Bull Put Spread": """
        **Bull Put Spread Strategy**: Selling a put at a higher strike price and buying a put at a lower strike price.
        This strategy is moderately bullish with defined risk and reward.
        - **Maximum Loss**: Difference between strikes minus net premium received
        - **Maximum Gain**: Limited to the net premium received
        - **Breakeven Point**: Higher strike price - net premium received
        """,
        
        "Bear Put Spread": """
        **Bear Put Spread Strategy**: Buying a put at a higher strike price and selling a put at a lower strike price.
        This strategy is moderately bearish with defined risk and reward.
        - **Maximum Loss**: Limited to the net premium paid
        - **Maximum Gain**: Difference between strikes minus net premium paid
        - **Breakeven Point**: Higher strike price - net premium paid
        """
    }
    
    return descriptions.get(strategy, "Strategy description not available.")

@st.cache_data()
def short_call(stock_prices, strike, premium):
    """
    Calculate profit/loss for a short call option.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Strike price of the call option.
        premium (float): Premium received for the call option.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    
    Raises:
        TypeError: If inputs are not of the expected types
    """
    if not isinstance(stock_prices, np.ndarray):
        stock_prices = np.array(stock_prices)
    
    if not isinstance(strike, (int, float)):
        raise TypeError("strike must be a number")
        
    if not isinstance(premium, (int, float)):
        raise TypeError("premium must be a number")
        
    result = []
    for stock_price in stock_prices:
        if stock_price < strike:
            result.append(premium)  # Profit is the premium received
        else:
            result.append(premium - (stock_price - strike))
    return np.array(result)

@st.cache_data()
def long_put(stock_prices, strike, premium):
    """
    Calculate profit/loss for a long put option.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Strike price of the put option.
        premium (float): Premium paid for the put option.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    
    Raises:
        TypeError: If inputs are not of the expected types
    """
    if not isinstance(stock_prices, np.ndarray):
        stock_prices = np.array(stock_prices)
    
    if not isinstance(strike, (int, float)):
        raise TypeError("strike must be a number")
        
    if not isinstance(premium, (int, float)):
        raise TypeError("premium must be a number")
        
    result = []
    for stock_price in stock_prices:
        if stock_price > strike:
            result.append(-premium)  # Loss is the premium paid
        else:
            result.append((strike - stock_price) - premium)
    return np.array(result)

@st.cache_data()
def short_put(stock_prices, strike, premium):
    """
    Calculate profit/loss for a short put option.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Strike price of the put option.
        premium (float): Premium received for the put option.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    
    Raises:
        TypeError: If inputs are not of the expected types
    """
    if not isinstance(stock_prices, np.ndarray):
        stock_prices = np.array(stock_prices)
    
    if not isinstance(strike, (int, float)):
        raise TypeError("strike must be a number")
        
    if not isinstance(premium, (int, float)):
        raise TypeError("premium must be a number")
        
    result = []
    for stock_price in stock_prices:
        if stock_price > strike:
            result.append(premium)  # Profit is the premium received
        else:
            result.append(premium - (strike - stock_price))
    return np.array(result)
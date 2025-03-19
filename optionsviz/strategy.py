"""
Module Name: strategy.py

Description:
    This module serves as the streamlit options strategy visualization page. 
    It fetches stock and option data, calculates profit/loss for various option strategies, 
    and plots the results using Streamlit and Plotly.

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

def get_stock_data(stock_ticker, period='1y'):
    """Fetch historical stock data for a given ticker and period.

    Args:
        stock_ticker (str): The stock ticker symbol.
        period (str, optional): The historical period. Defaults to '1y'.

    Returns:
        pandas.DataFrame: Historical stock data.
    """
    stock = yf.Ticker(stock_ticker)
    return stock.history(period=period)

def get_option_data(stock_ticker):
    """Retrieve option chain data for a given ticker.

    Args:
        stock_ticker (str): The stock ticker symbol.

    Returns:
        tuple: (atm_call_strike, call_premium, atm_put_strike, put_premium) 
        or (None, None, None, None)
    """
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
    """Calculate profit/loss for a long call option.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Strike price of the call option.
        premium (float): Premium paid for the call option.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    """
    result = []
    for stock_price in stock_prices:
        if stock_price < strike:
            result.append(-premium)  # Loss is the premium paid
        else:
            result.append((stock_price - strike) - premium)
    return np.array(result)

def short_call(stock_prices, strike, premium):
    """Calculate profit/loss for a short call option.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Strike price of the call option.
        premium (float): Premium received for the call option.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    """
    result = []
    for stock_price in stock_prices:
        if stock_price < strike:
            result.append(premium)  # Profit is the premium received
        else:
            result.append(premium - (stock_price - strike))
    return np.array(result)


def long_put(stock_prices, strike, premium):
    """Calculate profit/loss for a long put option.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Strike price of the put option.
        premium (float): Premium paid for the put option.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    """
    result = []
    for stock_price in stock_prices:
        if stock_price > strike:
            result.append(-premium)  # Loss is the premium paid
        else:
            result.append((strike - stock_price) - premium)
    return np.array(result)

def short_put(stock_prices, strike, premium):
    """Calculate profit/loss for a short put option.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Strike price of the put option.
        premium (float): Premium received for the put option.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    """
    result = []
    for stock_price in stock_prices:
        if stock_price > strike:
            result.append(premium)  # Profit is the premium received
        else:
            result.append(premium - (strike - stock_price))
    return np.array(result)

def long_straddle(stock_prices, strike, premium_call, premium_put):
    """Calculate profit/loss for a long straddle strategy.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Strike price of the options.
        premium_call (float): Premium paid for the call option.
        premium_put (float): Premium paid for the put option.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    """
    return (long_call(stock_prices, strike, premium_call) +
        long_put(stock_prices, strike, premium_put))

def short_straddle(stock_prices, strike, premium_call, premium_put):
    """Calculate profit/loss for a short straddle strategy.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Strike price of the options.
        premium_call (float): Premium received for the call option.
        premium_put (float): Premium received for the put option.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    """
    return (short_call(stock_prices, strike, premium_call) +
        short_put(stock_prices, strike, premium_put))

def bull_call_spread(stock_prices, strike, premium, strike_high, premium_high):
    """Calculate profit/loss for a bull call spread strategy.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Lower strike price.
        premium (float): Premium paid for the lower strike call.
        strike_high (float): Higher strike price.
        premium_high (float): Premium received for the higher strike call.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    """
    return (long_call(stock_prices, strike, premium) +
        short_call(stock_prices, strike_high, premium_high))

def bear_call_spread(stock_prices, strike, premium, strike_high, premium_high):
    """Calculate profit/loss for a bear call spread strategy.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Lower strike price.
        premium (float): Premium received for the lower strike call.
        strike_high (float): Higher strike price.
        premium_high (float): Premium paid for the higher strike call.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    """
    return (short_call(stock_prices, strike, premium) +
        long_call(stock_prices, strike_high, premium_high))

def bull_put_spread(stock_prices, strike, premium, strike_high, premium_high):
    """Calculate profit/loss for a bull put spread strategy.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Higher strike price.
        premium (float): Premium received for the higher strike put.
        strike_high (float): Lower strike price.
        premium_high (float): Premium paid for the lower strike put.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    """
    return (short_put(stock_prices, strike, premium) +
        long_put(stock_prices, strike_high, premium_high))

def bear_put_spread(stock_prices, strike, premium, strike_high, premium_high):
    """Calculate profit/loss for a bear put spread strategy.

    Args:
        stock_prices (numpy.ndarray): Array of stock prices.
        strike (float): Higher strike price.
        premium (float): Premium paid for the higher strike put.
        strike_high (float): Lower strike price.
        premium_high (float): Premium received for the lower strike put.

    Returns:
        numpy.ndarray: Array of profit/loss values.
    """
    return (long_put(stock_prices, strike, premium) +
        short_put(stock_prices, strike_high, premium_high))

def plot_strategy(stock_ticker, strategy, strike):
    """Plot the profit/loss diagram for the selected option strategy.

    Args:
        stock_ticker (str): The stock ticker symbol.
        strategy (str): The option strategy to plot.
        strike (float): The strike price for the strategy.
    """
    data = get_stock_data(stock_ticker)
    if data.empty:
        st.error("Could not fetch stock data. Please check the ticker.")
        return

    atm_call_strike, call_premium, atm_put_strike, put_premium = get_option_data(stock_ticker)
    if atm_call_strike is None:
        st.error("Could not fetch options data. Please try another ticker.")
        return

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
    st.plotly_chart(fig)

ticker = st.text_input("Enter stock ticker:", value="", placeholder='e.g. NVDA, AAPL, AMZN').upper()
if ticker:
    strategies = [
        "Long Call", "Short Call", "Long Put", "Short Put", 
        "Long Straddle", "Short Straddle", "Bull Call Spread", "Bear Call Spread",
        "Bull Put Spread", "Bear Put Spread"
    ]
    strat = st.selectbox("Select a strategy", strategies)
    atm_strike_call, _, atm_strike_put, _ = get_option_data(ticker)
    if atm_strike_call:
        strike_price = (st.slider("Select Strike Price", int(atm_strike_call * 0.8)
        , int(atm_strike_call * 1.2), int(atm_strike_call)))
        plot_strategy(ticker, strat, strike_price)

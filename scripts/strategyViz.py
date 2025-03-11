import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Define a function to fetch stock data
def get_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

# Define strategy simulation functions
def long_call(stock_prices, strike_price, premium):
    return np.maximum(stock_prices - strike_price, 0) - premium

def short_call(stock_prices, strike_price, premium):
    return np.minimum(-(stock_prices - strike_price), 0) + premium

def long_put(stock_prices, strike_price, premium):
    return np.maximum(strike_price - stock_prices, 0) - premium

def short_put(stock_prices, strike_price, premium):
    return np.minimum(-(strike_price - stock_prices), 0) + premium

def bull_call_spread(stock_prices, lower_strike, upper_strike, lower_premium, upper_premium):
    long_call_profit = long_call(stock_prices, lower_strike, lower_premium)
    short_call_profit = short_call(stock_prices, upper_strike, upper_premium)
    return long_call_profit + short_call_profit

def bear_call_spread(stock_prices, lower_strike, upper_strike, lower_premium, upper_premium):
    short_call_profit = short_call(stock_prices, lower_strike, lower_premium)
    long_call_profit = long_call(stock_prices, upper_strike, upper_premium)
    return short_call_profit + long_call_profit

def bull_put_spread(stock_prices, lower_strike, upper_strike, lower_premium, upper_premium):
    short_put_profit = short_put(stock_prices, upper_strike, upper_premium)
    long_put_profit = long_put(stock_prices, lower_strike, lower_premium)
    return short_put_profit + long_put_profit

def bear_put_spread(stock_prices, lower_strike, upper_strike, lower_premium, upper_premium):
    long_put_profit = long_put(stock_prices, upper_strike, upper_premium)
    short_put_profit = short_put(stock_prices, lower_strike, lower_premium)
    return long_put_profit + short_put_profit

# Define function to plot strategies
def plot_strategy(ticker, strategy, strike_price, premium):
    data = get_stock_data(ticker)
    stock_prices = np.linspace(strike_price * 0.5, strike_price * 1.5, 200)  # Expand price range
    
    strategy_funcs = {
        "Long Call": long_call,
        "Short Call": short_call,
        "Long Put": long_put,
        "Short Put": short_put,
        "Bull Call Spread": bull_call_spread,
        "Bear Call Spread": bear_call_spread,
        "Bull Put Spread": bull_put_spread,
        "Bear Put Spread": bear_put_spread,
    }
    
    if strategy in strategy_funcs:
        if "Spread" in strategy:
            profit_loss = strategy_funcs[strategy](stock_prices, strike_price, strike_price + 10, premium, premium * 0.8)
        else:
            profit_loss = strategy_funcs[strategy](stock_prices, strike_price, premium)
    else:
        messagebox.showerror("Error", "Strategy not implemented.")
        return
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_prices, y=profit_loss, mode='lines', name=strategy))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    
    fig.update_layout(
        title=f"{strategy} Strategy for {ticker}",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit / Loss",
        template="plotly_dark",
        hovermode="x",
    )
    
    fig.show()

# GUI Setup
def run_gui():
    def on_generate():
        ticker = ticker_input.get()
        strategy = strategy_input.get()
        try:
            strike_price = float(strike_price_input.get())
            premium = float(premium_input.get())
            plot_strategy(ticker, strategy, strike_price, premium)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values.")
    
    root = tk.Tk()
    root.title("Stock Strategy Visualizer")
    
    ttk.Label(root, text="Stock Ticker:").grid(row=0, column=0)
    ticker_input = ttk.Entry(root)
    ticker_input.grid(row=0, column=1)
    ticker_input.insert(0, "AAPL")
    
    ttk.Label(root, text="Strategy:").grid(row=1, column=0)
    strategy_input = ttk.Combobox(root, values=[
        "Long Call", "Short Call", "Long Put", "Short Put", "Bull Call Spread", "Bear Call Spread", "Bull Put Spread", "Bear Put Spread"
    ])
    strategy_input.grid(row=1, column=1)
    strategy_input.current(0)
    
    ttk.Label(root, text="Strike Price:").grid(row=2, column=0)
    strike_price_input = ttk.Entry(root)
    strike_price_input.grid(row=2, column=1)
    strike_price_input.insert(0, "150")
    
    ttk.Label(root, text="Premium:").grid(row=3, column=0)
    premium_input = ttk.Entry(root)
    premium_input.grid(row=3, column=1)
    premium_input.insert(0, "5")
    
    generate_button = ttk.Button(root, text="Generate Plot", command=on_generate)
    generate_button.grid(row=4, columnspan=2)
    
    root.mainloop()

if __name__ == "__main__":
    run_gui()

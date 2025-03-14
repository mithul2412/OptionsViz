import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import streamlit as st

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

# Long Straddle and Short Straddle strategies
def long_straddle(stock_prices, strike_price, premium_call, premium_put):
    # Long call + Long put
    return long_call(stock_prices, strike_price, premium_call) + long_put(stock_prices, strike_price, premium_put)

def short_straddle(stock_prices, strike_price, premium_call, premium_put):
    # Short call + Short put
    return short_call(stock_prices, strike_price, premium_call) + short_put(stock_prices, strike_price, premium_put)

# Spread strategies
def bull_call_spread(stock_prices, strike_price_low, strike_price_high, premium_low, premium_high):
    return long_call(stock_prices, strike_price_low, premium_low) + short_call(stock_prices, strike_price_high, premium_high)

def bear_call_spread(stock_prices, strike_price_low, strike_price_high, premium_low, premium_high):
    return short_call(stock_prices, strike_price_low, premium_low) + long_call(stock_prices, strike_price_high, premium_high)

def bull_put_spread(stock_prices, strike_price_low, strike_price_high, premium_low, premium_high):
    return short_put(stock_prices, strike_price_low, premium_low) + long_put(stock_prices, strike_price_high, premium_high)

def bear_put_spread(stock_prices, strike_price_low, strike_price_high, premium_low, premium_high):
    return long_put(stock_prices, strike_price_low, premium_low) + short_put(stock_prices, strike_price_high, premium_high)

# Function to calculate Vega (approximation)
def calculate_vega(stock_price, strike_price, premium):
    return abs(stock_price - strike_price) * 0.01  # Simplified estimate

# Define function to plot strategies
def plot_strategy(ticker, strategy, strike_price, premium, premium_put=None, strike_price_high=None, premium_high=None):
    data = get_stock_data(ticker)
    current_price = data['Close'][-1]
    stock_prices = np.linspace(strike_price * 0.5, strike_price * 1.5, 200)
    
    strategy_funcs = {
        "Long Call": long_call,
        "Short Call": short_call,
        "Long Put": long_put,
        "Short Put": short_put,
        "Bull Call Spread": bull_call_spread,
        "Bear Call Spread": bear_call_spread,
        "Bull Put Spread": bull_put_spread,
        "Bear Put Spread": bear_put_spread,
        "Long Straddle": long_straddle,  # Add Long Straddle
        "Short Straddle": short_straddle  # Add Short Straddle
    }
    
    if strategy in strategy_funcs:
        if "Spread" in strategy:
            # Ensure user input is valid for spreads
            if strike_price_high is None or premium_high is None:
                messagebox.showerror("Input Error", "Please enter both higher strike price and higher premium for spreads.")
                return
            profit_loss = strategy_funcs[strategy](stock_prices, strike_price, strike_price_high, premium, premium_high)
        elif "Straddle" in strategy:
            if premium_put is None:
                messagebox.showerror("Input Error", "Please enter both call and put premiums for straddle strategies.")
                return
            profit_loss = strategy_funcs[strategy](stock_prices, strike_price, premium, premium_put)
        else:
            profit_loss = strategy_funcs[strategy](stock_prices, strike_price, premium)
    else:
        messagebox.showerror("Error", "Strategy not implemented.")
        return
    
    # For spread strategies, use both strike prices and premiums to calculate current profit/loss
    if "Spread" in strategy:
        current_profit_loss = strategy_funcs[strategy](np.array([current_price]), strike_price, strike_price_high, premium, premium_high)[0]
    elif "Straddle" in strategy:
        current_profit_loss = strategy_funcs[strategy](np.array([current_price]), strike_price, premium, premium_put)[0]
    else:
        current_profit_loss = strategy_funcs[strategy](np.array([current_price]), strike_price, premium)[0]
    
    vega = calculate_vega(current_price, strike_price, premium)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_prices, y=profit_loss, mode='lines', name=strategy))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    
    # Shade profit and loss areas
    fig.add_trace(go.Scatter(x=stock_prices, y=np.maximum(profit_loss, 0), fill='tozeroy', fillcolor='rgba(0,255,0,0.3)', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=stock_prices, y=np.minimum(profit_loss, 0), fill='tozeroy', fillcolor='rgba(255,0,0,0.3)', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    
    # Add vertical line for current price
    fig.add_vline(x=current_price, line=dict(color="blue", dash="dot"), annotation_text=f"Current Price: {current_price:.2f}\nProfit/Loss: {current_profit_loss:.2f}\nVega: {vega:.4f}", annotation_position="top left")
    
    fig.update_layout(
        title=f"{strategy} Strategy for {ticker}",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit / Loss",
        template="plotly_dark",
        hovermode="x",
    )
    
    # fig.show()
    st.plotly_chart(fig)


ticker = st.text_input("Enter stock ticker:", value=None, placeholder='e.g. NVDA, AAPL, AMZN')
if ticker:
    ticker = ticker.upper()


    strategies=[
    "Long Call",
    "Short Call",
    "Long Put",
    "Short Put",
    "Bull Call Spread",
    "Bear Call Spread",
    "Bull Put Spread",
    "Bear Put Spread",
    "Long Straddle","Short Straddle"]

    strat = st.selectbox(
                "Select a strategy",
                strategies,
        )


    strike = st.number_input("Strike", min_value=1, value=1_000)
    prem = st.number_input("Premium", min_value=1, value=1_000)

    plot_strategy(ticker, strat,
                strike, prem, 
                strike_price_high=160,
                    premium_high=10.0)


# # GUI Setup
# def run_gui():
#     def on_generate():
#         ticker = ticker_input.get()
#         strategy = strategy_input.get()
#         try:
#             strike_price = float(strike_price_input.get())
#             premium = float(premium_input.get())

#             # If a spread strategy is selected, get additional inputs
#             if "Spread" in strategy:
#                 strike_price_high = float(strike_price_high_input.get())
#                 premium_high = float(premium_high_input.get())
                
#                 # Check that both higher strike price and premium are entered
#                 if not strike_price_high or not premium_high:
#                     messagebox.showerror("Input Error", "Please enter both higher strike price and higher premium for spread strategies.")
#                     return
                
#                 plot_strategy(ticker, strategy, strike_price, premium, strike_price_high=strike_price_high, premium_high=premium_high)
#             elif "Straddle" in strategy:  # For Long/Short Straddle
#                 premium_call = float(premium_call_input.get())
#                 premium_put = float(premium_put_input.get())
                
#                 # Check that both premiums are entered for straddle strategies
#                 if not premium_call or not premium_put:
#                     messagebox.showerror("Input Error", "Please enter both call and put premiums for straddle strategies.")
#                     return
                    
#                 plot_strategy(ticker, strategy, strike_price, premium_call, premium_put=premium_put)
#             else:
#                 plot_strategy(ticker, strategy, strike_price, premium)
#         except ValueError:
#             messagebox.showerror("Input Error", "Please enter valid numerical values.")

#     def update_fields(event):
#         strategy = strategy_input.get()
#         if "Spread" in strategy:
#             strike_price_high_label.grid()
#             strike_price_high_input.grid()
#             premium_high_label.grid()
#             premium_high_input.grid()
#             premium_call_label.grid_remove()
#             premium_call_input.grid_remove()
#             premium_put_label.grid_remove()
#             premium_put_input.grid_remove()
#         elif "Straddle" in strategy:
#             premium_call_label.grid()
#             premium_call_input.grid()
#             premium_put_label.grid()
#             premium_put_input.grid()
#             strike_price_high_label.grid_remove()
#             strike_price_high_input.grid_remove()
#             premium_high_label.grid_remove()
#             premium_high_input.grid_remove()
#         else:
#             strike_price_high_label.grid_remove()
#             strike_price_high_input.grid_remove()
#             premium_high_label.grid_remove()
#             premium_high_input.grid_remove()
#             premium_call_label.grid_remove()
#             premium_call_input.grid_remove()
#             premium_put_label.grid_remove()
#             premium_put_input.grid_remove()

#     root = tk.Tk()
#     root.title("Stock Strategy Visualizer")

#     ttk.Label(root, text="Stock Ticker:").grid(row=0, column=0)
#     ticker_input = ttk.Entry(root)
#     ticker_input.grid(row=0, column=1)
#     ticker_input.insert(0, "AAPL")

#     ttk.Label(root, text="Strategy:").grid(row=1, column=0)
#     strategy_input = ttk.Combobox(root, values=[
#         "Long Call", "Short Call", "Long Put", "Short Put", 
#         "Bull Call Spread", "Bear Call Spread", 
#         "Bull Put Spread", "Bear Put Spread",
#         "Long Straddle", "Short Straddle"  # Add new strategies here
#     ])
#     strategy_input.grid(row=1, column=1)
#     strategy_input.current(0)
#     strategy_input.bind("<<ComboboxSelected>>", update_fields)

#     ttk.Label(root, text="Strike Price:").grid(row=2, column=0)
#     strike_price_input = ttk.Entry(root)
#     strike_price_input.grid(row=2, column=1)
#     strike_price_input.insert(0, "150")

#     ttk.Label(root, text="Premium:").grid(row=3, column=0)
#     premium_input = ttk.Entry(root)
#     premium_input.grid(row=3, column=1)
#     premium_input.insert(0, "5")

#     # Add additional inputs for Long Straddle and Short Straddle
#     premium_call_label = ttk.Label(root, text="Call Premium:")
#     premium_call_input = ttk.Entry(root)
#     premium_call_label.grid(row=4, column=0)
#     premium_call_input.grid(row=4, column=1)

#     premium_put_label = ttk.Label(root, text="Put Premium:")
#     premium_put_input = ttk.Entry(root)
#     premium_put_label.grid(row=5, column=0)
#     premium_put_input.grid(row=5, column=1)

#     # Add inputs for spread strategies
#     strike_price_high_label = ttk.Label(root, text="Higher Strike Price:")
#     strike_price_high_input = ttk.Entry(root)
#     strike_price_high_label.grid(row=6, column=0)
#     strike_price_high_input.grid(row=6, column=1)

#     premium_high_label = ttk.Label(root, text="Higher Premium:")
#     premium_high_input = ttk.Entry(root)
#     premium_high_label.grid(row=7, column=0)
#     premium_high_input.grid(row=7, column=1)

#     generate_button = ttk.Button(root, text="Generate Strategy Plot", command=on_generate)
#     generate_button.grid(row=8, column=0, columnspan=2)

#     root.mainloop()

# # Run GUI
# if __name__ == "__main__":
#     run_gui()
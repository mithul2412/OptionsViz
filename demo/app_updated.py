# IMPORTANT: Set environment variables FIRST, before any imports
import os
os.environ["STREAMLIT_WATCH_MODULE_PATH"] = "false"
os.environ["STREAMLIT_DISABLE_WATCHER"] = "true"

import streamlit as st
import pandas as pd
import sys
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import math
from typing import List, Dict, Any
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Add the news and strategies folders to sys.path
sys.path.append("news")
sys.path.append("strategies")
sys.path.append("dataviz")  # Add dataviz directory to path

try:
    from news.news_core import fetch_news, process_news_articles_whole
    from news.embedding_utils import (
        create_pinecone_index, 
        clear_pinecone_index, 
        upload_news_to_pinecone, 
        preprocess_query,
        NewsRetriever
    )
    from news.llm_interface import LLMInterface as NewsLLM
    NEWS_MODULES_AVAILABLE = True
except ImportError:
    NEWS_MODULES_AVAILABLE = False

# Import from Options Analysis modules
try:
    from strategies.json_packaging import build_compact_options_json
    from strategies.main import summarize_options_data, ask_llm_about_options
    from strategies.options_data import (
        get_fundamental_data,
        get_historical_data,
        compute_historical_volatility
    )
    OPTIONS_MODULES_AVAILABLE = True
except ImportError:
    OPTIONS_MODULES_AVAILABLE = False

# App configuration
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for navigation
if 'current_app' not in st.session_state:
    st.session_state['current_app'] = "eod_chain"  # Default to eod_chain

# Initialize News Analyzer session state
if 'fetched_articles' not in st.session_state:
    st.session_state['fetched_articles'] = []
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 1
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Initialize Options Analysis session state
if 'options_data' not in st.session_state:
    st.session_state['options_data'] = None
if 'options_summary' not in st.session_state:
    st.session_state['options_summary'] = None
if 'extracted_orders' not in st.session_state:
    st.session_state['extracted_orders'] = []
if 'alpaca_configured' not in st.session_state:
    st.session_state['alpaca_configured'] = False
if 'options_ticker' not in st.session_state:
    st.session_state['options_ticker'] = "AAPL"

# Function to change page for news articles
def change_page(direction):
    if direction == "next":
        st.session_state['current_page'] += 1
    else:
        st.session_state['current_page'] -= 1

# Function to format large numbers (for Options Analysis)
def format_large_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .app-selector {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        background-color: #1E1E1E;
        border: 1px solid #333;
    }
    .app-selector-btn {
        width: 100%;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .app-selector-btn-active {
        background-color: #2196F3;
        color: white;
    }
    .app-selector-btn-inactive {
        background-color: #333;
        color: #CCC;
    }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 1.2rem;
        color: #9E9E9E;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #FFFFFF;
    }
    .metric-value-green {
        color: #4CAF50;
    }
    .metric-value-red {
        color: #F44336;
    }
    .metric-value-blue {
        color: #2196F3;
    }
    .metric-value-amber {
        color: #FFC107;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2196F3;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .expiry-header {
        background-color: #232323;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .analyst-section {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #333;
        margin-top: 30px;
    }
    .analyst-header {
        color: #2196F3;
        font-size: 1.8rem;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

#############################################
# Sidebar with Application Selector
#############################################

# App selector in the sidebar
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    if st.button("📈    EOD Chain", key="eod_chain_btn", 
                 help="Visualize options chain data",
                 use_container_width=True, 
                 type="primary" if st.session_state['current_app'] == "eod_chain" else "secondary"):
        st.session_state['current_app'] = "eod_chain"
        st.rerun()

with col2:
    if st.button("📰 News Analyzer", key="news_btn", 
                 help="Analyze news articles with AI",
                 use_container_width=True, 
                 type="primary" if st.session_state['current_app'] == "news" else "secondary"):
        st.session_state['current_app'] = "news"
        st.rerun()

with col3:
    if st.button("📊 Option Insights", key="options_btn", 
                 help="Analyze stock options data",
                 use_container_width=True,
                 type="primary" if st.session_state['current_app'] == "options" else "secondary"):
        st.session_state['current_app'] = "options"
        st.rerun()

st.sidebar.divider()

#############################################
# App-specific sidebar options
#############################################

# Check for required API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
NEWSAPI_KEY = os.getenv('NEWS_API_KEY')

if st.session_state['current_app'] == "eod_chain":
    # EOD Chain sidebar
    st.sidebar.header("EOD Chain Settings")
    
    ticker_input = st.sidebar.text_input("Enter stock ticker:", value=None, placeholder='e.g. NVDA, AAPL, AMZN', key="eod_chain_ticker_sidebar_input")
    if ticker_input:
        ticker_input = ticker_input.upper()
        st.sidebar.success(f"Analyzing options chain for {ticker_input}")
    else:
        st.sidebar.info("Enter a ticker symbol to analyze the options chain.")
    
elif st.session_state['current_app'] == "news":
    # News Analyzer sidebar
    st.sidebar.header("News Analyzer Settings")
    
    # Pinecone settings
    index_name = st.sidebar.text_input("Pinecone Index Name", "newsdata", key="index_name_input")
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["intfloat/e5-large-v2", "intfloat/e5-base-v2", "BAAI/bge-large-en-v1.5", "BAAI/bge-base-v1.5"],
        index=0,
        key="embedding_model_select"
    )

    # LLM settings
    llm_model = st.sidebar.selectbox(
        "LLM Model",
        ["deepseek/deepseek-chat:free", "google/gemini-pro", "anthropic/claude-3-sonnet", "meta-llama/llama-3-8b-instruct"],
        index=0,
        key="llm_model_select"
    )

    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        show_query_processing = st.checkbox("Show Query Processing", value=False, key="show_query_processing_checkbox")
        results_count = st.slider("Number of Results", min_value=3, max_value=20, value=10, key="results_count_slider")
        
    # Initialize News services
    @st.cache_resource
    def get_retriever(index_name, model_name):
        try:
            from news.embedding_utils import NewsRetriever
            retriever = NewsRetriever(
                pinecone_api_key=PINECONE_API_KEY,
                index_name=index_name,
                model_name=model_name
            )
            return retriever
        except Exception as e:
            st.error(f"Error initializing retriever: {str(e)}")
            return None

    @st.cache_resource
    def get_llm(model_name):
        try:
            from news.llm_interface import LLMInterface as NewsLLM
            llm = NewsLLM(
                api_key=OPENROUTER_API_KEY,
                model=model_name
            )
            return llm
        except Exception as e:
            st.error(f"Error initializing LLM interface: {str(e)}")
            return None
            
    retriever = get_retriever(index_name, embedding_model)
    llm = get_llm(llm_model)
    
    # API key warnings
    if not PINECONE_API_KEY:
        st.sidebar.error("PINECONE_API_KEY not set. Required for vector database.")
    if not OPENROUTER_API_KEY:
        st.sidebar.error("OPENROUTER_API_KEY not set. Required for AI analysis.")
    if not NEWSAPI_KEY:
        st.sidebar.error("NEWS_API_KEY not set. Required for fetching news.")
    
else:
    # Options Analysis sidebar
    st.sidebar.header("Options Analysis Settings")

    ticker = st.sidebar.text_input("Ticker Symbol", "AAPL", key="options_ticker_input").upper()
    expirations_limit = st.sidebar.slider("Number of Expirations", min_value=1, max_value=10, value=3, key="expirations_limit_slider")
    include_hv = st.sidebar.checkbox("Include Historical Volatility", value=True, key="include_hv_checkbox")
    
    # Store ticker in session state for persistence
    if 'options_ticker' not in st.session_state or st.session_state['options_ticker'] != ticker:
        st.session_state['options_ticker'] = ticker
    
    # Button to fetch options data
    if st.sidebar.button("Analyze Options", key="analyze_options_button") or st.session_state.get('trigger_options_analysis', False):
        # Reset trigger if it was set
        if st.session_state.get('trigger_options_analysis'):
            st.session_state['trigger_options_analysis'] = False
            
        st.sidebar.info(f"Fetching data for {ticker}...")
        try:
            from strategies.json_packaging import build_compact_options_json
            from strategies.main import summarize_options_data, ask_llm_about_options
            
            with st.spinner(f"Fetching options data for {ticker}..."):
                data = build_compact_options_json(ticker, expirations_limit, include_hv)
                summary = summarize_options_data(data)
                
                st.session_state['options_data'] = data
                st.session_state['options_summary'] = summary
                
                # Clear any previous orders when analyzing a new ticker
                if 'last_ticker' not in st.session_state or st.session_state.get('last_ticker') != ticker:
                    st.session_state['extracted_orders'] = []
                
                st.session_state['last_ticker'] = ticker
                
            st.sidebar.success(f"Successfully retrieved options data for {ticker}")
            # Force rerun to display the data immediately
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error retrieving options data: {str(e)}")
            st.error(f"Error retrieving options data: {str(e)}")
    
    # Option to download the raw JSON data
    if 'options_data' in st.session_state and st.session_state['options_data']:
        if st.sidebar.button("Download Analysis Data", key="download_data_button"):
            json_str = json.dumps(st.session_state['options_data'], indent=2)
            st.sidebar.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{st.session_state['options_data']['ticker']}_options_analysis.json",
                mime="application/json",
                key="download_json_button"
            )
    
    # API key warnings
    if not OPENROUTER_API_KEY:
        st.sidebar.error("OPENROUTER_API_KEY not set. Required for strategy advisor.")

#############################################
# Main Content Area
#############################################

if st.session_state['current_app'] == "eod_chain":
    #############################################
    # EOD CHAIN APPLICATION
    #############################################
    
    st.markdown('<div class="main-header">Financial Analysis Dashboard</div>', unsafe_allow_html=True)
    # App title and description
    st.markdown("EOD Options Chain Visualization")
    st.markdown("""
    This fetches options chain data, and creates interactive vizualizations.
    """)
    # st.markdown('<div class="main-header">EOD Options Chain Visualization</div>', unsafe_allow_html=True)
    
    # Get ticker from sidebar instead of main page
    if not ticker_input:
        st.info("👈 Please enter a ticker symbol in the sidebar to analyze the options chain.")
    else:
        # Display the selected ticker prominently at the top
        st.subheader(f"Options Chain Analysis for {ticker_input}")
        st.divider()

    if ticker_input is not None:
        ticker_input = ticker_input.upper()

        # get option chain and proc
        yfticker = yf.Ticker(ticker_input)
        expiration_dates = yfticker.options

        ######### show unusual activity table
        df_full_chain_calls = None
        df_full_chain_puts = None
        for e in expiration_dates:
            opt = yfticker.option_chain(e)
            calls = opt.calls
            puts = opt.puts

            if df_full_chain_calls is None:
                df_full_chain_calls = calls.copy()
            else:
                df_full_chain_calls = pd.concat([df_full_chain_calls, calls])

            if df_full_chain_puts is None:
                df_full_chain_puts = puts.copy()
            else:
                df_full_chain_puts = pd.concat([df_full_chain_puts, puts])

        col_activity = st.columns((4,4), gap='small')

        with col_activity[0]:
            oi_min = st.number_input("Minumum OI", min_value=1, value=1_000,
                                    help='Minumum Open Interest to consider when computing unusual options activity.',
                                    key="oi_min_calls_input")
            show_itm = st.checkbox("Show ITM", value=False,
                                help='Only show in-the-money (ITM) contracts, otherwise show only out-of-money.',
                                key="show_itm_calls_checkbox")

            if not df_full_chain_calls.empty:
                df_full_chain_calls = df_full_chain_calls[df_full_chain_calls.volume != 0.]
                df_full_chain_calls = df_full_chain_calls[df_full_chain_calls.openInterest != 0.]
                df_full_chain_calls = df_full_chain_calls[~pd.isna(df_full_chain_calls.volume)]
                df_full_chain_calls = df_full_chain_calls[~pd.isna(df_full_chain_calls.openInterest)]
                df_full_chain_calls = df_full_chain_calls[df_full_chain_calls.openInterest >= oi_min]

                if not df_full_chain_calls.empty:
                    df_full_chain_calls['unusual_activity'] = df_full_chain_calls.volume / df_full_chain_calls.openInterest

                    df_full_chain_calls = df_full_chain_calls.sort_values('unusual_activity', ascending=False)
                    df_full_chain_calls = df_full_chain_calls[df_full_chain_calls.unusual_activity > 1]
                    df_full_chain_calls = df_full_chain_calls[df_full_chain_calls.inTheMoney==show_itm]
                    df_full_chain_calls['spread'] = df_full_chain_calls.ask - df_full_chain_calls.bid
                    
                    if not df_full_chain_calls.empty:
                        df_full_chain_calls = df_full_chain_calls[['contractSymbol', 'strike',
                                                                'lastPrice','spread','percentChange',
                                                                    'volume','openInterest','impliedVolatility',
                                                                    'unusual_activity']]

                        def colorize_rows(row):
                            if df_full_chain_calls["unusual_activity"].max() == df_full_chain_calls["unusual_activity"].min():
                                norm = 0.5
                            else:
                                norm = (row.unusual_activity - df_full_chain_calls["unusual_activity"].min()) / \
                                    (df_full_chain_calls["unusual_activity"].max() - df_full_chain_calls["unusual_activity"].min())
                            color = f'background-color: rgba({255 * (1 - norm)}, {255 * norm}, 0, 0.5)'
                            return [color] * len(row)

                        # Apply styling to whole row
                        df_full_chain_calls = df_full_chain_calls.reset_index(drop=True)
                        styled_df = df_full_chain_calls.style.apply(colorize_rows, axis=1)
                        st.dataframe(styled_df, key="calls_unusual_activity_df")
                    else:
                        st.info("No calls with unusual activity matching the criteria.")
                else:
                    st.info("No calls with sufficient open interest.")
            else:
                st.info("No call options data available.")

        with col_activity[1]:
            oi_min_puts = st.number_input("Minumum OI", min_value=1, value=1_000,
                                        help='Minumum Open Interest to consider when computing unusual options activity.',
                                        key="oi_min_puts_input")
            show_itm_puts = st.checkbox("Show ITM", value=False,
                                        help='Only show in-the-money (ITM) contracts, otherwise show only out-of-money.',
                                        key="show_itm_puts_checkbox")

            if not df_full_chain_puts.empty:
                df_full_chain_puts = df_full_chain_puts[df_full_chain_puts.volume != 0.]
                df_full_chain_puts = df_full_chain_puts[df_full_chain_puts.openInterest != 0.]
                df_full_chain_puts = df_full_chain_puts[~pd.isna(df_full_chain_puts.volume)]
                df_full_chain_puts = df_full_chain_puts[~pd.isna(df_full_chain_puts.openInterest)]
                df_full_chain_puts = df_full_chain_puts[df_full_chain_puts.openInterest >= oi_min_puts]
                
                if not df_full_chain_puts.empty:
                    df_full_chain_puts['unusual_activity'] = df_full_chain_puts.volume / df_full_chain_puts.openInterest
                    df_full_chain_puts = df_full_chain_puts.sort_values('unusual_activity', ascending=False)
                    df_full_chain_puts = df_full_chain_puts[df_full_chain_puts.unusual_activity > 1]
                    df_full_chain_puts = df_full_chain_puts[df_full_chain_puts.inTheMoney==show_itm_puts]
                    df_full_chain_puts['spread'] = df_full_chain_puts.ask - df_full_chain_puts.bid
                    
                    if not df_full_chain_puts.empty:
                        df_full_chain_puts = df_full_chain_puts[['contractSymbol', 'strike',
                                                                'lastPrice','spread','percentChange',
                                                                'volume','openInterest',
                                                                'impliedVolatility','unusual_activity']]

                        def colorize_rows_puts(row):
                            if df_full_chain_puts["unusual_activity"].max() == df_full_chain_puts["unusual_activity"].min():
                                norm = 0.5
                            else:
                                norm = (row.unusual_activity - df_full_chain_puts["unusual_activity"].min()) / \
                                    (df_full_chain_puts["unusual_activity"].max() - df_full_chain_puts["unusual_activity"].min())
                            color = f'background-color: rgba({255 * (1 - norm)}, {255 * norm}, 0, 0.5)'
                            return [color] * len(row)

                        # Apply styling to whole row
                        df_full_chain_puts = df_full_chain_puts.reset_index(drop=True)
                        styled_df_puts = df_full_chain_puts.style.apply(colorize_rows_puts, axis=1)
                        st.dataframe(styled_df_puts, key="puts_unusual_activity_df")
                    else:
                        st.info("No puts with unusual activity matching the criteria.")
                else:
                    st.info("No puts with sufficient open interest.")
            else:
                st.info("No put options data available.")

        st.divider()

        if expiration_dates:
            exp_date = st.selectbox(
                    "Select an expiration date",
                    expiration_dates,
                    key="expiration_date_select"
            )

            opt = yfticker.option_chain(exp_date)
            calls = opt.calls
            puts = opt.puts

            calls = calls.sort_values(by='strike')
            puts = puts.sort_values(by='strike')

            ###########
            # create widget
            tradingview_widget = """
                    <div class="tradingview-widget-container">
                        <div id="tradingview_chart"></div>
                        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                        <script type="text/javascript">
                            new TradingView.widget({
                                "width": "100%",
                                "height": 400,
                                "symbol": "NASDAQ:""" + ticker_input + """",
                                "interval": "1",
                                "timezone": "Etc/UTC",
                                "theme": "dark",
                                "style": "1",
                                "locale": "en",
                                "toolbar_bg": "#f1f3f6",
                                "enable_publishing": false,
                                "hide_top_toolbar": false,
                                "save_image": false,
                                "container_id": "tradingview_chart"
                            });
                        </script>
                    </div>
                    """

            col_inner = st.columns((4,4), gap='small')
            
            try:
                ATM = opt.underlying['regularMarketPrice']
                span_end = int((calls.strike-ATM).abs().argmin())

                with col_inner[0]:
                    if 'openInterest' in calls.columns and 'openInterest' in puts.columns:
                        max_oi = np.maximum(calls.openInterest.values.max() if len(calls) > 0 else 0, 
                                            puts.openInterest.values.max() if len(puts) > 0 else 0)

                        fig = go.Figure()
                        
                        if len(puts) > 0:
                            fig.add_trace(go.Bar(
                                x=puts['strike'],
                                y=puts['openInterest'].values,
                                name='Puts',
                                orientation='v',
                                marker_color='#D9534F'
                            ))

                        if len(calls) > 0:
                            fig.add_trace(go.Bar(
                                x=calls['strike'],
                                y=calls['openInterest'],
                                name='Calls',
                                orientation='v',
                                marker_color='#00C66B'
                            ))
                            
                        # Add vertical span using layout shapes (highlight region)
                        fig.add_shape(
                            type="rect",
                            x0=0, x1=ATM,
                            y0=0, y1=max_oi,
                            fillcolor="#B39DDB",
                            opacity=0.15,
                            line_width=0,
                            name='ITM Level'
                        )

                        fig.update_layout(
                            title="Open Interest by Strike ($)",
                            xaxis_title="Strike Price ($)",
                            yaxis_title="Open Interest",
                            barmode="overlay",
                            template="plotly_dark",
                            bargap=0.01,
                            bargroupgap=0.01,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Open interest data not available for this expiration date.")

                with col_inner[1]:
                    if 'volume' in calls.columns and 'volume' in puts.columns:
                        max_vol = np.maximum(calls.volume.values.max() if len(calls) > 0 else 0, 
                                            puts.volume.values.max() if len(puts) > 0 else 0)

                        fig2 = go.Figure()
                        
                        if len(puts) > 0:
                            fig2.add_trace(go.Bar(
                                x=puts['strike'],
                                y=puts['volume'],
                                name='Puts',
                                orientation='v',
                                marker_color='#D9534F'
                            ))

                        if len(calls) > 0:
                            fig2.add_trace(go.Bar(
                                x=calls['strike'],
                                y=calls['volume'],
                                name='Calls',
                                orientation='v',
                                marker_color='#00C66B'
                            ))

                        fig2.add_shape(
                            type="rect",
                            x0=0, x1=ATM,
                            y0=0, y1=max_vol,
                            fillcolor="#B39DDB",
                            opacity=0.15,
                            line_width=0,
                            name='ITM Level'
                        )

                        fig2.update_layout(
                            title="Volume by Strike ($)",
                            xaxis_title="Strike Price ($)",
                            yaxis_title="Volume",
                            barmode="overlay",
                            template="plotly_dark",
                            bargap=0.01,
                            bargroupgap=0.01,
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("Volume data not available for this expiration date.")

                ################################################

                col_vol = st.columns((4,4), gap='small')

                with col_vol[0]:
                    if 'impliedVolatility' in calls.columns and 'impliedVolatility' in puts.columns:
                        # plot IV bar chart (overlap calls and puts)
                        fig = go.Figure()
                        
                        if len(calls) > 0:
                            fig.add_trace(go.Scatter(
                                x=calls.strike, 
                                y=calls['impliedVolatility']*100,
                                mode='lines+markers', 
                                name='Call IV', 
                                line=dict(color='#00C66B')
                            ))
                            
                        if len(puts) > 0:
                            fig.add_trace(go.Scatter(
                                x=puts.strike, 
                                y=puts['impliedVolatility']*100,
                                mode='lines+markers', 
                                name='Put IV', 
                                line=dict(color='#D9534F')
                            ))
                            
                        fig.update_layout(
                            title="Implied Volatility (%) by Strike ($); 'Volatility Smile'",
                            xaxis_title="Strike Price ($)", 
                            yaxis_title="IV (%)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Implied volatility data not available for this expiration date.")

                with col_vol[1]:
                    if 'impliedVolatility' in calls.columns:
                        # volatility surface
                        try:
                            xs, ys, zs = [], [], []
                            for e in expiration_dates:
                                xs.append(e)
                                
                                opt_e = yfticker.option_chain(e)
                                calls_e = opt_e.calls
                                
                                if len(calls_e) > 0:
                                    ys.append(list(calls_e.strike.values))
                                    zs.append(list(calls_e['impliedVolatility'].values * 100.))
                            
                            if xs and ys and zs:
                                # Create mesh grid for surface plot
                                x_mesh, y_mesh = np.meshgrid(np.arange(len(xs)), np.arange(len(ys[0])))
                                z_mesh = np.zeros_like(x_mesh, dtype=float)
                                
                                for i in range(len(xs)):
                                    for j in range(len(ys[0])):
                                        if j < len(zs[i]):
                                            z_mesh[j, i] = zs[i][j]
                                
                                fig = go.Figure(data=[go.Surface(z=z_mesh, x=x_mesh, y=y_mesh)])
                                fig.update_layout(
                                    title=dict(text='Volatility Surface'), 
                                    autosize=False,
                                    width=500, 
                                    height=500,
                                    margin=dict(l=65, r=50, b=65, t=90),
                                    scene=dict(
                                        xaxis_title='Expiration Date',
                                        yaxis_title='Strike Price',
                                        zaxis_title='IV (%)',
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Insufficient data for volatility surface visualization.")
                        except Exception as e:
                            st.error(f"Error creating volatility surface: {str(e)}")
                    else:
                        st.info("Implied volatility data not available for volatility surface.")

                # show price plot
                st.divider()
                st.components.v1.html(tradingview_widget, height=400)
                
            except Exception as e:
                st.error(f"Error processing options data: {str(e)}")
        else:
            st.warning("No options expiration dates available for this ticker.")
            
elif st.session_state['current_app'] == "news":
    #############################################
    # NEWS ANALYZER APPLICATION
    #############################################
    st.markdown('<div class="main-header">Financial Analysis Dashboard</div>', unsafe_allow_html=True)
    # App title and description
    st.markdown("News Analyzer with AI")
    st.markdown("""
    This app fetches news articles, analyzes them with AI, and answers your questions.
    Use the form below to fetch news on a specific topic, then ask questions to analyze the content.
    """)
    
    # Set up the tabs
    tab1, tab2 = st.tabs(["Fetch News", "Ask Questions"])
    
    # Tab 1: Fetch News
    with tab1:
        st.header("Fetch News Articles")
        
        # Import necessary modules
        try:
            from news.news_core import fetch_news, process_news_articles_whole
            from news.embedding_utils import (
                create_pinecone_index, 
                clear_pinecone_index, 
                upload_news_to_pinecone, 
                preprocess_query
            )
            
            # Form for fetching news
            with st.form(key="news_fetch_form"):
                # News query parameters
                query = st.text_input("Search Query (required)", "artificial intelligence", key="news_search_query")
                
                col1, col2 = st.columns(2)
                with col1:
                    category = st.selectbox(
                        "Category (for top headlines)",
                        [None, "business", "entertainment", "general", "health", "science", "sports", "technology"],
                        index=0,
                        key="news_category_select"
                    )
                    language = st.selectbox(
                        "Language", 
                        ["en", "es", "fr", "de", "it"], 
                        index=0,
                        key="news_language_select"
                    )
                    
                with col2:
                    from_date = st.date_input(
                        "From Date", 
                        datetime.now() - timedelta(days=7),
                        key="news_from_date"
                    )
                    to_date = st.date_input(
                        "To Date", 
                        datetime.now(),
                        key="news_to_date"
                    )
                    
                sources = st.text_input("News Sources (comma-separated)", "", key="news_sources_input")
                
                # Submit button
                fetch_submit = st.form_submit_button("Fetch News")
            
            # Process form submission (outside the form)
            if fetch_submit:
                if not query:
                    st.error("Search query is required.")
                else:
                    # Fetch news articles
                    with st.spinner(f"Fetching news for query: '{query}'..."):
                        try:
                            articles = fetch_news(
                                query=query,
                                sources=sources if sources else None,
                                from_date=from_date.strftime('%Y-%m-%d'),
                                to_date=to_date.strftime('%Y-%m-%d'),
                                language=language,
                                category=category
                            )
                            
                            # Store in session state
                            st.session_state['fetched_articles'] = articles
                            st.session_state['current_page'] = 1
                            
                            if not articles:
                                st.warning("No articles found.")
                            else:
                                st.success(f"Fetched {len(articles)} news articles.")
                        except Exception as e:
                            st.error(f"Error fetching news: {str(e)}")
            
            # Display fetched articles (always show if available)
            if st.session_state['fetched_articles']:
                st.subheader("Fetched News Articles")
                
                # Pagination
                articles = st.session_state['fetched_articles']
                total_articles = len(articles)
                articles_per_page = 10
                total_pages = max(1, math.ceil(total_articles / articles_per_page))
                
                # Navigation buttons
                col1, col2, col3 = st.columns([2, 3, 2])
                
                with col1:
                    prev_disabled = st.session_state['current_page'] <= 1
                    if st.button("← Previous", disabled=prev_disabled, key="prev_btn", on_click=change_page, args=("prev",)):
                        pass
                
                with col2:
                    st.markdown(f"<div style='text-align: center'>Page {st.session_state['current_page']} of {total_pages}</div>", 
                              unsafe_allow_html=True)
                
                with col3:
                    next_disabled = st.session_state['current_page'] >= total_pages
                    if st.button("Next →", disabled=next_disabled, key="next_btn", on_click=change_page, args=("next",)):
                        pass
                
                # Current page content
                current_page = st.session_state['current_page']
                start_idx = (current_page - 1) * articles_per_page
                end_idx = min(start_idx + articles_per_page, total_articles)
                
                # Display articles
                for i in range(start_idx, end_idx):
                    try:
                        article = articles[i]
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            # Image
                            if article.get('urlToImage'):
                                try:
                                    st.image(article['urlToImage'], use_container_width=True)
                                except:
                                    st.image("https://via.placeholder.com/150x100?text=Image+Error", use_container_width=True)
                            else:
                                st.image("https://via.placeholder.com/150x100?text=No+Image", use_container_width=True)
                        
                        with col2:
                            # Article details
                            st.markdown(f"### {article.get('title', 'No Title')}")
                            
                            source_name = "Unknown Source"
                            if 'source' in article and isinstance(article['source'], dict):
                                source_name = article['source'].get('name', 'Unknown Source')
                            
                            published = article.get('publishedAt', 'Unknown Date')
                            st.markdown(f"**Source:** {source_name} | **Published:** {published}")
                            st.markdown(article.get('description', 'No description available'))
                            
                            if article.get('url'):
                                st.markdown(f"[Read more]({article['url']})")
                        
                        st.divider()
                    except Exception as e:
                        st.error(f"Error displaying article {i}: {str(e)}")
                
                st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_articles} articles")
                
                # Vector DB upload section
                st.subheader("Store In Vector Database")
                st.write("Process the fetched articles and store them in the vector database for querying.")
                
                storage_method = st.radio(
                    "Storage Method",
                    ["Whole Articles (Recommended)", "Chunked Articles (Legacy)"],
                    index=0,
                    help="Whole Articles: Store each article as a complete document (better for news). Chunked: Split articles into smaller chunks (better for long documents).",
                    key="storage_method_radio"
                )
                
                if st.button("Process and Store in Vector Database", key="process_store_btn"):
                    try:
                        # Process articles based on selected method
                        with st.spinner("Processing articles..."):
                            if storage_method == "Whole Articles (Recommended)":
                                chunks = process_news_articles_whole(articles)
                                st.success(f"Created {len(chunks)} documents (one per article).")
                            else:
                                # We'll use the whole article method since chunk method isn't available
                                chunks = process_news_articles_whole(articles)
                                st.success(f"Created {len(chunks)} text chunks from the articles.")
                        
                        # Set up Vector DB
                        with st.spinner("Setting up vector database..."):
                            message = create_pinecone_index(
                                pinecone_api_key=PINECONE_API_KEY,
                                index_name=index_name
                            )
                            st.info(message)
                        
                        # Clear existing data
                        with st.spinner("Clearing previous data from vector database..."):
                            message = clear_pinecone_index(PINECONE_API_KEY, index_name)
                            st.info(message)
                        
                        # Upload to Vector DB
                        with st.spinner("Uploading to vector database..."):
                            message = upload_news_to_pinecone(
                                document_chunks=chunks,
                                pinecone_api_key=PINECONE_API_KEY,
                                index_name=index_name,
                                model_name=embedding_model
                            )
                            st.success(message)
                        
                        st.success("✅ News articles have been processed and stored successfully!")
                        st.info("🔍 Now you can go to the 'Ask Questions' tab to analyze these articles!")
                    
                    except Exception as e:
                        st.error(f"Error processing and storing articles: {str(e)}")
        except ImportError as e:
            st.error(f"Error: Could not import required modules: {str(e)}")
            st.error("Please check that all required News Analyzer modules are installed and properly configured.")
    
    # Tab 2: Ask Questions
    with tab2:
        st.header("Ask Questions About the News")
        
        # Display chat history
        for i, (question, answer, sources) in enumerate(st.session_state['chat_history']):
            st.chat_message("user").write(question)
            
            with st.chat_message("assistant"):
                st.write(answer)
                
                # Display sources if available
                if sources:
                    with st.expander("View Sources", expanded=False):
                        for j, src in enumerate(sources):
                            st.markdown(f"**Source {j+1}:** {src['title']} ({src['source']})")
                            st.markdown(f"Relevance: {src['score']:.4f}")
                            if 'url' in src and src['url']:
                                st.markdown(f"[Read more]({src['url']})")
                            st.divider()
        
        # Handle new questions
        user_question = st.chat_input("Ask a question about the news")
        
        if user_question:
            # Show user question
            st.chat_message("user").write(user_question)
            
            # Process and show answer
            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                answer_placeholder.info("Searching for relevant articles...")
                
                if retriever is None:
                    answer_placeholder.error("Retriever is not available. Please check your Pinecone API key and connection.")
                    sources = []
                    answer = "I couldn't access the news database. Please make sure you've fetched news articles in the 'Fetch News' tab and check your API keys."
                else:
                    try:
                        # Show query processing if enabled
                        if show_query_processing:
                            enhanced_query, original_query = preprocess_query(user_question)
                            st.info(f"Query processing: '{original_query}' → '{enhanced_query}'")
                        
                        # Search for relevant articles
                        search_results = retriever.search(user_question, top_k=results_count)
                        
                        if not search_results:
                            answer_placeholder.warning("No relevant articles found.")
                            sources = []
                            answer = "I couldn't find any relevant information in the news articles. Try asking a different question or fetch more news articles related to your topic."
                        else:
                            sources = search_results
                            
                            if llm is None:
                                answer_placeholder.warning("LLM is not available. Showing article summaries instead.")
                                summaries = []
                                for i, result in enumerate(search_results[:3]):
                                    title = result.get('title', 'Untitled')
                                    source = result.get('source', 'Unknown')
                                    content = result.get('content_preview', result.get('text', ''))
                                    summaries.append(f"**Article {i+1}:** {title} from {source}\n\n{content[:200]}...")
                                    
                                answer = "Here are the most relevant articles I found:\n\n" + "\n\n".join(summaries)
                            else:
                                answer_placeholder.info("Generating answer based on retrieved articles...")
                                answer = llm.query(user_question, search_results)
                            
                            # Show sources
                            with st.expander("View Sources", expanded=False):
                                for i, result in enumerate(search_results):
                                    col1, col2 = st.columns([1, 4])
                                    
                                    with col1:
                                        # Score indicator
                                        score = result['score']
                                        st.progress(score)
                                        st.caption(f"Relevance: {score:.4f}")
                                    
                                    with col2:
                                        st.markdown(f"**{result['title']}**")
                                        st.caption(f"Source: {result['source']}")
                                        if 'url' in result and result['url']:
                                            st.markdown(f"[Read original article]({result['url']})")
                                    
                                    st.markdown(result.get('content_preview', result.get('text', ''))[:300] + "...")
                                    st.divider()
                                    
                    except Exception as e:
                        answer_placeholder.error(f"An error occurred: {str(e)}")
                        sources = []
                        answer = f"Sorry, I encountered an error while trying to answer your question: {str(e)}"
                
                # Display the answer
                answer_placeholder.write(answer)
            
            # Add to chat history
            st.session_state['chat_history'].append((user_question, answer, sources))

else:
    #############################################
    # OPTIONS ANALYSIS APPLICATION
    #############################################
    st.markdown('<div class="main-header">Financial Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Title and description
    st.markdown("Options Analysis Dashboard")
    st.markdown("""
    This dashboard provides comprehensive analysis of stock options, including key metrics like IV skew, Greeks, 
    max pain levels, and put/call ratios. Explore the data across multiple expiration dates and get expert insights.
    """)
    
    if not OPTIONS_MODULES_AVAILABLE:
        st.error("Options Analysis modules not available. Please check your installation.")
    else:
        # Display the options data if available
        if 'options_data' in st.session_state and st.session_state['options_data'] is not None:
            data = st.session_state['options_data']
            summary = st.session_state['options_summary']
            
            # Check for errors
            if "error" in data:
                st.error(data["error"])
            else:
                # Display fundamental data with custom formatting
                st.markdown(f'<div class="section-header">{data["ticker"]} Options Analysis ({data["analysis_date"]})</div>', unsafe_allow_html=True)
                
                # Main metrics in a 4-column grid with custom styling
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Price</div>
                        <div class="metric-value metric-value-green">${:,.2f}</div>
                    </div>
                    """.format(data['price']), unsafe_allow_html=True)
                    
                with col2:
                    # Format market cap properly
                    market_cap = format_large_number(data['market_cap'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Market Cap</div>
                        <div class="metric-value">{market_cap}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Beta</div>
                        <div class="metric-value metric-value-amber">{data['beta']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col4:
                    pcr_color = "metric-value-green" if data['put_call_ratio'] < 1 else "metric-value-red"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Put/Call Ratio</div>
                        <div class="metric-value {pcr_color}">{data['put_call_ratio']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional metrics in a 3-column grid
                st.markdown('<div class="section-header">Additional Information</div>', unsafe_allow_html=True)
                
                fund_col1, fund_col2, fund_col3 = st.columns(3)
                
                with fund_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">52-Week High</div>
                        <div class="metric-value metric-value-green">{data['fifty_two_week_high']}</div>
                    </div>
                    <br>
                    <div class="metric-card">
                        <div class="metric-label">52-Week Low</div>
                        <div class="metric-value metric-value-red">{data['fifty_two_week_low']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with fund_col2:
                    dividend = data['dividend_yield']
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Dividend Yield</div>
                        <div class="metric-value metric-value-green">{dividend}</div>
                    </div>
                    <br>
                    <div class="metric-card">
                        <div class="metric-label">Next Earnings</div>
                        <div class="metric-value">{data['next_earnings']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with fund_col3:
                    if "historical_volatility" in data and data["historical_volatility"] is not None:
                        hv = data["historical_volatility"]
                        hv_color = "metric-value-amber" if hv > 20 else "metric-value-blue"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Historical Volatility (1y)</div>
                            <div class="metric-value {hv_color}">{hv}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show options data for each expiration
                st.markdown('<div class="section-header">Options Expirations</div>', unsafe_allow_html=True)
                
                # Create tabs for each expiration date
                if data["expirations"]:
                    expiration_dates = list(data["expirations"].keys())
                    tabs = st.tabs(expiration_dates)
                    
                    for i, exp in enumerate(expiration_dates):
                        exp_data = data["expirations"][exp]
                        with tabs[i]:
                            # Expiration header with DTE
                            st.markdown(f"""
                            <div class="expiry-header">
                                <h3>Expiration: {exp} <span style="color: #9E9E9E;">(DTE: {exp_data['days_to_expiry']})</span></h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Key metrics
                            metrics_col1, metrics_col2 = st.columns(2)
                            
                            with metrics_col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Max Pain Strike</div>
                                    <div class="metric-value metric-value-blue">${exp_data['max_pain_strike']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with metrics_col2:
                                if exp_data["iv_skew_calls"]:
                                    skew = exp_data["iv_skew_calls"]
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">IV Skew (Calls)</div>
                                        <div style="display: flex; justify-content: space-between;">
                                            <div><span style="color: #9E9E9E;">ITM:</span> <span style="color: #4CAF50;">{skew['itm']}%</span></div>
                                            <div><span style="color: #9E9E9E;">ATM:</span> <span style="color: #FFC107;">{skew['atm']}%</span></div>
                                            <div><span style="color: #9E9E9E;">OTM:</span> <span style="color: #F44336;">{skew['otm']}%</span></div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Calls and Puts tables
                            calls_col, puts_col = st.columns(2)
                            
                            with calls_col:
                                st.markdown('<h4 style="text-align: center; color: #4CAF50;">Calls (Near ATM)</h4>', unsafe_allow_html=True)
                                if exp_data["calls"]:
                                    calls_df = pd.DataFrame(exp_data["calls"])
                                    
                                    # Format columns
                                    calls_df['iv'] = calls_df['iv'].apply(lambda x: f"{x}%")
                                    calls_df['delta'] = calls_df['delta'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                                    calls_df['gamma'] = calls_df['gamma'].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
                                    calls_df['theta'] = calls_df['theta'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                                    calls_df['vega'] = calls_df['vega'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                                    
                                    # Display dataframe with better formatting
                                    st.dataframe(
                                        calls_df,
                                        column_config={
                                            "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                                            "moneyness": st.column_config.TextColumn("Moneyness"),
                                            "last": st.column_config.NumberColumn("Last", format="$%.2f"),
                                            "bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
                                            "ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
                                            "iv": st.column_config.TextColumn("IV"),
                                            "delta": st.column_config.TextColumn("Delta"),
                                            "gamma": st.column_config.TextColumn("Gamma"),
                                            "theta": st.column_config.TextColumn("Theta"),
                                            "vega": st.column_config.TextColumn("Vega"),
                                            "open_interest": st.column_config.NumberColumn("Open Int"),
                                            "volume": st.column_config.NumberColumn("Volume")
                                        },
                                        use_container_width=True
                                    )
                                else:
                                    st.info("No call options data available")
                            
                            with puts_col:
                                st.markdown('<h4 style="text-align: center; color: #F44336;">Puts (Near ATM)</h4>', unsafe_allow_html=True)
                                if exp_data["puts"]:
                                    puts_df = pd.DataFrame(exp_data["puts"])
                                    
                                    # Format columns
                                    puts_df['iv'] = puts_df['iv'].apply(lambda x: f"{x}%")
                                    puts_df['delta'] = puts_df['delta'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                                    puts_df['gamma'] = puts_df['gamma'].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
                                    puts_df['theta'] = puts_df['theta'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                                    puts_df['vega'] = puts_df['vega'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                                    
                                    # Display dataframe with better formatting
                                    st.dataframe(
                                        puts_df,
                                        column_config={
                                            "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                                            "moneyness": st.column_config.TextColumn("Moneyness"),
                                            "last": st.column_config.NumberColumn("Last", format="$%.2f"),
                                            "bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
                                            "ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
                                            "iv": st.column_config.TextColumn("IV"),
                                            "delta": st.column_config.TextColumn("Delta"),
                                            "gamma": st.column_config.TextColumn("Gamma"),
                                            "theta": st.column_config.TextColumn("Theta"),
                                            "vega": st.column_config.TextColumn("Vega"),
                                            "open_interest": st.column_config.NumberColumn("Open Int"),
                                            "volume": st.column_config.NumberColumn("Volume")
                                        },
                                        use_container_width=True
                                    )
                                else:
                                    st.info("No put options data available")
                                    
                # Options Strategy Advisor Section
                st.markdown('<div class="section-header">Options Strategy Advisor</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="analyst-section">
                    <div class="analyst-header">Ask our AI Options Strategist</div>
                    <p>Ask a question about the options data to get expert insights and potential strategies.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Strategy Advisor form
                with st.form(key="llm_form"):
                    query = st.text_area(
                        "Your question:",
                        height=100,
                        placeholder="Example: Based on this options data, what trades would you recommend? Suggest specific option trades with strikes and expiration dates.",
                        key="options_query_textarea"
                    )
                    # Increase token limit to ensure complete JSON
                    submit_button = st.form_submit_button("Let's analyze!")

                if submit_button and query:
                    with st.spinner("Analyzing options data..."):
                        try:
                            # Get LLM analysis with structured JSON orders
                            llm_response = ask_llm_about_options(summary, query, max_tokens=2000)
                            
                            # Display the analysis
                            st.markdown("""
                            <div class="analyst-section">
                                <div class="analyst-header">Analysis</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(llm_response)
                            
                        except Exception as e:
                            st.error(f"Error generating analysis: {str(e)}")
                            st.error("Make sure you have set your API key in the .env file (OPENROUTER_API_KEY or OPENAI_API_KEY)")
        else:
            # No options data loaded yet
            ticker = st.session_state.get('options_ticker', 'AAPL')
            st.info(f"Enter a ticker symbol in the sidebar (current: {ticker}) and click 'Analyze Options' to get started.")
            
            # if st.button("Analyze Now", key="analyze_main_button"):
            #     with st.spinner(f"Fetching options data for {ticker}..."):
            #         try:
            #             # Get data
            #             data = build_compact_options_json(ticker, 3, True)  # Default to 3 expirations with volatility
            #             summary = summarize_options_data(data)
                        
            #             # Save to session state
            #             st.session_state['options_data'] = data
            #             st.session_state['options_summary'] = summary
            #             st.session_state['last_ticker'] = ticker
                        
            #             st.success(f"Data retrieved for {ticker}! Refreshing page...")
            #             time.sleep(0.5)  # Small delay to show message
            #             st.rerun()
            #         except Exception as e:
            #             st.error(f"Error retrieving options data: {str(e)}")

    

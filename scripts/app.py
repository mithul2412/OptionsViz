"""
Financial Analysis Dashboard

This application provides a comprehensive financial analysis dashboard with multiple features:
- Options & Strategies visualization and analysis
- News aggregation and AI-powered analysis
- Interactive charts and market data visualization

The app is built using Streamlit and integrates various financial data sources
and visualization libraries for a complete trading research experience.

Author: 
    Mithul Raaj

Created:
    March 2025

License:
    MIT
"""

# Standard library imports
import os
import sys
import json
import math
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from PIL import Image

# Set environment variables first, before any other imports or operations
os.environ["STREAMLIT_WATCH_MODULE_PATH"] = "false"
os.environ["STREAMLIT_DISABLE_WATCHER"] = "true"

# Initialize placeholder image
PLACEHOLDER_IMAGE = None
try:
    # Try to load the image once at startup
    PLACEHOLDER_IMAGE = Image.open("img/news_icon.jpg")
    PLACEHOLDER_LOADED = True
except Exception as e:
    PLACEHOLDER_LOADED = False

# App configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the news, strategies and dataviz folders to sys.path
sys.path.append("news")
sys.path.append("strategies")
sys.path.append("dataviz")

# Load environment variables from .env file
load_dotenv()

# Module imports - wrapped in try/except to handle missing modules gracefully
try:
    from dataviz.eod_chain_wrapper import (
        plot_surface, calc_unusual_table, colorize_rows, get_options_data,
        create_open_interest_chart, create_volume_chart, create_iv_chart, 
        get_tradingview_widgets
    )
    EOD_CHAIN_AVAILABLE = True
except ImportError:
    EOD_CHAIN_AVAILABLE = False

try:
    from dataviz.options_viz_wrapper import (
        get_option_data, plot_strategy, get_stock_data,
        get_available_strategies, get_strategy_description
    )
    OPTIONS_VIZ_AVAILABLE = True
except ImportError:
    OPTIONS_VIZ_AVAILABLE = False

# Import News modules
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

# Import Options Analysis modules
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


# Constants and configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
NEWSAPI_KEY = os.getenv('NEWS_API_KEY')


def initialize_session_state():
    """
    Initialize all session state variables required by the application.
    This ensures consistent state management across app interactions.
    """
    # Core app state
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True
        st.session_state['current_app'] = "options"
    
    # Navigation state
    if 'current_app' not in st.session_state:
        st.session_state['current_app'] = "options"  # Default to options
    
    # News Analyzer state
    if 'fetched_articles' not in st.session_state:
        st.session_state['fetched_articles'] = []
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 1
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Options Analysis state
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


def change_page(direction: str) -> None:
    """
    Function to change the current page for news articles pagination.
    
    Args:
        direction (str): Direction to change the page ('next' or 'prev')
    """
    if direction == "next":
        st.session_state['current_page'] += 1
    else:
        st.session_state['current_page'] -= 1


def format_large_number(num: Union[int, float]) -> str:
    """
    Format large numbers with appropriate suffixes (K, M, B) for readability.
    
    Args:
        num (Union[int, float]): The number to format
    
    Returns:
        str: Formatted number string with appropriate suffix
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def ask_llm_about_options_safe(summary_text: str, user_query: str, max_tokens: int = 2000) -> str:
    """
    Safely call the LLM directly for options analysis with robust error handling.
    
    This function provides an interface to query an LLM about options data.
    It supports both OpenRouter and OpenAI backends, with appropriate fallbacks
    and comprehensive error handling for a production environment.
    
    Args:
        summary_text (str): Text summary of options data containing key information
                           about strikes, expiration dates, premiums, and Greeks
        user_query (str): User's question about the options data to be analyzed
        max_tokens (int, optional): Maximum tokens for the LLM response. Defaults to 2000.
    
    Returns:
        str: LLM response with analysis or a formatted error message if the process fails
        
    Raises:
        No exceptions are raised as all errors are caught and returned as messages
    """
    # Check if summary_text is None or empty
    if not summary_text:
        return "Error: No options data available. Please enter a ticker symbol first to fetch data."
        
    try:
        # Retrieve API keys with proper validation
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openrouter_api_key and not openai_api_key:
            return "Error: No API key found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY in your environment variables."
        
        # Import OpenAI here to avoid issues with missing dependencies
        try:
            from openai import OpenAI
        except ImportError:
            return "Error: OpenAI library not installed. Please install it with: pip install openai"
        
        # Create the appropriate API client based on available keys
        client = None
        model = None
        extra_headers = {}
        
        if openrouter_api_key:
            try:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_api_key
                )
                model = "deepseek/deepseek-chat:free"
                extra_headers = {
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "OptionsAnalysisLLM"
                }
            except Exception as e:
                return f"Error initializing OpenRouter client: {str(e)}"
        elif openai_api_key:
            try:
                client = OpenAI(api_key=openai_api_key)
                model = "gpt-4"  # Default to GPT-4
            except Exception as e:
                return f"Error initializing OpenAI client: {str(e)}"
        
        # Verify client initialization succeeded
        if not client or not model:
            return "Error: Failed to initialize API client with the provided credentials."
        
        # Create system prompt with structured instructions
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
        
        # Make the API call with proper error handling
        try:
            # Configure API call based on which service we're using
            # For OpenRouter, include the extra headers
            if openrouter_api_key:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers
                )
            else:
                # For OpenAI, use standard parameters
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens
                )
            
            # Extract the response text safely with thorough validation
            if completion and hasattr(completion, 'choices') and len(completion.choices) > 0:
                if hasattr(completion.choices[0], 'message') and hasattr(completion.choices[0].message, 'content'):
                    response_text = completion.choices[0].message.content
                    return response_text
                else:
                    return "Error: Unable to extract content from the API response."
            else:
                return "Error: Received an empty or invalid response from the API."
            
        except Exception as api_error:
            # Handle specific API call errors with actionable feedback
            error_msg = str(api_error)
            if "rate limit" in error_msg.lower():
                return "Error: Rate limit exceeded. Please try again in a few moments."
            elif "context length" in error_msg.lower():
                return "Error: The options data is too large for the model's context window. Try analyzing fewer expiration dates."
            else:
                return f"Error: Could not generate a response: {error_msg}"

    except Exception as outer_e:
        # Catch any unexpected errors in the outer scope
        return f"Error in LLM communication: {str(outer_e)}"

@st.cache_resource
def get_retriever(index_name: str, model_name: str) -> Optional[NewsRetriever]:
    """
    Initialize and cache the news retriever to avoid recreating it on every rerun.
    
    Args:
        index_name (str): Name of the Pinecone index to use
        model_name (str): Name of the embedding model to use
    
    Returns:
        Optional[NewsRetriever]: Initialized retriever or None if initialization failed
    """
    try:
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
def get_llm(model_name: str) -> Optional[NewsLLM]:
    """
    Initialize and cache the LLM interface to avoid recreating it on every rerun.
    
    Args:
        model_name (str): Name of the LLM model to use
    
    Returns:
        Optional[NewsLLM]: Initialized LLM interface or None if initialization failed
    """
    try:
        llm = NewsLLM(
            api_key=OPENROUTER_API_KEY,
            model=model_name
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM interface: {str(e)}")
        return None
   
def render_custom_css() -> None:
    """
    Apply custom CSS styling to the application.
    This improves the visual appearance and user experience.
    """
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
        .tab-content {
            padding: 20px 0;
        }
        .widget-container {
            margin-bottom: 20px;
        }
        .full-width-select {
            width: 100% !important;
        }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar() -> tuple:
    """
    Render the application sidebar with navigation buttons and app-specific settings.
    
    Returns:
        tuple: Optional (retriever, llm) tuple for news app
    """
    # App selector in the sidebar - ALWAYS show these navigation buttons
    st.sidebar.button("Options & Strategies", key="options_btn", 
                     help="Options chain, analysis, and strategies",
                     use_container_width=True, 
                     type="primary" if st.session_state['current_app'] == "options" else "secondary",
                     on_click=lambda: st.session_state.update({"current_app": "options"}))

    st.sidebar.button("News Analyzer", key="news_btn", 
                     help="Analyze news articles with AI",
                     use_container_width=True, 
                     type="primary" if st.session_state['current_app'] == "news" else "secondary",
                     on_click=lambda: st.session_state.update({"current_app": "news"}))

    st.sidebar.divider()

    # App-specific sidebar options - these change based on the current app
    retriever, llm = None, None
    if st.session_state['current_app'] == "options":
        render_options_sidebar()
    elif st.session_state['current_app'] == "news":
        retriever, llm = render_news_sidebar()
        
    return retriever, llm


def render_options_sidebar() -> None:
    """
    Render the sidebar for the Options & Strategies application.
    """
    st.sidebar.header("Options Info")
    
    ticker_input = st.sidebar.text_input(
        "Enter stock ticker:", 
        value="", 
        placeholder='e.g. NVDA, AAPL, AMZN', 
        key="options_ticker_sidebar_input"
    )
    
    if ticker_input:
        ticker_input = ticker_input.upper()
        st.session_state['options_ticker'] = ticker_input
    
    # TradingView widget for the sidebar only if ticker is provided
    if ticker_input and EOD_CHAIN_AVAILABLE:
        try:
            # Get TradingView widgets HTML
            widgets = get_tradingview_widgets(ticker_input)
            if len(widgets) >= 3:
                _, tech_perf, _ = widgets[0], widgets[1], widgets[2]
                
                # Place technical analysis widget in sidebar
                with st.sidebar:
                    st.components.v1.html(tech_perf, height=400)
            else:
                st.sidebar.error("Failed to load TradingView widgets. Not enough widgets returned.")
        except Exception as e:
            st.sidebar.error(f"Error getting TradingView widgets: {str(e)}")


def render_news_sidebar() -> tuple:
    """
    Render the sidebar for the News Analyzer application.
    
    Returns:
        tuple: (retriever, llm) - The initialized news retriever and LLM interface
    """
    st.sidebar.header("News Analyzer Settings")
    
    # Pinecone settings with unique keys
    index_name = st.sidebar.text_input("Pinecone Index Name", "newsdata", key="news_index_name")
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["intfloat/e5-large-v2", "intfloat/e5-base-v2", "BAAI/bge-large-en-v1.5", "BAAI/bge-base-v1.5"],
        index=0,
        key="news_embedding_model"
    )

    # LLM settings
    llm_model = st.sidebar.selectbox(
        "LLM Model",
        ["deepseek/deepseek-chat:free", "google/gemini-pro", "anthropic/claude-3-sonnet", "meta-llama/llama-3-8b-instruct"],
        index=0,
        key="news_llm_model"
    )

    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        show_query_processing = st.checkbox("Show Query Processing", value=False, key="news_show_query")
        results_count = st.slider("Number of Results", min_value=3, max_value=20, value=10, key="news_results_count")
    
    # Initialize News services
    retriever = get_retriever(index_name, embedding_model)
    llm = get_llm(llm_model)
    
    # API key warnings
    if not PINECONE_API_KEY:
        st.sidebar.error("PINECONE_API_KEY not set. Required for vector database.")
    if not OPENROUTER_API_KEY:
        st.sidebar.error("OPENROUTER_API_KEY not set. Required for AI analysis.")
    if not NEWSAPI_KEY:
        st.sidebar.error("NEWS_API_KEY not set. Required for fetching news.")
        
    return retriever, llm


def render_eod_chain_tab() -> None:
    """
    Render the EOD Chain Analysis tab content.
    This includes options chain visualization, unusual activity, volatility surface, etc.
    """
    ticker_input = st.session_state.get('options_ticker', "")
    
    if not ticker_input:
        st.info("👈 Please enter a ticker symbol in the sidebar to analyze the options chain.")
    else:
        if EOD_CHAIN_AVAILABLE:
            try:
                # Get all TradingView widgets
                widgets = get_tradingview_widgets(ticker_input)
                
                if len(widgets) >= 10:  # Make sure all widgets are loaded
                    _, _, tv_advanced_plot = widgets[0], widgets[1], widgets[2]
                    symbol_info = widgets[3]
                    financial_info = widgets[4]
                    company_profile = widgets[5]
                    
                    # Display symbol info widget at the top
                    st.components.v1.html(symbol_info, height=180)  # Increased height to avoid cutoff
                    
                    # Company Profile in dropdown/expander
                    with st.expander("Company Information", expanded=False):
                        st.components.v1.html(company_profile, height=300)

                    # Financial Info in dropdown/expander
                    with st.expander("Financial Information", expanded=False):
                        st.components.v1.html(financial_info, height=500)

                    st.divider()  # Add separator line
                
                # Get options data using our wrapper functions
                expiration_dates, df_full_chain_calls, df_full_chain_puts, df_full_chain_calls_dict, df_full_chain_puts_dict, underlying_price = get_options_data(ticker_input)

                if expiration_dates:
                    # Unusual activity section
                    st.write("#### Unusual Options Activity")

                    col_activity = st.columns((4,4), gap='small')
                    with col_activity[0]:
                        st.write("#### Calls")
                        oi_min_calls = st.number_input("Minimum OI", min_value=1, value=1_000,
                                            help='Minimum Open Interest to consider when computing unusual options activity.',
                                            key="oi_min_calls_input")
                        show_itm_calls = st.checkbox("Show ITM", value=False,
                                        help='Only show in-the-money (ITM) contracts, otherwise show only out-of-money.',
                                        key="show_itm_calls_checkbox")

                        if df_full_chain_calls is not None and not df_full_chain_calls.empty:
                            df_full_chain_calls_proc = calc_unusual_table(df_full_chain_calls, show_itm_calls, oi_min_calls)

                            if not df_full_chain_calls_proc.empty:
                                styled_df_calls = df_full_chain_calls_proc.style.apply(
                                    lambda row: colorize_rows(row, df_full_chain_calls_proc), axis=1)
                                st.dataframe(styled_df_calls, key="calls_unusual_activity_df")
                            else:
                                st.info("No calls with unusual activity matching the criteria.")
                        else:
                            st.info("No call options data available.")

                    with col_activity[1]:
                        st.write("#### Puts")
                        oi_min_puts = st.number_input("Minimum OI", min_value=1, value=1_000,
                                            help='Minimum Open Interest to consider when computing unusual options activity.',
                                            key="oi_min_puts_input")
                        show_itm_puts = st.checkbox("Show ITM", value=False,
                                            help='Only show in-the-money (ITM) contracts, otherwise show only out-of-money.',
                                            key="show_itm_puts_checkbox")
                        
                        if df_full_chain_puts is not None and not df_full_chain_puts.empty:
                            df_full_chain_puts_proc = calc_unusual_table(df_full_chain_puts, show_itm_puts, oi_min_puts)
                            
                            if not df_full_chain_puts_proc.empty:
                                styled_df_puts = df_full_chain_puts_proc.style.apply(
                                    lambda row: colorize_rows(row, df_full_chain_puts_proc), axis=1)
                                st.dataframe(styled_df_puts, key="puts_unusual_activity_df")
                            else:
                                st.info("No puts with unusual activity matching the criteria.")
                        else:
                            st.info("No put options data available.")

                    st.divider()
                    st.write("#### Chain Analysis")

                    # Expiration date selector
                    exp_date = st.selectbox(
                            "Select an expiration date",
                            expiration_dates,
                            key="expiration_date_select"
                    )

                    # Get option chain for selected expiration
                    if exp_date in df_full_chain_calls_dict and exp_date in df_full_chain_puts_dict:
                        calls = df_full_chain_calls_dict[exp_date]
                        puts = df_full_chain_puts_dict[exp_date]
                        
                        calls = calls.sort_values(by='strike')
                        puts = puts.sort_values(by='strike')

                        # Use underlying price or fallback to mid-point of available strikes
                        ATM = underlying_price if underlying_price else None
                        if ATM is None and not calls.empty and 'strike' in calls.columns:
                            # Calculate midpoint if underlying price not available
                            strikes = sorted(calls['strike'].unique())
                            ATM = float(strikes[len(strikes) // 2])
                        elif ATM is None:
                            ATM = 100.0  # Default fallback value
                        
                        # Create advanced charts
                        col_inner = st.columns((4,4), gap='small')
                        
                        with col_inner[0]:
                            # Create open interest chart
                            oi_fig = create_open_interest_chart(calls, puts, ATM)
                            st.plotly_chart(oi_fig, use_container_width=True)
                            
                        with col_inner[1]:
                            # Create volume chart
                            vol_fig = create_volume_chart(calls, puts, ATM)
                            st.plotly_chart(vol_fig, use_container_width=True)

                        # IV charts
                        col_vol = st.columns((4,4), gap='small')
                        
                        with col_vol[0]:
                            st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

                            # Create IV chart
                            iv_fig = create_iv_chart(calls, puts)
                            st.plotly_chart(iv_fig, use_container_width=True)
                            
                        with col_vol[1]:
                            # Volatility surface plot
                            show_calls = st.checkbox("Calls", value=True, key='volatility_surface_calls',
                                                help='Show surface for calls (checked) or puts (unchecked)')
                            
                            try:
                                if show_calls:
                                    surface_fig = plot_surface(df_full_chain_calls_dict, expiration_dates)
                                else:
                                    surface_fig = plot_surface(df_full_chain_puts_dict, expiration_dates)
                                st.plotly_chart(surface_fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating volatility surface: {str(e)}")
                    else:
                        st.warning(f"No option chain data available for expiration date: {exp_date}")

                    # TradingView chart
                    st.divider()
                    st.write("#### Underlying Price Chart")
                    try:
                        if len(widgets) >= 3:
                            st.components.v1.html(tv_advanced_plot, height=400)
                        else:
                            st.error("Advanced plot widget not available.")
                    except Exception as e:
                        st.error(f"Error displaying chart: {str(e)}")
                else:
                    st.warning(f"No options expiration dates available for {ticker_input}.")
            except Exception as e:
                st.error(f"Error analyzing options chain: {str(e)}")
        else:
            st.error("EOD Chain module not available. Please check that eod_chain_wrapper.py is in the dataviz folder.")

        # Add Options Strategy Advisor Section below the EOD Chain
        st.divider()
        
        # Include options advisor settings
        if OPTIONS_MODULES_AVAILABLE:
            st.markdown('<div class="section-header">Options Strategy Advisor</div>', unsafe_allow_html=True)
            
            # Create a form with a submit button
            with st.form(key="options_analysis_form"):
                # Form elements in a single row
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    query = st.text_area(
                        label="Your question:",
                        height=120,
                        placeholder="Example: What trades would you recommend for this stock? Suggest specific option trades with strikes and expiration dates.",
                        key="options_query_textarea"
                    )
                
                with col2:
                    # Options in a dropdown
                    st.write("Analysis Settings")
                    expirations_limit = st.selectbox(
                        label="Expirations to analyze:",
                        options=[1, 2, 3, 5, 10],
                        index=2,  # Default to 3
                        key="expirations_dropdown"
                    )
                    include_hv = st.checkbox(label="Include Historical Volatility", value=True, key="include_hv_checkbox")
                
                # Submit button with more descriptive text
                analyze_submitted = st.form_submit_button(label="Fetch Data & Analyze")
            
            # Handle form submission - directly check if form was submitted
            if analyze_submitted:
                # Get values from form
                query = st.session_state.options_query_textarea
                expirations_limit = st.session_state.expirations_dropdown
                include_hv = st.session_state.include_hv_checkbox
                
                if not query:
                    st.warning("Please enter a question about options data.")
                else:
                    # Process with spinner
                    with st.spinner(f"Fetching data for {ticker_input} and analyzing..."):
                        try:
                            # Step 1: Fetch options data
                            data = build_compact_options_json(ticker_input, expirations_limit, include_hv)
                            summary = summarize_options_data(data)
                            
                            st.session_state['options_data'] = data
                            st.session_state['options_summary'] = summary
                            
                            # Step 2: Get LLM analysis with structured JSON orders
                            llm_response = ask_llm_about_options_safe(summary, query, max_tokens=2000)
                            
                            # Display the LLM response
                            st.markdown('<div class="analyst-header">Analysis Results</div>', unsafe_allow_html=True)
                            with st.container():
                                st.markdown(llm_response)

                        except Exception as e:
                            st.error(f"Error generating analysis: {str(e)}")
                            st.error("Make sure you have set your API key in the .env file (OPENROUTER_API_KEY or OPENAI_API_KEY)")
        else:
            st.error("Options Analysis modules not available.")

def render_strategy_viz_tab() -> None:
    """
    Render the Strategy Visualization tab content.
    This tab provides interactive visualization of various options strategies
    and displays information about strategy characteristics and use cases.
    """
    ticker_input = st.session_state.get('options_ticker', "")

    if EOD_CHAIN_AVAILABLE:
        try:
            # Get TradingView widgets
            widgets = get_tradingview_widgets(ticker_input)
            
            if len(widgets) >= 10:
                symbol_info = widgets[3]
                
                # Display symbol info widget at the top
                st.components.v1.html(symbol_info, height=180)
            else:
                st.error("Failed to load TradingView widgets for the Strategy tab.")
        except Exception as e:
            st.error(f"Error loading Strategy tab widgets: {str(e)}")
    else:
        st.error("Widgets are not available. Please check that eod_chain_wrapper.py is in the dataviz folder.")
        
    ticker = st.session_state['options_ticker']
    if not ticker:
        st.info("👈 Please enter a ticker symbol in the sidebar first.")
    else:
        if OPTIONS_VIZ_AVAILABLE:
            # Get available strategies
            strategies = get_available_strategies()
            selected_strategy = st.selectbox("Select a strategy", strategies, key="strat_select")
            
            # Get option data for the ticker
            atm_call_strike, _, _, _ = get_option_data(ticker)
            if atm_call_strike:

                # Strike price slider with better precision
                min_strike = max(1, round(atm_call_strike * 0.8, 1))  # Ensure minimum is positive
                max_strike = round(atm_call_strike * 1.2, 1)
                default_strike = round(atm_call_strike, 1)

                strike_price = st.slider(
                    "Select Strike Price", 
                    min_value=float(min_strike),
                    max_value=float(max_strike), 
                    value=float(default_strike),
                    step=0.5,  # Allow half-point strikes which are common
                    key="strike_slider"
                )
                
                # Plot strategy P/L diagram
                with st.spinner(f"Creating visualization for {selected_strategy}..."):
                    fig = plot_strategy(ticker, selected_strategy, strike_price)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Could not create strategy visualization for {ticker}.")
                
                # Get strategy description
                description = get_strategy_description(selected_strategy)
                if description:
                    st.markdown("#### About This Strategy")
                    st.markdown(description)
            else:
                st.error(f"Could not fetch options data for {ticker}. Please try another ticker.")
        else:
            st.error("Options Visualization module not available. Please check that options_viz_wrapper.py is in the dataviz folder.")


def render_watchlist_tab() -> None:
    """
    Render the Market Watchlist tab content.
    This tab provides market overviews, stock indices, and sector performance visualization.
    """
    ticker_input = st.session_state.get('options_ticker', "")
    
    if EOD_CHAIN_AVAILABLE:
        try:
            # Get TradingView widgets
            widgets = get_tradingview_widgets(ticker_input)
            
            if len(widgets) >= 10:
                market_overview = widgets[6]
                stock_overview = widgets[7]
                running_ticker = widgets[8]
                heatmap = widgets[9]
                
                # Display running ticker at the top - increased height for visibility
                st.components.v1.html(running_ticker, height=60)
                
                # Display market and stock overview side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Markets Overview")
                    st.components.v1.html(market_overview, height=535)
                
                with col2:
                    st.markdown("#### Stocks Overview")
                    st.components.v1.html(stock_overview, height=535)

                st.divider()  # Add separator line
                
                # Display heatmap with increased height
                st.markdown("#### Stocks Heatmap")
                st.components.v1.html(heatmap, height=485)  # Increased height for better visibility
            else:
                st.error("Failed to load TradingView widgets for the Watchlist tab.")
        except Exception as e:
            st.error(f"Error loading Watchlist widgets: {str(e)}")
    else:
        st.error("Widgets are not available. Please check that eod_chain_wrapper.py is in the dataviz folder.")

def render_news_fetch_tab() -> None:
    """
    Render the News Fetch tab content.
    This tab allows users to search for news articles, view them, and process them into a vector database.
    """
    st.markdown("#### Fetch News Articles")
    
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
            # Pagination
            articles = st.session_state['fetched_articles']
            total_articles = len(articles)
            articles_per_page = 10
            total_pages = max(1, math.ceil(total_articles / articles_per_page))
            
            # Navigation buttons with better alignment
            prev_col, page_col, next_col = st.columns([1, 3, 1])

            # Previous button on left
            with prev_col:
                prev_disabled = st.session_state['current_page'] <= 1
                st.button("← Previous", disabled=prev_disabled, key="prev_btn", 
                        on_click=change_page, args=("prev",), use_container_width=True)

            # Page counter in center
            with page_col:
                st.markdown(f"<div style='text-align: center'>Page {st.session_state['current_page']} of {total_pages}</div>", 
                        unsafe_allow_html=True)

            # Next button on right
            with next_col:
                next_disabled = st.session_state['current_page'] >= total_pages
                st.button("Next →", disabled=next_disabled, key="next_btn", 
                        on_click=change_page, args=("next",), use_container_width=True)
            
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
                        # Article image handling with robust error catching
                        try:
                            if article.get('urlToImage') and article['urlToImage'].strip():
                                try:
                                    st.image(article['urlToImage'], use_container_width=True)
                                except:
                                    # If article image fails, use our preloaded placeholder
                                    if PLACEHOLDER_LOADED:
                                        st.image(PLACEHOLDER_IMAGE, use_container_width=True)
                            else:
                                # No article image at all
                                if PLACEHOLDER_LOADED:
                                    st.image(PLACEHOLDER_IMAGE, use_container_width=True)
                                
                        except Exception as img_error:
                            # Catch any other unexpected errors with images
                            st.error(f"Couldn't load image: {str(img_error)}")
                            
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
            
            # Simplified Vector DB upload section
            st.subheader("Store In Vector Database")
            st.write("Process articles and store them in the vector database for AI analysis.")

            index_name = st.session_state.get("news_index_name", "newsdata")
            embedding_model = st.session_state.get("news_embedding_model", "intfloat/e5-large-v2")

            if st.button("Process and Store in Vector Database", key="process_store_btn"):
                try:
                    # Process articles
                    with st.spinner("Processing articles..."):
                        chunks = process_news_articles_whole(articles)
                        st.success(f"Created {len(chunks)} documents (one per article).")
                    
                    # Set up Vector DB
                    with st.spinner("Setting up vector database..."):
                        message = create_pinecone_index(
                            pinecone_api_key=PINECONE_API_KEY,
                            index_name=index_name
                        )
                        st.info(message)
                    
                    # # Clear existing data
                    with st.spinner("Clearing previous data from vector database..."):
                        message = clear_pinecone_index(PINECONE_API_KEY, index_name)
                        st.info(message)
                    
                    # Upload to Vector DB - use our custom function that handles namespace issues
                    with st.spinner("Uploading to vector database..."):
                        message = upload_news_to_pinecone(
                                document_chunks=chunks,
                                pinecone_api_key=PINECONE_API_KEY,
                                index_name=index_name,
                                model_name=embedding_model
                            )
                        st.success(message)
                        
                        if "Error:" in message:
                            st.error(message)
                        else:
                            st.success(message)
                            st.success("✅ News articles have been processed and stored successfully!")
                            st.info("🔍 Now you can go to the 'Ask Questions' tab to analyze these articles!")
                    
                except Exception as e:
                    st.error(f"Error processing and storing articles: {str(e)}")
    except ImportError as e:
        st.error(f"Error: Could not import required modules: {str(e)}")
        st.error("Please check that all required News Analyzer modules are installed and properly configured.")


def render_news_questions_tab(retriever: Optional[NewsRetriever], llm: Optional[NewsLLM]) -> None:
    """
    Render the News Questions tab content.
    This tab allows users to ask questions about the fetched news articles using AI.
    
    Args:
        retriever: The initialized news retriever
        llm: The initialized LLM interface
    """
    st.header("Ask Questions About the News")
    
    # Advanced options from sidebar
    show_query_processing = st.session_state.get("news_show_query", False)
    results_count = st.session_state.get("news_results_count", 10)
    
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
                        search_query = enhanced_query  # Use enhanced query
                    else:
                        search_query = user_question

                    # Search using the appropriate query
                    search_results = retriever.search(search_query, top_k=results_count)
                    
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

def render_options_app() -> None:
    """
    Render the Options & Strategies application.
    This includes the main content area with tabs for different options analysis features.
    """
    st.image("img/header_img2.png", use_container_width=True)
    
    # Create tabs for EOD Chain, Strategy Viz, and Watchlist
    options_tabs = st.tabs(["EOD Chain Analysis", "Strategy Visualization", "Market Watchlist"])
    
    with options_tabs[0]:  # EOD Chain Analysis Tab
        render_eod_chain_tab()
            
    with options_tabs[1]:  # Strategy Visualization Tab
        render_strategy_viz_tab()

    with options_tabs[2]:  # Watchlist Tab
        render_watchlist_tab()

def render_news_app(retriever=None, llm=None) -> None:
    """
    Render the News Analyzer application.
    This includes tabs for fetching news and asking AI-powered questions about the news.
    
    Args:
        retriever: The initialized news retriever
        llm: The initialized LLM interface
    """
    st.image("img/header_img.png", use_container_width=True)
    
    # Set up the tabs
    tab1, tab2 = st.tabs(["NewsViz", "Ask Questions"])
    
    # Tab 1: Fetch News
    with tab1:
        render_news_fetch_tab()
    
    # Tab 2: Ask Questions
    with tab2:
        render_news_questions_tab(retriever, llm)


def main():
    """
    Main application function - initializes and runs the Streamlit app.
    """
    # Initialize session state variables
    initialize_session_state()
    
    # Apply custom CSS
    render_custom_css()
    
    # Always render the sidebar - it returns retriever and llm if in news mode
    retriever, llm = render_sidebar()
    
    # Render the main app based on current selection
    if st.session_state['current_app'] == "options":
        render_options_app()
    elif st.session_state['current_app'] == "news":
        render_news_app(retriever, llm)

# Run the application
if __name__ == "__main__":
    main()


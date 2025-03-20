"""
Financial Analysis Dashboard

This application provides a comprehensive financial analysis dashboard with multiple features:
- Options & Strategies visualization and analysis
- News aggregation and AI-powered analysis
- Interactive charts and market data visualization

The app is built using Streamlit and integrates various financial data sources
and visualization libraries for a complete trading research experience.

Author: Mithul Raaj
Created: March 2025
License: MIT
"""



# Standard library imports
import os
import sys
from typing import Union

# Third-party imports
from dotenv import load_dotenv
from PIL import Image
import streamlit as st

# Import local modules after path setup and config initialization
from news_module import (
    render_news_app, render_news_sidebar,
)
from options_module import (
    render_options_app, render_options_sidebar,
)

# IMPORTANT: Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set environment variables first, before any other imports or operations
os.environ["STREAMLIT_WATCH_MODULE_PATH"] = "false"
os.environ["STREAMLIT_DISABLE_WATCHER"] = "true"

# Add the news, strategies and dataviz folders to sys.path
sys.path.append("news")
sys.path.append("optionsllm")

# Load environment variables from .env file
load_dotenv()

# Initialize placeholder image
PLACEHOLDER_IMAGE = None
# Try to load the image once at startup
PLACEHOLDER_IMAGE = Image.open("img/news_icon.jpg")
PLACEHOLDER_LOADED = True

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
    # if 'alpaca_configured' not in st.session_state:
    #     st.session_state['alpaca_configured'] = False
    if 'options_ticker' not in st.session_state:
        st.session_state['options_ticker'] = "AAPL"

    # Store API keys in session state for modules to access
    if 'PINECONE_API_KEY' not in st.session_state:
        st.session_state['PINECONE_API_KEY'] = PINECONE_API_KEY
    if 'OPENROUTER_API_KEY' not in st.session_state:
        st.session_state['OPENROUTER_API_KEY'] = OPENROUTER_API_KEY
    if 'NEWSAPI_KEY' not in st.session_state:
        st.session_state['NEWSAPI_KEY'] = NEWSAPI_KEY


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
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)


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
    st.sidebar.button(
        "Options Strategies",
        key="options_btn",
        help="Options chain, analysis, and strategies",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state['current_app'] == "options"
            else "secondary"
        ),
        on_click=lambda: st.session_state.update({"current_app": "options"})
    )

    st.sidebar.button(
        "News Analyzer",
        key="news_btn",
        help="Analyze news articles with AI",
        use_container_width=True,
        type="primary" if st.session_state['current_app'] == "news" else "secondary",
        on_click=lambda: st.session_state.update({"current_app": "news"})
    )

    st.sidebar.divider()

    # App-specific sidebar options - these change based on the current app
    retriever, llm = None, None
    if st.session_state['current_app'] == "options":
        render_options_sidebar()
    elif st.session_state['current_app'] == "news":
        retriever, llm = render_news_sidebar(PINECONE_API_KEY, OPENROUTER_API_KEY, NEWSAPI_KEY)

    return retriever, llm


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

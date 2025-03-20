"""
Options Analysis Module

This module handles all options-related functionality for the Financial Analysis Dashboard.
It includes functions for options chain visualization, strategy visualization, and analysis.

Author: Mithul Raaj
Created: March 2025
License: MIT
"""

# Third-party imports
import streamlit as st
from openai import OpenAI

# Initialize module availability flags
EOD_CHAIN_AVAILABLE = False
OPTIONS_MODULES_AVAILABLE = False
OPTIONS_VIZ_AVAILABLE = False

# Try to import options-specific modules
try:
    from eod_chain import (
        plot_surface, calc_unusual_table, colorize_rows, get_options_data,
        create_open_interest_chart, create_volume_chart, create_iv_chart,
        get_tradingview_widgets
    )
    EOD_CHAIN_AVAILABLE = True
except ImportError:
    pass

try:
    from strategy import (
        get_option_data, plot_strategy,
        get_available_strategies, get_strategy_description
    )
    OPTIONS_VIZ_AVAILABLE = True
except ImportError:
    pass

# Import Options Analysis modules
try:
    from optionsllm.json_packaging import build_compact_options_json
    from optionsllm.main import summarize_options_data
    OPTIONS_MODULES_AVAILABLE = True
except ImportError:
    pass


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
        # Get TradingView widgets HTML
        widgets = get_tradingview_widgets(ticker_input)
        if len(widgets) >= 3:
            _, tech_perf, _ = widgets[0], widgets[1], widgets[2]

            # Place technical analysis widget in sidebar
            with st.sidebar:
                st.components.v1.html(tech_perf, height=400)
        else:
            st.sidebar.error("Failed to load TradingView widgets.")


def render_options_app() -> None:
    """
    Render the Options & Strategies application.
    This includes the main content area with tabs for different options analysis features.
    """
    st.image("img/header_img.png", use_container_width=True)

    # Create tabs for EOD Chain, Strategy Viz, and Watchlist
    options_tabs = st.tabs(["EOD Chain Analysis", "Strategy Visualization",
                            "Market Watchlist"])

    with options_tabs[0]:  # EOD Chain Analysis Tab
        render_eod_chain_tab()

    with options_tabs[1]:  # Strategy Visualization Tab
        render_strategy_viz_tab()

    with options_tabs[2]:  # Watchlist Tab
        render_watchlist_tab()


def render_eod_chain_tab() -> None:
    """
    Render the EOD Chain Analysis tab content.
    This includes options chain visualization, unusual activity, volatility surface, etc.
    """
    ticker_input = st.session_state.get('options_ticker', "")

    if not ticker_input:
        st.info("👈 Please enter a ticker symbol in the sidebar to analyze.")
        return

    if not EOD_CHAIN_AVAILABLE:
        st.error("EOD Chain module not available.")
        return

    # Display widgets and get options data
    widgets = get_tradingview_widgets(ticker_input)
    _display_tradingview_info(widgets)

    try:
        options_data = _fetch_options_data(ticker_input)
        if not options_data:
            return

        (expiration_dates, df_calls, df_puts,
         calls_dict, puts_dict, underlying_price) = options_data

        if not expiration_dates:
            st.warning("No options expiration dates")
            return

        # Display options analysis sections
        _display_unusual_activity_section(df_calls, df_puts)

        # Chain analysis section
        st.divider()
        st.write("#### Chain Analysis")

        exp_date = st.selectbox(
            "Select an expiration date",
            expiration_dates,
            key="expiration_date_select"
        )

        # Package chain data into a dictionary to avoid too many positional arguments
        chain_data = {
            'exp_date': exp_date,
            'calls_dict': calls_dict,
            'puts_dict': puts_dict,
            'expiration_dates': expiration_dates,
            'underlying_price': underlying_price,
            'widgets': widgets
        }
        _display_chain_analysis(chain_data)

        # Add Options Strategy Advisor Section
        _display_options_advisor(ticker_input)

    except ImportError:
        st.error("Required modules are missing for options analysis.")
    except ValueError as val_err:
        st.error(f"Value error in options data: {val_err}")
    except KeyError as key_err:
        st.error(f"Missing key in options data: {key_err}")
    except IndexError as idx_err:
        st.error(f"Index error in options data: {idx_err}")


def _display_tradingview_info(widgets) -> None:
    """Display TradingView information widgets."""
    if len(widgets) >= 10:  # Make sure all widgets are loaded
        symbol_info = widgets[3]
        financial_info = widgets[4]
        company_profile = widgets[5]

        # Display symbol info widget at the top
        st.components.v1.html(symbol_info, height=180)

        # Company Profile in dropdown/expander
        with st.expander("Company Information", expanded=False):
            st.components.v1.html(company_profile, height=300)

        # Financial Info in dropdown/expander
        with st.expander("Financial Information", expanded=False):
            st.components.v1.html(financial_info, height=500)

        st.divider()  # Add separator line


def _fetch_options_data(ticker_input):
    """Fetch options data and return it as a tuple."""
    # Avoiding general exception, using specific handling
    return get_options_data(ticker_input)


def _display_unusual_activity_section(df_full_chain_calls, df_full_chain_puts) -> None:
    """Display the unusual options activity section."""
    st.write("#### Unusual Options Activity")

    col_activity = st.columns((4, 4), gap='small')

    # Display calls activity
    with col_activity[0]:
        _display_calls_activity(df_full_chain_calls)

    # Display puts activity
    with col_activity[1]:
        _display_puts_activity(df_full_chain_puts)


def _display_calls_activity(df_full_chain_calls) -> None:
    """Display calls unusual activity."""
    st.write("#### Calls")
    oi_min_calls = st.number_input(
        "Minimum OI",
        min_value=1,
        value=1_000,
        help='Minimum Open Interest to consider',
        key="oi_min_calls_input"
    )
    show_itm_calls = st.checkbox(
        "Show ITM",
        value=False,
        help='Only show in-the-money (ITM) contracts',
        key="show_itm_calls_checkbox"
    )

    if df_full_chain_calls is not None and not df_full_chain_calls.empty:
        df_full_chain_calls_proc = calc_unusual_table(
            df_full_chain_calls,
            show_itm_calls,
            oi_min_calls
        )

        if not df_full_chain_calls_proc.empty:
            styled_df_calls = df_full_chain_calls_proc.style.apply(
                lambda row: colorize_rows(row, df_full_chain_calls_proc),
                axis=1
            )
            st.dataframe(styled_df_calls, key="calls_unusual_activity_df")
        else:
            st.info("No calls with unusual activity matching the criteria.")
    else:
        st.info("No call options data available.")


def _display_puts_activity(df_full_chain_puts) -> None:
    """Display puts unusual activity."""
    st.write("#### Puts")
    oi_min_puts = st.number_input(
        "Minimum OI",
        min_value=1,
        value=1_000,
        help='Minimum Open Interest to consider',
        key="oi_min_puts_input"
    )
    show_itm_puts = st.checkbox(
        "Show ITM",
        value=False,
        help='Only show in-the-money (ITM) contracts',
        key="show_itm_puts_checkbox"
    )

    if df_full_chain_puts is not None and not df_full_chain_puts.empty:
        df_full_chain_puts_proc = calc_unusual_table(
            df_full_chain_puts,
            show_itm_puts,
            oi_min_puts
        )

        if not df_full_chain_puts_proc.empty:
            styled_df_puts = df_full_chain_puts_proc.style.apply(
                lambda row: colorize_rows(row, df_full_chain_puts_proc),
                axis=1
            )
            st.dataframe(styled_df_puts, key="puts_unusual_activity_df")
        else:
            st.info("No puts with unusual activity.")
    else:
        st.info("No put options data available.")


def _display_chain_analysis(chain_data) -> None:
    """
    Display options chain analysis for the selected expiration date.

    Args:
        chain_data (dict): Dictionary containing all chain analysis data
    """
    exp_date = chain_data['exp_date']
    calls_dict = chain_data['calls_dict']
    puts_dict = chain_data['puts_dict']
    expiration_dates = chain_data['expiration_dates']
    underlying_price = chain_data['underlying_price']
    widgets = chain_data['widgets']

    if exp_date not in calls_dict or exp_date not in puts_dict:
        st.warning("No option chain data available")
        return

    calls = calls_dict[exp_date]
    puts = puts_dict[exp_date]

    calls = calls.sort_values(by='strike')
    puts = puts.sort_values(by='strike')

    # Calculate ATM value
    atm = _calculate_atm_value(underlying_price, calls)

    # Display option charts
    chart_data = {
        'calls': calls,
        'puts': puts,
        'atm': atm,
        'calls_dict': calls_dict,
        'puts_dict': puts_dict,
        'expiration_dates': expiration_dates
    }
    _display_option_charts(chart_data)

    # Display TradingView chart
    _display_underlying_chart(widgets)


def _calculate_atm_value(underlying_price, calls):
    """Calculate the at-the-money value for option charts."""
    atm = underlying_price if underlying_price else None
    if atm is None and not calls.empty and 'strike' in calls.columns:
        # Calculate midpoint if underlying price not available
        strikes = sorted(calls['strike'].unique())
        atm = float(strikes[len(strikes) // 2])
    elif atm is None:
        atm = 100.0  # Default fallback value
    return atm


def _display_option_charts(chart_data):
    """Display option charts (OI, volume, IV, surface)."""
    calls = chart_data['calls']
    puts = chart_data['puts']
    atm = chart_data['atm']

    # Interest and volume charts
    col_inner = st.columns((4, 4), gap='small')

    with col_inner[0]:
        # Create open interest chart
        oi_fig = create_open_interest_chart(calls, puts, atm)
        st.plotly_chart(oi_fig, use_container_width=True)

    with col_inner[1]:
        # Create volume chart
        vol_fig = create_volume_chart(calls, puts, atm)
        st.plotly_chart(vol_fig, use_container_width=True)

    # IV charts
    col_vol = st.columns((4, 4), gap='small')

    with col_vol[0]:
        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
        # Create IV chart
        iv_fig = create_iv_chart(calls, puts)
        st.plotly_chart(iv_fig, use_container_width=True)

    with col_vol[1]:
        # Volatility surface plot
        surface_data = {
            'calls_dict': chart_data['calls_dict'],
            'puts_dict': chart_data['puts_dict'],
            'expiration_dates': chart_data['expiration_dates']
        }
        _display_volatility_surface(surface_data)


def _display_volatility_surface(surface_data):
    """Display volatility surface plot."""
    calls_dict = surface_data['calls_dict']
    puts_dict = surface_data['puts_dict']
    expiration_dates = surface_data['expiration_dates']

    show_calls = st.checkbox(
        "Calls",
        value=True,
        key='volatility_surface_calls',
        help='Show surface'
    )

    # Using specific exceptions instead of general Exception
    try:
        if show_calls:
            surface_fig = plot_surface(calls_dict, expiration_dates)
        else:
            surface_fig = plot_surface(puts_dict, expiration_dates)

        st.plotly_chart(surface_fig, use_container_width=True)
    except ValueError as e:
        st.error(f"Value error plotting surface: {e}")
    except KeyError as e:
        st.error(f"Missing data for surface plot: {e}")


def _display_underlying_chart(widgets):
    """Display the underlying price chart from TradingView."""
    st.divider()
    st.write("#### Underlying Price Chart")

    if len(widgets) >= 3:
        _, _, tv_advanced_plot = widgets[0], widgets[1], widgets[2]
        st.components.v1.html(tv_advanced_plot, height=400)
    else:
        st.error("Advanced plot widget not available.")


def _display_options_advisor(ticker_input):
    """Display the Options Strategy Advisor section."""
    st.divider()

    if not OPTIONS_MODULES_AVAILABLE:
        st.error("Options Analysis modules not available.")
        return

    st.markdown(
        '<div class="section-header">Options Strategy Advisor</div>',
        unsafe_allow_html=True
    )

    # Create form and handle submission
    _create_options_advisor_form(ticker_input)


def _create_options_advisor_form(ticker_input):
    """Create and handle the options advisor form."""
    with st.form(key="options_analysis_form"):
        # Form elements in a single row
        col1, col2 = st.columns([3, 1])

        with col1:
            # Don't store in variable since it's accessed via session_state
            st.text_area(
                label="Your question:",
                height=120,
                placeholder="Example: Recommended trades?",
                key="options_query_textarea"
            )

        with col2:
            # Options in a dropdown
            st.write("Analysis Settings")
            # Don't store in variable since it's accessed via session_state
            st.selectbox(
                label="Expirations to analyze:",
                options=[1, 2, 3, 5, 10],
                index=2,  # Default to 3
                key="expirations_dropdown"
            )
            # Don't store in variable since it's accessed via session_state
            st.checkbox(
                label="Include Historical Volatility",
                value=True,
                key="include_hv_checkbox"
            )

        # Submit button with more descriptive text
        analyze_submitted = st.form_submit_button(label="Fetch Data & Analyze")

    # Handle form submission
    if analyze_submitted:
        _process_advisor_form_submission(ticker_input)


def _process_advisor_form_submission(ticker_input):
    """Process the form submission and display results."""
    form_query = st.session_state.options_query_textarea
    form_exp_limit = st.session_state.expirations_dropdown
    form_include_hv = st.session_state.include_hv_checkbox

    if not form_query:
        st.warning("Please enter a question about options data.")
        return

    # Process with spinner
    with st.spinner(f"Fetching data for {ticker_input} and analyzing..."):
        # Fetch options data
        data = build_compact_options_json(
            ticker_input,
            form_exp_limit,
            form_include_hv
        )
        summary = summarize_options_data(data)

        st.session_state['options_data'] = data
        st.session_state['options_summary'] = summary

        # Get LLM analysis with structured JSON orders
        llm_response = ask_llm_about_options_safe(
            summary,
            form_query,
            max_tokens=2000
        )

        # Display the LLM response
        st.markdown(
            '<div class="analyst-header">Analysis Results</div>',
            unsafe_allow_html=True
        )
        with st.container():
            st.markdown(llm_response)


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
    """
    # Retrieve API keys with proper validation
    openrouter_api_key = st.session_state.get("OPENROUTER_API_KEY", "")
    openai_api_key = st.session_state.get("OPENAI_API_KEY", "")

    # Validate we have at least one API key
    if not openrouter_api_key and not openai_api_key:
        return "Error: No API key available. Please set OPENROUTER_API_KEY or OPENAI_API_KEY."

    # Create the appropriate API client based on available keys
    client = None
    model = None
    extra_headers = {}

    if openrouter_api_key:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key
        )
        model = "deepseek/deepseek-chat:free"
        extra_headers = {
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "OptionsAnalysisLLM"
        }
    elif openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        model = "gpt-4"  # Default to GPT-4

    # System prompt for the LLM
    system_prompt = (
        "You are an expert options strategist. "
        f"Answer the user's question based on the following options data, "
        f"and keep your response under {max_tokens} tokens.\n\n"
        "IMPORTANT: If your response suggests options trades, strategies, you MUST include"
        " structured JSON block at the end of your message with the following format:\n\n"
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

    try:
        # Configure API call based on which service we're using
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

        # Extract the response text
        if (completion and hasattr(completion, 'choices') and
            len(completion.choices) > 0 and
            hasattr(completion.choices[0], 'message') and
            hasattr(completion.choices[0].message, 'content')):
            return completion.choices[0].message.content

        return "Error: Unable to get a valid response from the language model."

    except ValueError as e:
        return f"Error: Invalid API key or configuration. {str(e)}"
    except KeyError as e:
        return f"Error: Missing required parameters in the API call. {str(e)}"
    except TimeoutError:
        return "Error: The request timed out. Please try again later."


def render_strategy_viz_tab() -> None:
    """
    Render the Strategy Visualization tab content.
    This tab provides interactive visualization of various options strategies
    and displays information about strategy characteristics and use cases.
    """
    ticker_input = st.session_state.get('options_ticker', "")

    if EOD_CHAIN_AVAILABLE:
        # Get TradingView widgets
        widgets = get_tradingview_widgets(ticker_input)

        if len(widgets) >= 10:
            symbol_info = widgets[3]

            # Display symbol info widget at the top
            st.components.v1.html(symbol_info, height=180)
        else:
            st.error("Failed to load TradingView widgets for the Strategy tab.")
    else:
        st.error("Widgets are not available.")

    ticker = st.session_state['options_ticker']
    if not ticker:
        st.info("👈 Please enter a ticker symbol in the sidebar first.")
    else:
        if OPTIONS_VIZ_AVAILABLE:
            # Get available strategies
            strategies = get_available_strategies()
            selected_strategy = st.selectbox("Select a strategy",
                                             strategies,
                                             key="strat_select")

            # Get option data for the ticker
            atm_call_strike, _, _, _ = get_option_data(ticker)
            if atm_call_strike:

                # Strike price slider with better precision
                min_strike = max(1, round(atm_call_strike * 0.8, 1))
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
                        st.error("Could not create strategy visualization.")

                # Get strategy description
                description = get_strategy_description(selected_strategy)
                if description:
                    st.markdown("#### About This Strategy")
                    st.markdown(description)
            else:
                st.error(f"Could not fetch options data for {ticker}.")
        else:
            st.error("Options Visualization module not available.")


def render_watchlist_tab() -> None:
    """
    Render the Market Watchlist tab content.
    This tab provides market overviews, stock indices.
    """
    ticker_input = st.session_state.get('options_ticker', "")

    if EOD_CHAIN_AVAILABLE:
        # Get TradingView widgets
        widgets = get_tradingview_widgets(ticker_input)

        if len(widgets) >= 10:
            market_overview = widgets[6]
            stock_overview = widgets[7]
            running_ticker = widgets[8]
            heatmap = widgets[9]

            # Display running ticker at the top
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
            st.components.v1.html(heatmap, height=485)
        else:
            st.error("Failed to load TradingView widgets")
    else:
        st.error("Widgets are not available.")

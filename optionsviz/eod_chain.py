"""
Module Name: eod_chain.py

Description:
    This module serves as the streamlit end-of-day (EOD) options chain visualization page. 
    It provides functionality to visualize options data, including implied volatility,
    volume, open interest, and unusual activity.

Author:
    Ryan J Richards

Created:
    Feb 2025

License:
    MIT
"""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

@st.cache_data()
def create_iv_smile(calls: pd.DataFrame,
                    puts: pd.DataFrame,
                    atm: float) -> go.Figure:
    """
    Function: create_iv_smile
    
    Description:
        The function creates a plotly figure showing the implied volatility smile for
        call and put options. The function takes in dataframes for calls and puts,
        and the atm (at-the-money) strike price. It returns a plotly figure object.
    
    Parameters:
        calls (pd.DataFrame): DataFrame containing call options data.
        puts (pd.DataFrame): DataFrame containing put options data.
        atm (float): The atm strike price.
        
    Returns:
        go.Figure: A plotly figure object containing the implied volatility smile.
    
    Raises:
        TypeError: type is incorrect for any inputs
        ValueError: invalid value for atm (less than 0)
    
    Example:
        fig = create_iv_smile(calls, puts, atm)
        fig.show()
    """
    if not isinstance(calls, pd.DataFrame):
        raise TypeError('calls must be a dataframe')

    if not isinstance(puts, pd.DataFrame):
        raise TypeError('puts must be a dataframe')

    if not isinstance(atm, float):
        raise TypeError('atm must be a float')

    if atm < 0:
        raise ValueError('atm must be gte 0')

    call_iv = calls['impliedVolatility'].values
    call_iv[np.isnan(call_iv)] = 0.

    put_iv = puts['impliedVolatility'].values
    put_iv[np.isnan(put_iv)] = 0.

    max_iv = np.maximum(call_iv.max(), put_iv.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=calls.strike, y=calls['impliedVolatility']*100,
                                mode='lines+markers', name='Call IV',
                                line={'color':'#00C66B'}
                                ))
    fig.add_trace(go.Scatter(x=puts.strike, y=puts['impliedVolatility']*100,
                                mode='lines+markers', name='Put IV',
                                line={'color':'#D9534F'}
                                ))
    fig.add_shape(
        type="rect",
        x0=0, x1=atm,
        y0=0, y1=max_iv,
        fillcolor="#B39DDB",
        opacity=0.15,
        line_width=0,
        name='ITM Level'
    )
    fig.update_layout(title="Implied Volatility (%) by Strike ($); 'Volatility Smile'",
                        xaxis_title="Strike Price ($)", yaxis_title="IV (%)")
    return fig

@st.cache_data()
def create_vol_hists(calls: pd.DataFrame,
                     puts: pd.DataFrame,
                     atm: float) -> go.Figure:
    """
    Function: create_vol_hists
    
    Description:
        The function creates a plotly figure showing the Volume based histograms for
        call and put options. The function takes in dataframes for calls and puts,
        and the atm (at-the-money) strike price. It returns a plotly figure object.
    
    Parameters:
        calls (pd.DataFrame): DataFrame containing call options data.
        puts (pd.DataFrame): DataFrame containing put options data.
        atm (float): The atm strike price.
        
    Returns:
        go.Figure: A plotly figure object containing the volume histograms
         for calls and puts.
    
    Raises:
        TypeError: type is incorrect for any inputs
        ValueError: invalid value for atm (less than 0)
    
    Example:
        fig = create_vol_hists(calls, puts, atm)
        fig.show()
    """
    if not isinstance(calls, pd.DataFrame):
        raise TypeError('calls must be a dataframe')

    if not isinstance(puts, pd.DataFrame):
        raise TypeError('puts must be a dataframe')

    if not isinstance(atm, float):
        raise TypeError('atm must be a float')

    if atm < 0:
        raise ValueError('atm must be gte 0')

    call_vol = calls.volume.values
    call_vol[np.isnan(call_vol)] = 0.

    put_vol = puts.volume.values
    put_vol[np.isnan(put_vol)] = 0.

    max_vol = np.maximum(call_vol.max(), put_vol.max())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=puts['strike'],
        y=puts['volume'],
        name='Puts',
        orientation='v',
        marker_color='#D9534F'
    ))

    fig.add_trace(go.Bar(
        x=calls['strike'],
        y=calls['volume'],
        name='Calls',
        orientation='v',
        marker_color='#00C66B', marker_opacity=0.5
    ))

    fig.add_shape(
        type="rect",
        x0=0, x1=atm,
        y0=0, y1=max_vol,
        fillcolor="#B39DDB",
        opacity=0.15,
        line_width=0,
        name='ITM Level'
    )

    fig.update_layout(
        title="Volume by Strike ($)",
        xaxis_title="Strike Price ($)",
        yaxis_title="Volume",
        barmode="overlay",
        template="plotly_dark",
        bargap=0.01,
        bargroupgap=0.01,
    )
    return fig

@st.cache_data()
def create_oi_hists(calls: pd.DataFrame,
                    puts: pd.DataFrame,
                    atm: float) -> go.Figure:
    """
    Function: create_oi_hists
    
    Description:
        The function creates a plotly figure showing the Open Interest based histograms for
        call and put options. The function takes in dataframes for calls and puts,
        and the atm (at-the-money) strike price. It returns a plotly figure object.
    
    Parameters:
        calls (pd.DataFrame): DataFrame containing call options data.
        puts (pd.DataFrame): DataFrame containing put options data.
        atm (float): The atm strike price.
        
    Returns:
        go.Figure: A plotly figure object containing the Open Interest histograms
         for calls and puts.
    
    Raises:
        TypeError: type is incorrect for any inputs
        ValueError: invalid value for atm (less than 0)
    
    Example:
        fig = create_oi_hists(calls, puts, atm)
        fig.show()
    """
    if not isinstance(calls, pd.DataFrame):
        raise TypeError('calls must be a dataframe')

    if not isinstance(puts, pd.DataFrame):
        raise TypeError('puts must be a dataframe')

    if not isinstance(atm, float):
        raise TypeError('atm must be a float')

    if atm < 0:
        raise ValueError('atm must be gte 0')

    max_oi = np.maximum(calls.openInterest.values.max(), puts.openInterest.values.max())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=puts['strike'],
        y=puts['openInterest'].values,
        name='Puts',
        orientation='v',
        marker_color='#D9534F'
    ))
    fig.add_trace(go.Bar(
        x=puts['strike'],
        y=puts['openInterest'],
        name='Puts',
        orientation='v',
        marker_color='#D9534F'
    ))
    fig.add_trace(go.Bar(
        x=calls['strike'],
        y=calls['openInterest'],
        name='Calls',
        orientation='v',
        marker_color='#00C66B', marker_opacity=0.5
    ))

    fig.add_shape(
        type="rect",
        x0=0, x1=atm,
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
    return fig

@st.cache_data()
def plot_surface(chains: dict,
                 expiration_dates: list) -> go.Figure:
    """
    Function: plot_surface
    
    Description:
        The function creates a plotly figure showing the Volatility Surface for
        call or put option chains; this includes all shared strikes across all
        expiration dates. The function takes in a dictionary of dataframes for
        calls and puts, and a list of expiration dates. It returns a plotly figure object.

    Parameters:
        chains (dict): dictionary containing call or put options based on 
            expiration date (key).
        expiration_dates (list): list of expiration dates (list of strings).
        
    Returns:
        go.Figure: A plotly figure object containing the Volatility Surface
         for calls or puts.
    
    Raises:
        TypeError: type is incorrect for any inputs
        ValueError: invalid value for expiration_dates (empty) or 
            chains (empty)
    
    Example:
        fig = plot_surface(chains, expiration_dates)
        fig.show()
    """
    if len(expiration_dates) == 0:
        raise ValueError('Expiration dates is empty.')

    if len(list(chains.keys())) == 0:
        raise ValueError('Chain dataframes contains no keys, is empty.')

    if not isinstance(chains, dict):
        raise TypeError('Enter a dictionary for chains input arg')

    if not isinstance(expiration_dates, list):
        raise TypeError('Enter a list for expiration_dates input arg')

    ys_calls, zs_calls = [], []
    for expiration in expiration_dates:
        if len(chains[expiration]) > 0:
            ys_calls.append(list(chains[expiration].strike.values))
            zs_calls.append(list(chains[expiration]['impliedVolatility'].values * 100.))

    xs_matched, ys_matched, zs_matched = [], [], []
    for i, (y_c, z_c) in enumerate(zip(ys_calls, zs_calls)):
        xs_matched.extend([np.arange(len(expiration_dates))[i]]*len(y_c))
        ys_matched.extend(y_c)
        zs_matched.extend(z_c)

    uniq_strikes = {}
    for y_s in ys_calls:
        for y_strike in y_s:
            uniq_strikes[y_strike] = 1 if y_strike not \
                in uniq_strikes else uniq_strikes[y_strike] + 1

    uniq_strikes = np.array(list(uniq_strikes.keys()))\
                        [np.array(list(uniq_strikes.values()))==len(expiration_dates)]
    xs_matched = np.array(xs_matched)[np.isin(ys_matched, uniq_strikes)]
    zs_matched = np.array(zs_matched)[np.isin(ys_matched, uniq_strikes)]
    ys_matched = np.array(ys_matched)[np.isin(ys_matched, uniq_strikes)]

    fig = go.Figure(data=[go.Surface(z=zs_matched.reshape((len(ys_calls),uniq_strikes.shape[0])),
                                    x=xs_matched.reshape((len(ys_calls),uniq_strikes.shape[0])),
                                    y=ys_matched.reshape((len(ys_calls),uniq_strikes.shape[0])),
                                    cmin=0,
                                    cmax=zs_matched.max()+10)])

    fig.update_layout(
        title={'text':'Volatility Surface'},
        autosize=True,
        width=500,
        height=500,
        scene={
            'xaxis_title':'Expiration Date',
            'yaxis_title':'Strike Price ($)',
            'zaxis_title':'IV (%)',
            'xaxis':{
                    'tickmode': 'array',
                    'tickvals': xs_matched.reshape((len(ys_calls),
                                                       uniq_strikes.shape[0]))[:,0][::2],
                    'ticktext' : expiration_dates[::2],
                    'tickfont':{'size':10}
            },
        },
    )
    return fig

@st.cache_data()
def calc_unusual_table(df_full_chain: pd.DataFrame,
                       show_itm: bool = True,
                       oi_min: int = 1_000) -> pd.DataFrame:
    """
    Function: calc_unusual_table

    Description:
        This function calculates the unusual options activity table for a given
        options chain dataframe. It filters the dataframe based on volume, open interest,
        and unusual activity, and returns a styled dataframe.

    Parameters:
        df_full_chain (pd.DataFrame): DataFrame containing options chain data.
        show_itm (bool): Flag to show in-the-money (ITM) contracts. Default is True.
        oi_min (int): Minimum open interest to consider. Default is 1_000.    
    
    Returns:
        pd.DataFrame: A styled dataframe containing the unusual options activity table.    
    
    Raises:
        TypeError: type is incorrect for any inputs
        ValueError: invalid value for oi_min (less than 0)
    
    Example:
        df = calc_unusual_table(df_full_chain, show_itm, oi_min)
        print(df)
    """
    if oi_min < 0:
        raise ValueError('oi_min must be greater than or equal to zero.')

    if not isinstance(df_full_chain, pd.DataFrame):
        raise TypeError('df_full_chain must be a dataframe')

    if not isinstance(show_itm, bool):
        raise TypeError('show_itm must be a boolean')

    if not isinstance(oi_min, int):
        raise TypeError('oi_min must be an integer')

    df_full_chain_calls = df_full_chain.copy()

    df_full_chain_calls = df_full_chain_calls[df_full_chain_calls.volume != 0.]
    df_full_chain_calls = df_full_chain_calls[df_full_chain_calls.openInterest != 0.]
    df_full_chain_calls = df_full_chain_calls[~pd.isna(df_full_chain_calls.volume)]
    df_full_chain_calls = df_full_chain_calls[~pd.isna(df_full_chain_calls.openInterest)]
    df_full_chain_calls = df_full_chain_calls[df_full_chain_calls.openInterest >= oi_min]

    df_full_chain_calls['unusual_activity'] = df_full_chain_calls.volume / \
        df_full_chain_calls.openInterest

    df_full_chain_calls = df_full_chain_calls.sort_values('unusual_activity', ascending=False)
    df_full_chain_calls = df_full_chain_calls[df_full_chain_calls.unusual_activity > 1]
    df_full_chain_calls = df_full_chain_calls[df_full_chain_calls.inTheMoney==show_itm]
    df_full_chain_calls['spread'] = df_full_chain_calls.ask - df_full_chain_calls.bid
    df_full_chain_calls = df_full_chain_calls[['contractSymbol', 'strike',
                                            'lastPrice','spread','percentChange',
                                                'volume','openInterest','impliedVolatility',
                                                'unusual_activity']]

    # Apply styling to whole row
    df_full_chain_calls = df_full_chain_calls.reset_index(drop=True)
    return df_full_chain_calls

@st.cache_data()
def generate_widgets(ticker: str):
    """
    Function: generate_widgets

    Description:
        This function generates the TradingView widgets for the given ticker.
        It returns the HTML code for the widgets.

    Parameters:
        ticker (str): The stock ticker symbol.
    
    Returns:
        str: HTML code for the TradingView widgets.
    
    Raises:
        nothing
    
    Example:
        ticker = "AAPL"
        widgets = generate_widgets(ticker)
        print(widgets)
    """

    single_ticker_widget = f'''
    <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
        {{
            "symbol": "{ticker}",
            "locale": "en",
            "dateRange": "1M",
            "colorTheme": "dark",
            "isTransparent": true,
            "autosize": true,
            "largeChartUrl": ""
        }}
        </script>
    </div>
    '''

    tech_perf = f'''
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
                    <span class="blue-text">Track all markets on TradingView</span>
                </a>
            </div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
            {{
                "interval": "1m",
                "width": 425,
                "isTransparent": true,
                "height": 450,
                "symbol": "{ticker}",
                "showIntervalTabs": true,
                "displayMode": "single",
                "locale": "en",
                "colorTheme": "dark"
            }}
            </script>
        </div>
        '''

    tv_advanced_plot = f"""
        <div class="tradingview-widget-container">
            <div id="tradingview_chart"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
                new TradingView.widget({{
                    "width": "100%",
                    "height": 400,
                    "symbol": "{ticker}",
                    "interval": "1",
                    "timezone": "Etc/UTC",
                    "theme": "dark",
                    "style": "1",
                    "locale": "en",
                    "toolbar_bg": "#f1f3f6",
                    "enable_publishing": false,
                    "hide_top_toolbar": false,
                    "save_image": false,
                    "container_id": "tv_advanced_plot"
                }});
            </script>
        </div>
        """

    return single_ticker_widget, tech_perf, tv_advanced_plot

@st.cache_data()
def get_data(ticker: str):
    """
    Function: get_data

    Description:
        This function retrieves the options chain data for the given ticker.
        It returns the options chain dataframes for calls and puts, the expiration
        dates, and the underlying price.

    Parameters:
        ticker (str): The stock ticker symbol.
    
    Returns:
        tuple: A tuple containing the options chain dataframes for calls and puts,
            the expiration dates, the underlying price, and a boolean indicating
            if the ticker is valid.
    
    Raises:
        TypeError: type is incorrect for any inputs
        ValueError: invalid value for ticker (empty)
    
    Example:
        ticker = "AAPL"
        calls, puts, expiration_dates, underlying_price, valid_ticker = get_data(ticker)
        print(calls, puts, expiration_dates, underlying_price, valid_ticker)
    """
    # get option chain and proc
    df_full_chain_calls_dict=None
    df_full_chain_puts_dict=None
    df_full_chain_calls=None
    df_full_chain_puts=None
    money_level=None
    valid_ticker=False

    yfticker = yf.Ticker(ticker)
    expiration_dates = list(yfticker.options)

    if len(expiration_dates) > 0:

        money_level = float(yfticker.option_chain(expiration_dates[0]).\
                            underlying['regularMarketPrice'])

        # show unusual activity table
        valid_ticker=True
        df_full_chain_calls = None
        df_full_chain_puts = None
        df_full_chain_calls_dict = {}
        df_full_chain_puts_dict = {}

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

            # update master dicts
            df_full_chain_calls_dict[e] = calls
            df_full_chain_puts_dict[e] = puts
    else:
        st.error(f'Unable to find option chain for ticker: {ticker}', icon="🚨")

    return df_full_chain_calls_dict, df_full_chain_puts_dict, df_full_chain_calls, \
        df_full_chain_puts, expiration_dates, money_level, valid_ticker

def process_ticker(ticker: str, ticker_cols: list):
    """
    Function: process_ticker

    Description:
        This function processes the given ticker symbol, retrieves the options data,
        generates widgets, and displays the options chain analysis.

    Parameters:
        ticker (str): The stock ticker symbol.
        ticker_cols (list): List of columns for the Streamlit app.
    
    Returns:
        None
    
    Raises:
        None

    Example:
        ticker = "AAPL"
        ticker_cols = [col1, col2]
        process_ticker(ticker, ticker_cols)
    """
    df_calls_dict, df_puts_dict, df_calls, \
            df_puts, expiration_dates, atm, \
                valid_ticker = get_data(ticker)

    if valid_ticker:
        single_ticker_widget, tech_perf, tv_advanced_plot = generate_widgets(ticker)

        with ticker_cols[1]:
            st.components.v1.html(single_ticker_widget, height=100)

        with st.sidebar:
            st.components.v1.html(tech_perf, height=400)

        st.divider()
        st.write("#### Unusual Options Activity")

        display_unusual_activity(st.columns((4,4), gap='small'), df_calls, df_puts)

        st.divider()
        st.write("#### Chain Analysis")

        exp_date = st.selectbox(
                "Select an expiration date",
                expiration_dates,
        )
        calls = df_calls_dict[exp_date]
        puts = df_puts_dict[exp_date]
        calls = calls.sort_values(by='strike')
        puts = puts.sort_values(by='strike')

        display_chain_analysis(col_inner=st.columns((4,4), gap='small'),
                               calls=calls,
                               puts=puts,
                               atm=atm,
                               df_calls_dict=df_calls_dict,
                               df_puts_dict=df_puts_dict,
                               expiration_dates=expiration_dates)

        st.write("#### Underlying Price Chart")
        st.components.v1.html(tv_advanced_plot, height=400)

def display_unusual_activity(col_activity, df_calls, df_puts):
    """
    Function: display_unusual_activity

    Description:
        This function displays the unusual options activity for calls and puts
        in the Streamlit app. It takes in the columns for the app and the dataframes
        for calls and puts.

    Parameters:
        col_activity (list): List of columns for the Streamlit app.
        df_calls (pd.DataFrame): DataFrame containing call options data.
        df_puts (pd.DataFrame): DataFrame containing put options data.
    
    Returns:
        None
    
    Raises:
        None

    Example:
        col_activity = [col1, col2]
        df_calls = pd.DataFrame(...)
        df_puts = pd.DataFrame(...)
        display_unusual_activity(col_activity, df_calls, df_puts)
    """
    with col_activity[0]:
        st.write("#### Calls")
        oi_min_calls = st.number_input("Minumum OI", min_value=1,
                                    key='oi_min_calls',
                                    value=1_000,
                                help='Minumum Open Interest to consider \
                                    when computing unusual options activity.')
        show_itm_calls = st.checkbox("Show ITM", value=False, key='show_itm',
                            help='Only show in-the-money (ITM) contracts, \
                                otherwise show only out-of-money.')
        df_full_chain_calls_proc = calc_unusual_table(df_calls,
                                                      show_itm_calls,
                                                      oi_min_calls)
        styled_df_calls = style_unusual_activity(df_full_chain_calls_proc)
        st.dataframe(styled_df_calls)

    with col_activity[1]:
        st.write("#### Puts")
        oi_min_puts = st.number_input("Minumum OI", min_value=1,
                                    key='oi_min_puts',
                                    value=1_000,
                                    help='Minumum Open Interest to consider when \
                                        computing unusual options activity.')
        show_itm_puts = st.checkbox("Show ITM", value=False, key='show_itm_puts',
                                    help='Only show in-the-money (ITM) contracts, \
                                        otherwise show only out-of-money.')
        df_full_chain_puts_proc = calc_unusual_table(df_puts, show_itm_puts, oi_min_puts)
        styled_df_puts = style_unusual_activity(df_full_chain_puts_proc)
        st.dataframe(styled_df_puts)

def style_unusual_activity(df_full_chain_proc):
    """
    Function: style_unusual_activity

    Description:
        This function styles the unusual options activity dataframe for display
        in the Streamlit app. It applies a color gradient to the rows based on
        the unusual activity values (green to red for high to low activity).

    Parameters:
        df_full_chain_proc (pd.DataFrame): DataFrame containing the processed
            unusual options activity data.
    
    Returns:
        None
    
    Raises:
        None

    Example:
        df_full_chain_proc = pd.DataFrame(...)
        styled_df = style_unusual_activity(df_full_chain_proc)
        print(styled_df)
    """
    def colorize_rows(row):
        norm = (row.unusual_activity - \
                df_full_chain_proc["unusual_activity"].min()) / \
            (df_full_chain_proc["unusual_activity"].max() - \
                df_full_chain_proc["unusual_activity"].min())
        color = f'background-color: rgba({255 * (1 - norm)}, \
            {255 * norm}, 0, 0.5)'
        return [color] * len(row)
    return df_full_chain_proc.style.apply(colorize_rows, axis=1)

def display_chain_analysis(**kwargs):
    """
    Function: display_chain_analysis

    Description:
        This function displays the chain analysis for calls and puts in the
        Streamlit app. It takes in the columns for the app, the dataframes for
        calls and puts, the atm strike price, and the expiration dates.

    Parameters:
        kwargs (dict): Dictionary containing the following keys:
            col_inner (list): List of columns for the Streamlit app.
            calls (pd.DataFrame): DataFrame containing call options data.
            puts (pd.DataFrame): DataFrame containing put options data.
            atm (float): The atm strike price.
            df_calls_dict (dict): Dictionary containing call options dataframes
                based on expiration date (key).
            df_puts_dict (dict): Dictionary containing put options dataframes
                based on expiration date (key).
            expiration_dates (list): List of expiration dates (list of strings).
    
    Returns:
        None
    
    Raises:
        None

    Example:
        kwargs = {
            "col_inner": [col1, col2],
            "calls": pd.DataFrame(...),
            "puts": pd.DataFrame(...),
            "atm": 150.0,
            "df_calls_dict": {...},
            "df_puts_dict": {...},
            "expiration_dates": ["2025-02-28", "2025-03-01"]
        }
        display_chain_analysis(**kwargs)
    """
    col_inner = kwargs.get('col_inner')
    calls = kwargs.get('calls')
    puts = kwargs.get('puts')
    atm = kwargs.get('atm')
    df_calls_dict = kwargs.get('df_calls_dict')
    df_puts_dict = kwargs.get('df_puts_dict')
    expiration_dates = kwargs.get('expiration_dates')

    if not isinstance(col_inner, list):
        raise TypeError('col_inner must be a list')

    if not isinstance(calls, pd.DataFrame):
        raise TypeError('calls must be a dataframe')

    if not isinstance(puts, pd.DataFrame):
        raise TypeError('puts must be a dataframe')

    if not isinstance(atm, float):
        raise TypeError('atm must be a float')

    if not isinstance(df_calls_dict, dict):
        raise TypeError('df_calls_dict must be a dictionary')

    if not isinstance(df_puts_dict, dict):
        raise TypeError('df_puts_dict must be a dictionary')

    if not isinstance(expiration_dates, list):
        raise TypeError('expiration_dates must be a list')

    with col_inner[0]:
        oi_hist = create_oi_hists(calls, puts, atm)
        st.plotly_chart(oi_hist)

    with col_inner[1]:
        vol_hists = create_vol_hists(calls, puts, atm)
        st.plotly_chart(vol_hists)

    col_vol = st.columns((4,4), gap='small')

    with col_vol[0]:
        iv_smile = create_iv_smile(calls, puts, atm)
        st.plotly_chart(iv_smile)

    with col_vol[1]:
        show_calls = st.checkbox("Calls",
                                value=True,
                                key='volatility_surface_calls',
                                help='Show surface for calls (checked) or puts (unchecked)')

        if show_calls:
            surface_fig = plot_surface(df_calls_dict, expiration_dates)
        else:
            surface_fig = plot_surface(df_puts_dict, expiration_dates)
        st.plotly_chart(surface_fig, use_container_width=True)

    return True

def main():
    """
    Function: main

    Description:
        This is the main function, which serves as the entry point for the
        Streamlit application. It initializes the app, sets the page configuration,
        and handles user input for stock tickers. It retrieves options data, generates
        widgets, and displays the options chain analysis.

    Parameters:
        None
    
    Returns:
        None    
    
    Raises:
        None    
    
    Example:
        main()
    """
    ticker_cols = st.columns((3,4), gap='small')

    with ticker_cols[0]:
        ticker = st.text_input("Enter stock ticker:",
                               value=None,
                               placeholder='e.g. NVDA, AAPL, AMZN')

    if ticker is not None:
        ticker = ticker.upper()
        process_ticker(ticker, ticker_cols)

# call main
main()

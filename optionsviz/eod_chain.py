"""
EOD Chain Analysis Wrapper Module
This module wraps the functionality from eod_chain.py to be used in other Streamlit applications
without causing conflicts with Streamlit's page configuration.

Author:
    Ryan J Richards
Created:
    March 2025
License:
    MIT
"""
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
logging.basicConfig(level=logging.WARNING)

@st.cache_data()
def create_iv_smile(calls: pd.DataFrame,
                    puts: pd.DataFrame,
                    atm: float) -> go.Figure:
    """
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
    """
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
    """

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
    """
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
def _prepare_surface_data(chains: dict, expiration_dates: list):
    """
    Prepares the data for plotting the volatility surface.

    Returns:
        tuple: (xs_matched, ys_matched, zs_matched, shape)
        or None if there is insufficient data.
    """
    xs_matched, ys_matched, zs_matched = [], [], []
    for i, exp in enumerate(expiration_dates):
        df = chains[exp]
        if len(df) > 0:
            strikes = list(df.strike.values)
            iv = list(df['impliedVolatility'].values * 100.)
            xs_matched.extend([i] * len(strikes))
            ys_matched.extend(strikes)
            zs_matched.extend(iv)

    # Use NumPy to get unique strikes and counts
    unique, counts = np.unique(ys_matched, return_counts=True)
    uniq_strikes = unique[counts == len(expiration_dates)]
    if len(uniq_strikes) == 0:
        uniq_strikes = unique[counts >= max(1, len(expiration_dates) // 2)]
        if len(uniq_strikes) == 0:
            uniq_strikes = unique
            if len(uniq_strikes) == 0:
                return None

    mask = np.isin(ys_matched, uniq_strikes)
    xs_matched = np.array(xs_matched)[mask]
    ys_matched = np.array(ys_matched)[mask]
    zs_matched = np.array(zs_matched)[mask]
    shape = (len(expiration_dates), uniq_strikes.shape[0])
    return xs_matched, ys_matched, zs_matched, shape

def plot_surface(chains: dict, expiration_dates: list) -> go.Figure:
    """
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
    """
    data = _prepare_surface_data(chains, expiration_dates)
    if data is None:
        fig = go.Figure()
        fig.update_layout(
            title='Volatility Surface - No shared strikes available',
            annotations=[{'text': 'Insufficient data to plot volatility surface',
                          'showarrow': False, 'xref': "paper",
                          'yref': "paper", 'x': 0.5, 'y': 0.5}]
        )
        return fig

    xs_matched, ys_matched, zs_matched, shape = data

    fig = go.Figure(data=[go.Surface(
        x=xs_matched.reshape(shape),
        y=ys_matched.reshape(shape),
        z=zs_matched.reshape(shape),
        cmin=0,
        cmax=zs_matched.max() + 10
    )])
    fig.update_layout(
        title={'text': 'Volatility Surface'},
        autosize=True,
        width=500,
        height=500,
        scene={
            'xaxis_title': 'Expiration Date',
            'yaxis_title': 'Strike Price ($)',
            'zaxis_title': 'IV (%)',
            'xaxis': {
                'tickmode': 'array',
                'tickvals': xs_matched.reshape(shape)[:, 0][::2],
                'ticktext': expiration_dates[::2],
                'tickfont': {'size': 10}
            },
        },
    )
    return fig

@st.cache_data()
def calc_unusual_table(df_full_chain: pd.DataFrame,
                       show_itm: bool = True,
                       oi_min: int = 1_000) -> pd.DataFrame:
    """
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
    """
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

def colorize_rows(row, df_full_chain_proc):
    """
    Description:
        This function styles the unusual options activity dataframe for display
        in the Streamlit app. It applies a color gradient to the rows based on
        the unusual activity values.

    Parameters:
        row: Current row being processed.
        df_full_chain_proc: The processed DataFrame with unusual activity data.

    Returns:
        list: A list of CSS color styles for each cell in the row.
    """
    if df_full_chain_proc["unusual_activity"].max() == df_full_chain_proc["unusual_activity"].min():
        norm = 0.5
    else:
        norm = (row.unusual_activity - df_full_chain_proc["unusual_activity"].min()) / \
            (df_full_chain_proc["unusual_activity"].max()
             - df_full_chain_proc["unusual_activity"].min())
    color = f'background-color: rgba({255 * (1 - norm)}, {255 * norm}, 0, 0.5)'
    return [color] * len(row)

@st.cache_data()
def generate_widgets(ticker: str):
    """
    Description:
        This function generates the TradingView widgets for the given ticker.
        It returns the HTML code for the widgets.

    Parameters:
        ticker (str): The stock ticker symbol.

    Returns:
        tuple: A tuple containing three HTML strings for different TradingView widgets.
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
                "width": 280,
                "isTransparent": true,
                "height": 500,
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
                    "container_id": "tradingview_chart"
                }});
            </script>
        </div>
        """

    symbol_info = f"""
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
                    <span class="blue-text">Track all markets on TradingView</span>
                </a>
            </div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js" async>
            {{
            "symbol": "{ticker}",
            "width": "100%",
            "locale": "en",
            "colorTheme": "dark",
            "isTransparent": true
            }}
            </script>
        </div>
        """

    financial_info = f"""
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
                    <span class="blue-text">Track all markets on TradingView</span>
                </a>
            </div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-financials.js" async>
            {{
            "isTransparent": true,
            "largeChartUrl": "",
            "displayMode": "adaptive",
            "width": "100%",
            "height": "550",
            "colorTheme": "dark",
            "symbol": "{ticker}",
            "locale": "en"
            }}
            </script>
        </div>
        """

    company_profile = f"""
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
                    <span class="blue-text">Track all markets on TradingView</span>
                </a>
            </div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-profile.js" async>
            {{
            "width": "100%",
            "height": "550",
            "isTransparent": true,
            "colorTheme": "dark",
            "symbol": "{ticker}",
            "locale": "en"
            }}
            </script>
        </div>
        """

    market_overview = """
        <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>
        {{
        "colorTheme": "dark",
        "dateRange": "12M",
        "showChart": true,
        "locale": "en",
        "largeChartUrl": "",
        "isTransparent": true,
        "showSymbolLogo": true,
        "showFloatingTooltip": true,
        "width": "500",
        "height": "550",
        "plotLineColorGrowing": "rgba(41, 98, 255, 1)",
        "plotLineColorFalling": "rgba(41, 98, 255, 1)",
        "gridLineColor": "rgba(242, 242, 242, 0)",
        "scaleFontColor": "rgba(219, 219, 219, 1)",
        "belowLineFillColorGrowing": "rgba(41, 98, 255, 0.12)",
        "belowLineFillColorFalling": "rgba(41, 98, 255, 0.12)",
        "belowLineFillColorGrowingBottom": "rgba(41, 98, 255, 0)",
        "belowLineFillColorFallingBottom": "rgba(0, 0, 255, 0)",
        "symbolActiveColor": "rgba(255, 152, 0, 0.12)",
        "tabs": [
            {{
            "title": "Indices",
            "symbols": [
                {{
                "s": "FOREXCOM:SPXUSD",
                "d": "S&P 500 Index"
                }},
                {{
                "s": "FOREXCOM:NSXUSD",
                "d": "US 100 Cash CFD"
                }},
                {{
                "s": "FOREXCOM:DJI",
                "d": "Dow Jones Industrial Average Index"
                }},
                {{
                "s": "INDEX:NKY",
                "d": "Japan 225"
                }},
                {{
                "s": "INDEX:DEU40",
                "d": "DAX Index"
                }},
                {{
                "s": "FOREXCOM:UKXGBP",
                "d": "FTSE 100 Index"
                }}
            ],
            "originalTitle": "Indices"
            }},
            {{
            "title": "Futures",
            "symbols": [
                {{
                "s": "CME_MINI:ES1!",
                "d": "S&P 500"
                }},
                {{
                "s": "CME:6E1!",
                "d": "Euro"
                }},
                {{
                "s": "COMEX:GC1!",
                "d": "Gold"
                }},
                {{
                "s": "NYMEX:CL1!",
                "d": "WTI Crude Oil"
                }},
                {{
                "s": "NYMEX:NG1!",
                "d": "Gas"
                }},
                {{
                "s": "CBOT:ZC1!",
                "d": "Corn"
                }}
            ],
            "originalTitle": "Futures"
            }},
            {{
            "title": "Bonds",
            "symbols": [
                {{
                "s": "CBOT:ZB1!",
                "d": "T-Bond"
                }},
                {{
                "s": "CBOT:UB1!",
                "d": "Ultra T-Bond"
                }},
                {{
                "s": "EUREX:FGBL1!",
                "d": "Euro Bund"
                }},
                {{
                "s": "EUREX:FBTP1!",
                "d": "Euro BTP"
                }},
                {{
                "s": "EUREX:FGBM1!",
                "d": "Euro BOBL"
                }}
            ],
            "originalTitle": "Bonds"
            }},
            {{
            "title": "Forex",
            "symbols": [
                {{
                "s": "FX:EURUSD",
                "d": "EUR to USD"
                }},
                {{
                "s": "FX:GBPUSD",
                "d": "GBP to USD"
                }},
                {{
                "s": "FX:USDJPY",
                "d": "USD to JPY"
                }},
                {{
                "s": "FX:USDCHF",
                "d": "USD to CHF"
                }},
                {{
                "s": "FX:AUDUSD",
                "d": "AUD to USD"
                }},
                {{
                "s": "FX:USDCAD",
                "d": "USD to CAD"
                }}
            ],
            "originalTitle": "Forex"
            }}
        ]
        }}
        </script>
        </div>
        """

    stock_overview = """
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-hotlists.js" async>
            {{
            "colorTheme": "dark",
            "dateRange": "12M",
            "exchange": "US",
            "showChart": true,
            "locale": "en",
            "largeChartUrl": "",
            "isTransparent": true,
            "showSymbolLogo": true,
            "showFloatingTooltip": true,
            "width": "500",
            "height": "550",
            "plotLineColorGrowing": "rgba(41, 98, 255, 1)",
            "plotLineColorFalling": "rgba(41, 98, 255, 1)",
            "gridLineColor": "rgba(152, 152, 152, 0)",
            "scaleFontColor": "rgba(219, 219, 219, 1)",
            "belowLineFillColorGrowing": "rgba(41, 98, 255, 0.12)",
            "belowLineFillColorFalling": "rgba(41, 98, 255, 0.12)",
            "belowLineFillColorGrowingBottom": "rgba(41, 98, 255, 0)",
            "belowLineFillColorFallingBottom": "rgba(41, 98, 255, 0)",
            "symbolActiveColor": "rgba(41, 98, 255, 0.12)"
            }}
            </script>
        </div>
    """

    running_ticker = """
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
            {{
            "symbols": [
                {{
                "description": "",
                "proName": "NASDAQ:TSLA"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:NVDA"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:AAPL"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:AMZN"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:META"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:PLTR"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:MSFT"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:MSTR"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:AMD"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:GOOGL"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:NFLX"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:INTC"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:INTC"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:COIN"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:GOOG"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:SMCI"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:AVGO"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:HOOD"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:MU"
                }},
                {{
                "description": "",
                "proName": "NYSE:JPM"
                }},
                {{
                "description": "",
                "proName": "NASDAQ:ADBE"
                }}
            ],
            "showSymbolLogo": true,
            "isTransparent": true,
            "displayMode": "regular",
            "colorTheme": "dark",
            "locale": "en"
            }}
            </script>
        </div>
        """

    heatmap = """
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
            {{
            "exchanges": [],
            "dataSource": "SPX500",
            "grouping": "sector",
            "blockSize": "market_cap_basic",
            "blockColor": "change",
            "locale": "en",
            "symbolUrl": "",
            "colorTheme": "dark",
            "hasTopBar": true,
            "isDataSetEnabled": true,
            "isZoomEnabled": true,
            "hasSymbolTooltip": true,
            "isMonoSize": false,
            "width": "100%",
            "height": 500
            }}
            </script>
        </div>
        """

    return (
    single_ticker_widget,
    tech_perf,
    tv_advanced_plot,
    symbol_info,
    financial_info,
    company_profile,
    market_overview,
    stock_overview,
    running_ticker,
    heatmap
    )

def get_options_data(ticker: str):
    """
    Description:
        This function retrieves the options chain data for the given ticker.
        It returns the options chain dataframes for calls and puts, the expiration
        dates, and the underlying price.

    Parameters:
        ticker (str): The stock ticker symbol.

    Returns:
        tuple: A tuple containing the expiration dates, call options dataframe,
        put options dataframe, call options dictionary, put options dictionary,
        and underlying price.
    """

    # Initialize return variables
    yfticker = yf.Ticker(ticker)
    expiration_dates = list(yfticker.options)  # Convert to list to ensure serializability

    # Initialize empty DataFrames and dictionaries
    df_full_chain_calls = None
    df_full_chain_puts = None
    df_full_chain_calls_dict = {}
    df_full_chain_puts_dict = {}
    underlying_price = None

    if expiration_dates:
        # Get the underlying price from first option chain
        try:
            opt_chain = yfticker.option_chain(expiration_dates[0])
            underlying_price = float(opt_chain.underlying['regularMarketPrice'])
        except (KeyError, AttributeError, IndexError):
            # Fallback: try to get current price directly
            try:
                hist = yfticker.history(period="1d")
                if not hist.empty:
                    underlying_price = float(hist['Close'].iloc[-1])
                else:
                    underlying_price = None
            except (KeyError, IndexError, AttributeError) as err:
                logging.warning("Failed to retrieve historical price data: %s", err)
                underlying_price = None


        # Retrieve option chains for all expiration dates
        for e in expiration_dates:
            try:
                opt = yfticker.option_chain(e)
                calls, puts = opt.calls, opt.puts

                if df_full_chain_calls is not None:
                    df_full_chain_calls = pd.concat([df_full_chain_calls, calls])
                else:
                    df_full_chain_calls = calls.copy()

                if df_full_chain_puts is not None:
                    df_full_chain_puts = pd.concat([df_full_chain_puts, puts])
                else:
                    df_full_chain_puts = puts.copy()

                df_full_chain_calls_dict[e] = calls
                df_full_chain_puts_dict[e] = puts
            except (KeyError, ValueError, AttributeError) as err:
                logging.warning("Skipping expiration due to error: %s", err)
                # Skip this expiration date if there's an error
                continue

    # Return all serializable data but not the yfticker object
    return (
    expiration_dates,
    df_full_chain_calls,
    df_full_chain_puts,
    df_full_chain_calls_dict,
    df_full_chain_puts_dict,
    underlying_price,
    )


def get_data(ticker: str):
    """
    Description:
        Wrapper for get_options_data to match the expected signature in tests.
    Parameters:
        ticker (str): The stock ticker symbol.
    Returns:
        tuple: A tuple containing call options dictionary, put options dictionary,
        call options dataframe, put options dataframe, expiration dates,
        underlying price, and valid_ticker flag.
    """
    try:
        expiration_dates,df_calls,df_puts,df_calls_dict,df_puts_dict,atm=get_options_data(ticker)
        valid_ticker = bool(expiration_dates and atm is not None)
        return df_calls_dict, df_puts_dict, df_calls, df_puts, expiration_dates, atm, valid_ticker
    except (KeyError, ValueError, AttributeError) as err:
        logging.error("Error retrieving options data : %s", err)
        return None, None, None, None, [], None, False


def create_open_interest_chart(calls: pd.DataFrame,
                               puts: pd.DataFrame,
                               atm: float) -> go.Figure:
    """
    Description:
        This function creates a plotly figure showing the Open Interest based histograms
        for call and put options. This is an alias for create_oi_hists for compatibility.
    Parameters:
        calls (pd.DataFrame): DataFrame containing call options data.
        puts (pd.DataFrame): DataFrame containing put options data.
        ATM (float): The at-the-money strike price (current stock price).
    Returns:
        go.Figure: A plotly figure object containing the Open Interest histograms
         for calls and puts.
    """
    return create_oi_hists(calls, puts, atm)

def create_volume_chart(calls: pd.DataFrame,
                        puts: pd.DataFrame,
                        atm: float) -> go.Figure:
    """
    Description:
        This function creates a plotly figure showing the Volume based histograms
        for call and put options. This is an alias for create_vol_hists for compatibility.
    Parameters:
        calls (pd.DataFrame): DataFrame containing call options data.
        puts (pd.DataFrame): DataFrame containing put options data.
        ATM (float): The at-the-money strike price (current stock price).
    Returns:
        go.Figure: A plotly figure object containing the volume histograms
         for calls and puts.
    """
    return create_vol_hists(calls, puts, atm)

def create_iv_chart(calls: pd.DataFrame,
                   puts: pd.DataFrame) -> go.Figure:
    """
    Description:
        This function creates a plotly figure showing the implied volatility smile for
        call and put options. This is a wrapper around create_iv_smile to maintain
        compatibility with the function name used in the app.
    Parameters:
        calls (pd.DataFrame): DataFrame containing call options data.
        puts (pd.DataFrame): DataFrame containing put options data.
    Returns:
        go.Figure: A plotly figure object containing the implied volatility smile.
    """
    # Get the current price
    if len(calls) > 0 and 'inTheMoney' in calls.columns:
        atm_call = calls.iloc[(calls['strike'] - calls.iloc[0]['lastPrice']).abs().idxmin()]
        atm = float(atm_call['strike'])
    else:
        # Default to middle strike if can't determine ATM
        if len(calls) > 0:
            strikes = sorted(calls['strike'].unique())
            atm = float(strikes[len(strikes) // 2])
        else:
            atm = 100.0  # Default value

    return create_iv_smile(calls, puts, atm)

def get_tradingview_widgets(ticker: str) -> tuple:
    """
    Description:
        Create TradingView widget HTML for a given ticker.
        This is an alias for generate_widgets to maintain compatibility.
    Parameters:
        ticker (str): Stock ticker symbol.
    Returns:
        tuple: A tuple of (single_ticker_widget, tech_perf, tv_advanced_plot) HTML strings.
    """
    return generate_widgets(ticker)

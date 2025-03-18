import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

@st.cache_data()
def create_iv_smile(calls: pd.DataFrame, 
                    puts: pd.DataFrame, 
                    ATM: float) -> go.Figure:
    

    if not isinstance(calls, pd.DataFrame):
        raise TypeError('calls must be a dataframe')
    
    if not isinstance(puts, pd.DataFrame):
        raise TypeError('puts must be a dataframe')
    
    if not isinstance(ATM, float):
        raise TypeError('ATM must be a float')
    
    if ATM < 0:
        raise ValueError('ATM must be gte 0')
    
    call_iv = calls['impliedVolatility'].values
    call_iv[np.isnan(call_iv)] = 0.

    put_iv = puts['impliedVolatility'].values
    put_iv[np.isnan(put_iv)] = 0.

    max_iv = np.maximum(call_iv.max(), put_iv.max())
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=calls.strike, y=calls['impliedVolatility']*100,
                                mode='lines+markers', name='Call IV', line=dict(color='#00C66B')))
    fig.add_trace(go.Scatter(x=puts.strike, y=puts['impliedVolatility']*100,
                                mode='lines+markers', name='Put IV',  line=dict(color='#D9534F')))
    fig.add_shape(
        type="rect",
        x0=0, x1=ATM,
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
                     ATM: float) -> go.Figure:
    '''
    create docstring here...


    '''
    if not isinstance(calls, pd.DataFrame):
        raise TypeError('calls must be a dataframe')
    
    if not isinstance(puts, pd.DataFrame):
        raise TypeError('puts must be a dataframe')
    
    if not isinstance(ATM, float):
        raise TypeError('ATM must be a float')
    
    if ATM < 0:
        raise ValueError('ATM must be gte 0')
    
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
        x0=0, x1=ATM,
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
                    ATM: float) -> go.Figure:
    '''
    create docstring here...


    '''
    if not isinstance(calls, pd.DataFrame):
        raise TypeError('calls must be a dataframe')
    
    if not isinstance(puts, pd.DataFrame):
        raise TypeError('puts must be a dataframe')
    
    if not isinstance(ATM, float):
        raise TypeError('ATM must be a float')
    
    if ATM < 0:
        raise ValueError('ATM must be gte 0')

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
    return fig

@st.cache_data()
def plot_surface(df_full_chain_side_dict: dict,
                 expiration_dates: list) -> go.Figure:
    """
    add docstring here
    
    """
    if len(expiration_dates) == 0:
        raise ValueError('Expiration dates is empty.')
    
    if len(list(df_full_chain_side_dict.keys())) == 0:
        raise ValueError('Chain dataframes contains no keys, is empty.')

    if not isinstance(df_full_chain_side_dict, dict):
        raise TypeError('Enter a dictionary for df_full_chain_side_dict input arg')
    
    if not isinstance(expiration_dates, list):
        raise TypeError('Enter a list for expiration_dates input arg')

    xs, ys_calls, zs_calls = [], [], []
    for e in expiration_dates:
        xs.append(e)
        calls_e = df_full_chain_side_dict[e]        
        if len(calls_e) > 0:
            ys_calls.append(list(calls_e.strike.values))
            zs_calls.append(list(calls_e['impliedVolatility'].values * 100.))

    unique_xs = np.arange(len(xs))
    xs_matched, ys_matched, zs_matched = [], [], []
    for i, (y_, z_) in enumerate(zip(ys_calls, zs_calls)):
        xs_matched.extend([unique_xs[i]]*len(y_))
        ys_matched.extend(y_)
        zs_matched.extend(z_)
        
    uniq_strikes = dict()
    shared_strikes = []
    for y in ys_calls:
        for y_ in y:
            if y_ not in uniq_strikes:
                uniq_strikes[y_] = 1
            else:
                uniq_strikes[y_] += 1
                
    reduced_strikes = np.array(list(uniq_strikes.keys()))[np.array(list(uniq_strikes.values()))==len(xs)]
    x_filtered = np.array(xs_matched)[np.isin(ys_matched, reduced_strikes)]
    y_filtered = np.array(ys_matched)[np.isin(ys_matched, reduced_strikes)]
    z_filtered = np.array(zs_matched)[np.isin(ys_matched, reduced_strikes)]

    fig = go.Figure(data=[go.Surface(z=z_filtered.reshape((len(ys_calls),reduced_strikes.shape[0])), 
                                    x=x_filtered.reshape((len(ys_calls),reduced_strikes.shape[0])), 
                                    y=y_filtered.reshape((len(ys_calls),reduced_strikes.shape[0])),
                                    cmin=0, 
                                    cmax=z_filtered.max()+10)])

    fig.update_layout(
        title=dict(text='Volatility Surface'),
        autosize=True,
        width=500, 
        height=500,
        scene=dict(
            xaxis_title='Expiration Date',
            yaxis_title='Strike Price ($)',
            zaxis_title='IV (%)',
            xaxis = dict(
                        tickmode='array',
                        tickvals = x_filtered.reshape((len(ys_calls),reduced_strikes.shape[0]))[:,0][::2],
                        ticktext = xs[::2],
                        tickfont={'size':10}
                        ),
        ),
    )
    return fig

@st.cache_data()
def calc_unusual_table(df_full_chain: pd.DataFrame, 
                       show_itm: bool = True,
                       oi_min: int = 1_000) -> pd.DataFrame:
    """
    add docstring here
    
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
    add docstring here
    
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
    '''
    get data
    
    '''
    # get option chain and proc
    yfticker = yf.Ticker(ticker)
    expiration_dates = list(yfticker.options)

    if len(expiration_dates):
    
        money_level = float(yfticker.option_chain(expiration_dates[0]).underlying['regularMarketPrice'])

        # show unusual activity table
        df_full_chain_calls = None
        df_full_chain_puts = None
        df_full_chain_calls_dict = dict()
        df_full_chain_puts_dict = dict()

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

        return df_full_chain_calls_dict, df_full_chain_puts_dict, \
            df_full_chain_calls, df_full_chain_puts, expiration_dates, \
                money_level, True
    else:
        st.error(f'Unable to find option chain for ticker: {ticker}', icon="🚨")
        return None, None, None, None, None, None, False

def main():
    '''
    docstring here
    
    '''
    ticker_cols = st.columns((3,4), gap='small')

    with ticker_cols[0]:
        ticker = st.text_input("Enter stock ticker:", value=None, placeholder='e.g. NVDA, AAPL, AMZN')

    if ticker is not None:

        ticker = ticker.upper()

        df_calls_dict, df_puts_dict, df_calls, \
                df_puts, expiration_dates, ATM, \
                     valid_ticker = get_data(ticker)
        
        if valid_ticker:

            single_ticker_widget, tech_perf, tv_advanced_plot = generate_widgets(ticker)

            with ticker_cols[1]:
                st.components.v1.html(single_ticker_widget, height=100)
            
            with st.sidebar:
                st.components.v1.html(tech_perf, height=400)

            st.divider()
            st.write(f"#### Unusual Options Activity")

            col_activity = st.columns((4,4), gap='small')
            with col_activity[0]:
                
                st.write(f"#### Calls") 
                
                oi_min_calls = st.number_input("Minumum OI", min_value=1, 
                                            key='oi_min_calls',
                                            value=1_000,
                                        help='Minumum Open Interest to consider \
                                            when computing unusual options activity.')
                
                show_itm_calls = st.checkbox("Show ITM", value=False, key='show_itm',
                                    help='Only show in-the-money (ITM) contracts, \
                                        otherwise show only out-of-money.')

                df_full_chain_calls_proc = calc_unusual_table(df_calls, show_itm_calls, oi_min_calls)

                def colorize_rows(row):
                    norm = (row.unusual_activity - df_full_chain_calls_proc["unusual_activity"].min()) / \
                        (df_full_chain_calls_proc["unusual_activity"].max() - \
                            df_full_chain_calls_proc["unusual_activity"].min())
                    color = f'background-color: rgba({255 * (1 - norm)}, \
                        {255 * norm}, 0, 0.5)'
                    return [color] * len(row)

                styled_df_calls = df_full_chain_calls_proc.style.apply(colorize_rows, axis=1)
                st.dataframe(styled_df_calls)

            with col_activity[1]:
                st.write(f"#### Puts") 
                oi_min_puts = st.number_input("Minumum OI", min_value=1, 
                                            key='oi_min_puts',
                                            value=1_000,
                                            help='Minumum Open Interest to consider when \
                                                computing unusual options activity.')
                show_itm_puts = st.checkbox("Show ITM", value=False, key='show_itm_puts',
                                            help='Only show in-the-money (ITM) contracts, \
                                                otherwise show only out-of-money.')
                df_full_chain_puts_proc = calc_unusual_table(df_puts, show_itm_puts, oi_min_puts)
                
                def colorize_rows(row):
                    norm = (row.unusual_activity - df_full_chain_puts_proc["unusual_activity"].min()) / \
                        (df_full_chain_puts_proc["unusual_activity"].max() - \
                            df_full_chain_puts_proc["unusual_activity"].min())
                    color = f'background-color: rgba({255 * (1 - norm)}, \
                        {255 * norm}, 0, 0.5)'
                    return [color] * len(row)

                styled_df_puts = df_full_chain_puts_proc.style.apply(colorize_rows, axis=1)
                st.dataframe(styled_df_puts)

            st.divider()
            st.write(f"#### Chain Analysis")

            exp_date = st.selectbox(
                    "Select an expiration date",
                    expiration_dates,
            )        
            calls = df_calls_dict[exp_date]
            puts = df_puts_dict[exp_date]
            calls = calls.sort_values(by='strike')
            puts = puts.sort_values(by='strike')

            col_inner = st.columns((4,4), gap='small')

            with col_inner[0]:
                oi_hist = create_oi_hists(calls, puts, ATM)
                st.plotly_chart(oi_hist)

            with col_inner[1]:
                vol_hists = create_vol_hists(calls, puts, ATM)
                st.plotly_chart(vol_hists)

            col_vol = st.columns((4,4), gap='small')

            # plot IV bar chart (overlap calls and puts)
            with col_vol[0]:
                iv_smile = create_iv_smile(calls, puts, ATM)
                st.plotly_chart(iv_smile)

            # volatility surface
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

            # add the chart widget
            st.write(f"#### Underlying Price Chart") 
            st.components.v1.html(tv_advanced_plot, height=400)

# call main
main()

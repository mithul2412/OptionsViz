import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

@st.cache_data()
def plot_surface(df_full_chain_side_dict, 
                    expiration_dates):
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
                        tickmode='array', #change 1
                        tickvals = x_filtered.reshape((len(ys_calls),reduced_strikes.shape[0]))[:,0][::2], #change 2
                        ticktext = xs[::2], #change 3,
                        tickfont={'size':10}
                        ),
        ),
    )
    return fig

@st.cache_data()
def calc_unusual_table(df_full_chain, show_itm=False, oi_min=1_000):
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

########################################################################################################################

ticker_cols = st.columns((3,4), gap='small')

with ticker_cols[0]:
    ticker = st.text_input("Enter stock ticker:", value=None, placeholder='e.g. NVDA, AAPL, AMZN')

if ticker is not None:

    ticker = ticker.upper()

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

    with ticker_cols[1]:
        st.components.v1.html(single_ticker_widget, height=100)

    tech_perf  = f'''
        <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>
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
    
    with st.sidebar:
        st.components.v1.html(tech_perf, height=400)

    # get option chain and proc
    yfticker = yf.Ticker(ticker)
    expiration_dates = yfticker.options

    ######### show unusual activity table
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

    st.divider()
    
    st.write(f"#### Unusual Options Activity")

    col_activity = st.columns((4,4), gap='small')
    with col_activity[0]:
        st.write(f"#### Calls") 
        oi_min_calls = st.number_input("Minumum OI", min_value=1, value=1_000,
                                 help='Minumum Open Interest to consider \
                                    when computing unusual options activity.')
        show_itm_calls = st.checkbox("Show ITM", value=False, key='show_itm',
                               help='Only show in-the-money (ITM) contracts, \
                                otherwise show only out-of-money.')
        df_full_chain_calls_proc = calc_unusual_table(df_full_chain_calls, show_itm_calls, oi_min_calls)

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
        oi_min_puts = st.number_input("Minumum OI", min_value=1, key=2, value=1_000,
                                      help='Minumum Open Interest to consider when \
                                        computing unusual options activity.')
        show_itm_puts = st.checkbox("Show ITM", value=False, key='show_itm_puts',
                                    help='Only show in-the-money (ITM) contracts, \
                                        otherwise show only out-of-money.')
        df_full_chain_puts_proc = calc_unusual_table(df_full_chain_puts, show_itm_puts, oi_min_puts)
        
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

    opt = yfticker.option_chain(exp_date)
    calls = opt.calls
    puts = opt.puts

    calls = calls.sort_values(by='strike')
    puts = puts.sort_values(by='strike')

    ###########
    # create widget
    col_inner = st.columns((4,4), gap='small')
    ATM = opt.underlying['regularMarketPrice']
    span_end = int((calls.strike-ATM).abs().argmin())

    with col_inner[0]:
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
        st.plotly_chart(fig)

    with col_inner[1]:
        max_vol = np.maximum(calls.volume.values.max(), puts.volume.values.max())

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=puts['strike'],
            y=puts['volume'],
            name='Puts',
            orientation='v',
            marker_color='#D9534F'
        ))

        fig2.add_trace(go.Bar(
            x=calls['strike'],
            y=calls['volume'],
            name='Calls',
            orientation='v',
            marker_color='#00C66B', marker_opacity=0.5
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
            bargap=0.01,  # Control the gap between bars (smaller value = thicker bars)
            bargroupgap=0.01, # Control the gap between groups of bars (if stacked or grouped),
        )
        st.plotly_chart(fig2)

        ################################################

    col_vol = st.columns((4,4), gap='small')

    with col_vol[0]:
        # plot IV bar chart (overlap calls and puts)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=calls.strike, y=calls['impliedVolatility']*100,
                                  mode='lines+markers', name='Call IV', line=dict(color='#00C66B')))
        fig.add_trace(go.Scatter(x=puts.strike, y=puts['impliedVolatility']*100,
                                 mode='lines+markers', name='Put IV',  line=dict(color='#D9534F')))
        fig.update_layout(title="Implied Volatility (%) by Strike ($); 'Volatility Smile'",
                          xaxis_title="Strike Price ($)", yaxis_title="IV (%)")
        st.plotly_chart(fig)

    # volatility surface
    with col_vol[1]:
        show_calls = st.checkbox("Calls", value=True, key='volatility_surface_calls',
                                help='Show surface for calls (checked) or puts (unchecked)')
        
        if show_calls:
            surface_fig = plot_surface(df_full_chain_calls_dict, expiration_dates)
        else:
            surface_fig = plot_surface(df_full_chain_puts_dict, expiration_dates)
        st.plotly_chart(surface_fig, use_container_width=True)

    # add the chart widget
    st.write(f"#### Underlying Price Chart") 
    st.components.v1.html(tv_advanced_plot, height=400)

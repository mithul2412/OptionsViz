import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

ticker_widget = '''
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
  {
  "symbols": [
    {
      "proName": "FOREXCOM:SPXUSD",
      "title": "S&P 500 Index"
    },
    {
      "proName": "FOREXCOM:NSXUSD",
      "title": "US 100 Cash CFD"
    },
    {
      "proName": "FX_IDC:EURUSD",
      "title": "EUR to USD"
    },
    {
      "proName": "BITSTAMP:BTCUSD",
      "title": "Bitcoin"
    },
    {
      "proName": "BITSTAMP:ETHUSD",
      "title": "Ethereum"
    }
  ],
  "height": "10%",
  "showSymbolLogo": true,
  "isTransparent": false,
  "displayMode": "adaptive",
  "colorTheme": "dark",
  "locale": "en"
}
  </script>
</div>
'''

ticker_cols = st.columns((4,4), gap='small')

with ticker_cols[0]:
    ticker = st.text_input("Enter stock ticker:", value=None, placeholder='e.g. NVDA, AAPL, AMZN')

if ticker is not None:

    ticker = ticker.upper()

    with ticker_cols[1]:        
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
        news_widget = f'''
                <div class="tradingview-widget-container">
                <div class="tradingview-widget-container__widget"></div>
                <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>
                <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>
                {{
                "feedMode": "symbol",
                "symbol": "{ticker}",
                "isTransparent": true,
                "displayMode": "compact",
                "width": "400",
                "height": "400",
                "autosize": true,
                "colorTheme": "dark",
                "locale": "en"
                }}
                </script>
                </div>
        '''

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
        with st.sidebar:
            st.components.v1.html(single_ticker_widget, height=100)
            st.components.v1.html(tech_perf, height=400)
            st.components.v1.html(news_widget, height=400)



    # get option chain and proc
    yfticker = yf.Ticker(ticker)
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
                                 help='Minumum Open Interest to consider \
                                    when computing unusual options activity.')
        show_itm = st.checkbox("Show ITM", value=False,
                               help='Only show in-the-money (ITM) contracts, \
                                otherwise show only out-of-money.')

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

        def colorize_rows(row):
            norm = (row.unusual_activity - df_full_chain_calls["unusual_activity"].min()) / \
                (df_full_chain_calls["unusual_activity"].max() - \
                 df_full_chain_calls["unusual_activity"].min())
            color = f'background-color: rgba({255 * (1 - norm)}, \
                {255 * norm}, 0, 0.5)'
            return [color] * len(row)

        # Apply styling to whole row
        df_full_chain_calls = df_full_chain_calls.reset_index(drop=True)
        styled_df = df_full_chain_calls.style.apply(colorize_rows, axis=1)
        st.dataframe(styled_df)

    with col_activity[1]:
        oi_min_puts = st.number_input("Minumum OI", min_value=1, key=2, value=1_000,
                                      help='Minumum Open Interest to consider when \
                                        computing unusual options activity.')
        show_itm_puts = st.checkbox("Show ITM", value=False, key=3,
                                    help='Only show in-the-money (ITM) contracts, \
                                        otherwise show only out-of-money.')

        df_full_chain_puts = df_full_chain_puts[df_full_chain_puts.volume != 0.]
        df_full_chain_puts = df_full_chain_puts[df_full_chain_puts.openInterest != 0.]
        df_full_chain_puts = df_full_chain_puts[~pd.isna(df_full_chain_puts.volume)]
        df_full_chain_puts = df_full_chain_puts[~pd.isna(df_full_chain_puts.openInterest)]
        df_full_chain_puts = df_full_chain_puts[df_full_chain_puts.openInterest >= oi_min_puts]
        df_full_chain_puts['unusual_activity'] = df_full_chain_puts.volume / \
            df_full_chain_puts.openInterest
        df_full_chain_puts = df_full_chain_puts.sort_values('unusual_activity', ascending=False)
        df_full_chain_puts = df_full_chain_puts[df_full_chain_puts.unusual_activity > 1]
        df_full_chain_puts = df_full_chain_puts[df_full_chain_puts.inTheMoney==show_itm_puts]
        df_full_chain_puts['spread'] = df_full_chain_puts.ask - df_full_chain_puts.bid
        df_full_chain_puts = df_full_chain_puts[['contractSymbol', 'strike',
                                                 'lastPrice','spread','percentChange',
                                                 'volume','openInterest',
                                                 'impliedVolatility','unusual_activity']]

        def colorize_rows_puts(row):
            norm = (row.unusual_activity - df_full_chain_puts["unusual_activity"].min()) / \
                (df_full_chain_puts["unusual_activity"].max() - \
                 df_full_chain_puts["unusual_activity"].min())
            color = f'background-color: rgba({255 * (1 - norm)}, \
                {255 * norm}, 0, 0.5)'  # Red to Green
            return [color] * len(row)

        # Apply styling to whole row
        df_full_chain_puts = df_full_chain_puts.reset_index(drop=True)
        styled_df_puts = df_full_chain_puts.style.apply(colorize_rows_puts, axis=1)
        st.dataframe(styled_df_puts)

    st.divider()

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
            marker_color='#00C66B'#, marker_opacity=0.5
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
            bargap=0.01,  # Control the gap between bars (smaller value = thicker bars)
            bargroupgap=0.01, # Control the gap between groups of bars (if stacked or grouped),
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
            marker_color='#00C66B'#, marker_opacity=0.5
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

    with col_vol[1]:
        # # volatility surface
        # xs, ys, zs = [], [], []
        # for e in expiration_dates:
        #     xs.append(expiration_dates)

        #     opt = yfticker.option_chain(e)
        #     calls = opt.calls
        #     ys.append(calls.strike.values)

        #     zs.append(calls['impliedVolatility'].values * 100.)

        # fig = go.Figure(data=[go.Surface(z=zs, x=xs, y=ys)])
        # fig.update_layout(title=dict(text='Volatility Surface'), autosize=False,
        #                 width=500, height=500,
        #                 margin=dict(l=65, r=50, b=65, t=90),
        #                 scene=dict(
        #                         xaxis_title='Expiration Date',
        #                         yaxis_title='Strike Price',
        #                         zaxis_title='IV (%)',
        #                     )
        #                 )
        # st.plotly_chart(fig)

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
    st.components.v1.html(tv_advanced_plot, height=400)

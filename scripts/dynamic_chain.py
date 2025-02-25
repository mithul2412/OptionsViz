import streamlit as st
import yfinance as yf

ticker = st.text_input("Enter stock ticker:", value=None, placeholder='e.g. NVDA, AAPL, AMZN')

if ticker is not None:

    ticker = ticker.upper()

    # Custom TradingView widget for stock chart (Users can navigate to options)
    tradingview_widget = """
    <div class="tradingview-widget-container">
        <div id="tradingview_chart"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
            new TradingView.widget({
                "width": "100%",
                "height": 600,
                "symbol": "NASDAQ:""" + ticker + """",
                "interval": "1",
                "timezone": "Etc/UTC",
                "theme": "light",
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

    # Embed the TradingView widget
    st.components.v1.html(tradingview_widget, height=600)
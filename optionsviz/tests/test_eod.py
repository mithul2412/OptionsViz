import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from eod_chain import (
    create_iv_smile, 
    create_vol_hists,
    create_oi_hists, 
    plot_surface, 
    calc_unusual_table, 
    generate_widgets, 
    get_data
)

class TestUtils(unittest.TestCase):

    # def test_iv_smile(self):

    # def test_vol_hist(self):

    # def test_create_oi_hists(self):

    # def test_plot_surface(self):

    # def test_calc_unusual_table(self):

    def test_generate_widgets(self):
        ticker = 'AAPL'

        single_ticker_widget_truth = f'''
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

        tech_perf_truth = f'''
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

        tv_advanced_plot_truth = f"""
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

        def remove_whitespace(html):
            return " ".join(html.strip().split())

        single_ticker_widget, tech_perf, tv_advanced_plot = generate_widgets(ticker)
        
        self.maxDiff = None
        self.assertEqual(remove_whitespace(single_ticker_widget), remove_whitespace(single_ticker_widget_truth))
        self.assertEqual(remove_whitespace(tech_perf), remove_whitespace(tech_perf_truth))
        self.assertEqual(remove_whitespace(tv_advanced_plot), remove_whitespace(tv_advanced_plot_truth))


    def test_get_data_valid(self):
        '''
        tests valid ticker data return and type check
        
        '''
        ticker = 'AAPL'
        df_calls_dict, df_puts_dict, df_calls, \
                df_puts, expiration_dates, ATM, \
                     valid_ticker = get_data(ticker)
        
        self.assertTrue(valid_ticker)
        self.assertIsInstance(df_calls_dict, dict)
        self.assertIsInstance(df_puts_dict, dict)
        self.assertIsInstance(df_calls, pd.DataFrame)
        self.assertIsInstance(df_puts, pd.DataFrame)
        self.assertIsInstance(expiration_dates, list)
        self.assertIsInstance(ATM, float)
        
    def test_get_data_invalid(self):
        '''
        tests INVALID ticker data return and type check
        
        '''
        ticker = 'dwakdjawdnawo'
        df_calls_dict, df_puts_dict, df_calls, \
                df_puts, expiration_dates, ATM, \
                     valid_ticker = get_data(ticker)
        
        self.assertFalse(valid_ticker)
        self.assertIsNone(df_calls_dict)
        self.assertIsNone(df_puts_dict)
        self.assertIsNone(df_calls)
        self.assertIsNone(df_puts)
        self.assertIsNone(expiration_dates)
        self.assertIsNone(ATM)


if __name__ == '__main__':
    unittest.main()
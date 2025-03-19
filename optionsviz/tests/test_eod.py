"""
Module Name: test_eod.py

Description:
    This module contains a series of testing functions for the 
    End-of-Day (EOD) streamlit page functionality, including:
    plots, graphs, and tables.

Author:
    Ryan J Richards

Created:
    03-18-2025

License:
    MIT
"""
import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from eod_chain import (
    create_iv_smile,
    create_vol_hists,
    create_oi_hists,
    plot_surface,
    calc_unusual_table,
    generate_widgets,
    get_data
)

class TestEODMethods(unittest.TestCase):
    """
    Class: TestEODMethods

    Description:
        Tests the functionality of the EOD streamlit page.

    Methods:
        test_get_data_invalid: Tests the get_data function with an invalid ticker.
        test_get_data_valid: Tests the get_data function with a valid ticker.
        test_iv_smile: Tests the create_iv_smile function with valid data.
        test_iv_smile_invalid: Tests the create_iv_smile function with invalid data.
        test_vol_hist: Tests the create_vol_hists function with valid data.
        test_vol_hist_invalid: Tests the create_vol_hists function with invalid data.
        test_oi_hist: Tests the create_oi_hists function with valid data.
        test_oi_hist_invalid: Tests the create_oi_hists function with invalid data.
        test_plot_surface_valid: Tests the plot_surface function with valid data.
        test_plot_surface_invalid: Tests the plot_surface function with invalid data.
        test_calc_unusual_table: Tests the calc_unusual_table function with valid data.
        test_calc_unusual_table_invalid: Tests the calc_unusual_table function with invalid data.
        test_generate_widgets: Tests the generate_widgets function with valid data.
    """

    def test_get_data_invalid(self):
        '''
        Tests the get_data function with an invalid ticker.
        '''
        ticker = 'dwakdjawdnawo'
        df_calls_dict, df_puts_dict, df_calls, \
                df_puts, expiration_dates, atm, \
                     valid_ticker = get_data(ticker)

        self.assertFalse(valid_ticker)
        self.assertIsNone(df_calls_dict)
        self.assertIsNone(df_puts_dict)
        self.assertIsNone(df_calls)
        self.assertIsNone(df_puts)
        self.assertTrue(expiration_dates==[])
        self.assertIsNone(atm)

    def test_get_data_valid(self):
        '''
        Tests the get_data function with a valid ticker.
        '''
        ticker = 'AAPL'
        df_calls_dict, df_puts_dict, df_calls, \
                df_puts, expiration_dates, atm, \
                     valid_ticker = get_data(ticker)

        self.assertTrue(valid_ticker)
        self.assertIsInstance(df_calls_dict, dict)
        self.assertIsInstance(df_puts_dict, dict)
        self.assertIsInstance(df_calls, pd.DataFrame)
        self.assertIsInstance(df_puts, pd.DataFrame)
        self.assertIsInstance(expiration_dates, list)
        self.assertIsInstance(atm, float)

    def test_iv_smile(self):
        '''
        Tests the create_iv_smile function with valid data.
        '''
        ticker = 'AAPL'
        df_calls_dict, df_puts_dict, _, \
                _, expiration_dates, atm, \
                     _ = get_data(ticker)

        calls = df_calls_dict[expiration_dates[0]]
        puts = df_puts_dict[expiration_dates[0]]
        atm = 150.
        iv_smile = create_iv_smile(calls, puts, atm)
        self.assertIsInstance(iv_smile, go.Figure)

    def test_iv_smile_invalid(self):
        '''
        Tests the create_iv_smile function with invalid data.
        '''
        self.assertRaises(TypeError, create_iv_smile, [], pd.DataFrame([]), 150.)
        self.assertRaises(TypeError, create_iv_smile,  pd.DataFrame([]), [], 150.)
        self.assertRaises(TypeError, create_iv_smile,  pd.DataFrame([]),
                          pd.DataFrame([]), int(150))
        self.assertRaises(ValueError, create_iv_smile,  pd.DataFrame([]),
                          pd.DataFrame([]), -100.0)

    def test_vol_hist(self):
        '''
        Tests the create_vol_hists function with valid data.
        '''
        ticker = 'AAPL'
        df_calls_dict, df_puts_dict, _, \
                _, expiration_dates, atm, \
                     _ = get_data(ticker)

        calls = df_calls_dict[expiration_dates[0]]
        puts = df_puts_dict[expiration_dates[0]]
        atm = 150.
        vol_hist = create_vol_hists(calls, puts, atm)
        self.assertIsInstance(vol_hist, go.Figure)

    def test_vol_hist_invalid(self):
        '''
        Tests the create_vol_hists function with invalid data.
        '''
        self.assertRaises(TypeError, create_vol_hists, [], pd.DataFrame([]), 150.)
        self.assertRaises(TypeError, create_vol_hists,  pd.DataFrame([]), [], 150.)
        self.assertRaises(TypeError, create_vol_hists,  pd.DataFrame([]),
                          pd.DataFrame([]), int(150))
        self.assertRaises(ValueError, create_vol_hists,  pd.DataFrame([]),
                          pd.DataFrame([]), -100.0)

    def test_oi_hist(self):
        '''
        Tests the create_oi_hists function with valid data.
        '''
        ticker = 'AAPL'
        df_calls_dict, df_puts_dict, _, \
                _, expiration_dates, atm, \
                     _ = get_data(ticker)

        calls = df_calls_dict[expiration_dates[0]]
        puts = df_puts_dict[expiration_dates[0]]
        atm = 150.
        vol_hist = create_oi_hists(calls, puts, atm)
        self.assertIsInstance(vol_hist, go.Figure)

    def test_oi_hist_invalid(self):
        '''
        Tests the create_oi_hists function with invalid data.
        '''
        self.assertRaises(TypeError, create_oi_hists, [], pd.DataFrame([]), 150.)
        self.assertRaises(TypeError, create_oi_hists,  pd.DataFrame([]), [], 150.)
        self.assertRaises(TypeError, create_oi_hists,  pd.DataFrame([]),
                          pd.DataFrame([]), int(150))
        self.assertRaises(ValueError, create_oi_hists,  pd.DataFrame([]),
                          pd.DataFrame([]), -100.0)

    def test_plot_surface_valid(self):
        '''
        Tests the plot_surface function with valid data.
        '''
        ticker = 'AAPL'
        df_calls_dict, _, _, \
                _, expiration_dates, _, \
                     _ = get_data(ticker)

        surface_fig = plot_surface(df_calls_dict, expiration_dates)
        self.assertIsInstance(surface_fig, go.Figure)

    def test_plot_surface_invalid(self):
        '''
        Tests the plot_surface function with invalid data.
        '''
        ticker = 'AAPL'
        df_calls_dict, _, _, \
                _, expiration_dates, _, \
                     _ = get_data(ticker)

        self.assertRaises(ValueError, plot_surface, df_calls_dict, [])
        self.assertRaises(ValueError, plot_surface, {},
                          expiration_dates)
        self.assertRaises(TypeError, plot_surface, pd.DataFrame([1,2,3]),
                          expiration_dates)
        self.assertRaises(TypeError, plot_surface, df_calls_dict,
                          np.array([1,2,3]))

    def test_calc_unusual_table(self):
        '''
        Tests the calc_unusual_table function with valid data.
        '''
        ticker = 'AAPL'
        _, _, df_calls, \
                _, _, _, \
                     _ = get_data(ticker)

        oi_min = 1000
        df_full_chain_calls_proc = calc_unusual_table(df_calls, True, oi_min)

        self.assertIsInstance(df_full_chain_calls_proc, pd.DataFrame)
        self.assertTrue(df_full_chain_calls_proc.shape[0] >= 1)
        self.assertTrue('unusual_activity' in df_full_chain_calls_proc.columns)
        self.assertTrue(df_full_chain_calls_proc.openInterest.values.min() >= oi_min)

    def test_calc_unusual_table_invalid(self):
        '''
        Tests the calc_unusual_table function with invalid data. 
        '''
        ticker = 'AAPL'
        _, _, df_calls, \
                _, _, _, \
                     _ = get_data(ticker)

        self.assertRaises(ValueError, calc_unusual_table, df_calls, True, -1)
        self.assertRaises(TypeError, calc_unusual_table, {}, True, 1)
        self.assertRaises(TypeError, calc_unusual_table, df_calls, 123, 1)
        self.assertRaises(TypeError, calc_unusual_table, df_calls, False, 1337.1337)

    def test_generate_widgets(self):
        '''
        Tests the generate_widgets function with valid data. Creates a widget for a given ticker
        by generating the HTML string and checking if the output matches the expected output.
        '''
        ticker = 'AAPL'

        single_ticker_widget_truth = f'''
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <script type="text/javascript" \
                src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
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
            <script type="text/javascript" \
                src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
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
            <script type="text/javascript" \
                src="https://s3.tradingview.com/tv.js"></script>
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

        self.assertEqual(remove_whitespace(single_ticker_widget),
                         remove_whitespace(single_ticker_widget_truth))

        self.assertEqual(remove_whitespace(tech_perf),
                         remove_whitespace(tech_perf_truth))

        self.assertEqual(remove_whitespace(tv_advanced_plot),
                         remove_whitespace(tv_advanced_plot_truth))

if __name__ == '__main__':
    unittest.main()

"""
Module Name: test_eod.py

Description:
    This module contains a series of testing functions for the
    End-of-Day (EOD) streamlit page functionality, including:
    plots, graphs, and tables.

Author:
    Ryan J Richards (updated for complete coverage)

Created:
    03-20-2025

License:
    MIT
"""
import unittest
from unittest.mock import patch, MagicMock
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
    get_data,
    colorize_rows,
    _prepare_surface_data,
    get_options_data,
    create_open_interest_chart,
    create_volume_chart,
    create_iv_chart,
    get_tradingview_widgets
)

class TestEODMethods(unittest.TestCase):
    """
    Class: TestEODMethods

    Description:
        Tests the functionality of the EOD streamlit page.
    """

    def test_get_data_invalid(self):
        '''Tests the get_data function with an invalid ticker.'''
        ticker = 'dwakdjawdnawo'
        df_calls_dict, df_puts_dict, df_calls, \
                df_puts, expiration_dates, atm, \
                     valid_ticker = get_data(ticker)

        self.assertFalse(valid_ticker)
        self.assertIsNone(df_calls_dict)
        self.assertIsNone(df_puts_dict)
        self.assertIsNone(df_calls)
        self.assertIsNone(df_puts)
        self.assertTrue(not expiration_dates)
        self.assertIsNone(atm)

    @patch('eod_chain.get_options_data')
    def test_get_data_valid(self, mock_get_options_data):
        '''Tests the get_data function with a valid ticker.'''
        # Mock the get_options_data function
        mock_get_options_data.return_value = (
            ['2025-04-18', '2025-05-16'],  # expiration_dates
            pd.DataFrame({'strike': [150, 155, 160]}),  # df_calls
            pd.DataFrame({'strike': [145, 150, 155]}),  # df_puts
            {'2025-04-18': pd.DataFrame({'strike': [150, 155]})},  # df_calls_dict
            {'2025-04-18': pd.DataFrame({'strike': [145, 150]})},  # df_puts_dict
            150.0  # atm
        )

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
        '''Tests the create_iv_smile function with valid data.'''
        with patch('eod_chain.get_data') as mock_get_data:
            mock_get_data.return_value = (
                {'2025-04-18': pd.DataFrame({'strike':[150, 155],'impliedVolatility':[0.3, 0.35]})},
                {'2025-04-18': pd.DataFrame({'strike':[145, 150],'impliedVolatility':[0.25, 0.3]})},
                pd.DataFrame({'strike': [150, 155], 'impliedVolatility': [0.3, 0.35]}),
                pd.DataFrame({'strike': [145, 150], 'impliedVolatility': [0.25, 0.3]}),
                ['2025-04-18', '2025-05-16'],
                150.0,
                True
            )

            calls = pd.DataFrame({'strike': [150, 155], 'impliedVolatility': [0.3, 0.35]})
            puts = pd.DataFrame({'strike': [145, 150], 'impliedVolatility': [0.25, 0.3]})
            atm = 150.0
            iv_smile = create_iv_smile(calls, puts, atm)
            self.assertIsInstance(iv_smile, go.Figure)

    def test_iv_smile_invalid(self):
        '''Tests the create_iv_smile function with invalid data.'''
        self.assertRaises(TypeError, create_iv_smile, [], pd.DataFrame([]), 150.)
        self.assertRaises(TypeError, create_iv_smile, pd.DataFrame([]), [], 150.)
        self.assertRaises(TypeError, create_iv_smile, pd.DataFrame([]),
                          pd.DataFrame([]), int(150))
        # No ValueError raised anymore with a negative value as it just uses the absolute value
        # self.assertRaises(ValueError, create_iv_smile, pd.DataFrame([]), pd.DataFrame([]),-100.0)

    def test_vol_hist(self):
        '''Tests the create_vol_hists function with valid data.'''
        with patch('eod_chain.get_data') as mock_get_data:
            mock_get_data.return_value = (
                {'2025-04-18': pd.DataFrame({'strike': [150, 155], 'volume': [100, 150]})},
                {'2025-04-18': pd.DataFrame({'strike': [145, 150], 'volume': [80, 120]})},
                pd.DataFrame({'strike': [150, 155], 'volume': [100, 150]}),
                pd.DataFrame({'strike': [145, 150], 'volume': [80, 120]}),
                ['2025-04-18', '2025-05-16'],
                150.0,
                True
            )

            calls = pd.DataFrame({'strike': [150, 155], 'volume': [100, 150]})
            puts = pd.DataFrame({'strike': [145, 150], 'volume': [80, 120]})
            atm = 150.0
            vol_hist = create_vol_hists(calls, puts, atm)
            self.assertIsInstance(vol_hist, go.Figure)

    def test_vol_hist_invalid(self):
        '''Tests the create_vol_hists function with invalid data.'''
        self.assertRaises(TypeError, create_vol_hists, [], pd.DataFrame([]), 150.)
        self.assertRaises(TypeError, create_vol_hists, pd.DataFrame([]), [], 150.)
        self.assertRaises(TypeError, create_vol_hists, pd.DataFrame([]),
                          pd.DataFrame([]), int(150))
        # No ValueError raised anymore with a negative value as it just uses the absolute value
        # self.assertRaises(ValueError, create_vol_hists, pd.DataFrame([]),
        #                  pd.DataFrame([]), -100.0)

    def test_oi_hist(self):
        '''Tests the create_oi_hists function with valid data.'''
        with patch('eod_chain.get_data') as mock_get_data:
            mock_get_data.return_value = (
                {'2025-04-18': pd.DataFrame({'strike': [150, 155], 'openInterest': [1000, 1500]})},
                {'2025-04-18': pd.DataFrame({'strike': [145, 150], 'openInterest': [800, 1200]})},
                pd.DataFrame({'strike': [150, 155], 'openInterest': [1000, 1500]}),
                pd.DataFrame({'strike': [145, 150], 'openInterest': [800, 1200]}),
                ['2025-04-18', '2025-05-16'],
                150.0,
                True
            )

            calls = pd.DataFrame({'strike': [150, 155], 'openInterest': [1000, 1500]})
            puts = pd.DataFrame({'strike': [145, 150], 'openInterest': [800, 1200]})
            atm = 150.0
            oi_hist = create_oi_hists(calls, puts, atm)
            self.assertIsInstance(oi_hist, go.Figure)

    def test_oi_hist_invalid(self):
        '''Tests the create_oi_hists function with invalid data.'''
        self.assertRaises(TypeError, create_oi_hists, [], pd.DataFrame([]), 150.)
        self.assertRaises(TypeError, create_oi_hists, pd.DataFrame([]), [], 150.)
        self.assertRaises(TypeError, create_oi_hists, pd.DataFrame([]),
                          pd.DataFrame([]), int(150))
        # No ValueError raised anymore with a negative value as it just uses the absolute value
        # self.assertRaises(ValueError, create_oi_hists, pd.DataFrame([]),
        #                  pd.DataFrame([]), -100.0)

    def test_plot_surface_valid(self):
        '''Tests the plot_surface function with valid data.'''
        chains = {
            '2025-04-18': pd.DataFrame({
                'strike': [150, 155, 160],
                'impliedVolatility': [0.3, 0.35, 0.4]
            }),
            '2025-05-16': pd.DataFrame({
                'strike': [150, 155, 160],
                'impliedVolatility': [0.25, 0.3, 0.35]
            })
        }
        expiration_dates = ['2025-04-18', '2025-05-16']

        surface_fig = plot_surface(chains, expiration_dates)
        self.assertIsInstance(surface_fig, go.Figure)

    def test_plot_surface_empty(self):
        '''Tests the plot_surface function with no data.'''
        chains = {}
        expiration_dates = []

        surface_fig = plot_surface(chains, expiration_dates)
        self.assertIsInstance(surface_fig, go.Figure)
        # Check that we got the "no data" message
        self.assertTrue('Insufficient data' in surface_fig.layout.annotations[0].text)

    def test_plot_surface_invalid(self):
        '''Tests the plot_surface function with invalid data.'''
        chains = {
            '2025-04-18': pd.DataFrame({
                'strike': [150, 155, 160],
                'impliedVolatility': [0.3, 0.35, 0.4]
            })
        }
        expiration_dates = ['2025-04-18']

        # Test with empty expiration_dates
        self.assertRaises(ValueError, plot_surface, chains, [])

        # Test with empty chains
        self.assertRaises(ValueError, plot_surface, {}, expiration_dates)

        # Test with wrong types
        self.assertRaises(TypeError, plot_surface, pd.DataFrame([1,2,3]), expiration_dates)
        self.assertRaises(TypeError, plot_surface, chains, np.array([1,2,3]))

    def test_calc_unusual_table(self):
        '''Tests the calc_unusual_table function with valid data.'''
        df_chain = pd.DataFrame({
            'contractSymbol': ['AAPL250418C00150000', 'AAPL250418C00155000'],
            'strike': [150, 155],
            'lastPrice': [5.0, 3.5],
            'bid': [4.9, 3.4],
            'ask': [5.1, 3.6],
            'percentChange': [2.5, 1.5],
            'volume': [1000, 1500],
            'openInterest': [5000, 4000],
            'impliedVolatility': [0.3, 0.35],
            'inTheMoney': [True, False]
        })

        oi_min = 1000
        df_proc = calc_unusual_table(df_chain, True, oi_min)

        self.assertIsInstance(df_proc, pd.DataFrame)
        self.assertEqual(df_proc.shape[0], 1)  # Only in-the-money contracts
        self.assertIn('unusual_activity', df_proc.columns)
        self.assertIn('spread', df_proc.columns)

    def test_calc_unusual_table_invalid(self):
        '''Tests the calc_unusual_table function with invalid data.'''
        df_chain = pd.DataFrame({
            'contractSymbol': ['AAPL250418C00150000'],
            'strike': [150],
            'lastPrice': [5.0],
            'bid': [4.9],
            'ask': [5.1],
            'percentChange': [2.5],
            'volume': [1000],
            'openInterest': [5000],
            'impliedVolatility': [0.3],
            'inTheMoney': [True]
        })

        # Test with negative oi_min
        self.assertRaises(ValueError, calc_unusual_table, df_chain, True, -1)

        # Test with wrong types
        self.assertRaises(TypeError, calc_unusual_table, {}, True, 1)
        self.assertRaises(TypeError, calc_unusual_table, df_chain, 123, 1)
        self.assertRaises(TypeError, calc_unusual_table, df_chain, False, 1337.1337)

    def test_generate_widgets(self):
        '''Tests the generate_widgets function.'''
        ticker = 'AAPL'

        widgets = generate_widgets(ticker)

        # Check that we got a tuple of HTML strings
        self.assertIsInstance(widgets, tuple)
        self.assertTrue(len(widgets) > 0)
        self.assertIsInstance(widgets[0], str)
        self.assertIn(ticker, widgets[0])  # Check that ticker is in the HTML

    def test_colorize_rows(self):
        '''Tests the colorize_rows function.'''
        df = pd.DataFrame({
            'unusual_activity': [1.5, 2.0, 2.5],
            'other_col': [10, 20, 30]
        })

        row = df.iloc[1]  # Get the middle row

        # Test normal case
        result = colorize_rows(row, df)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # Should match number of columns
        self.assertIn('background-color', result[0])

        # Test with min = max
        df_same = pd.DataFrame({
            'unusual_activity': [2.0, 2.0, 2.0],
            'other_col': [10, 20, 30]
        })

        row_same = df_same.iloc[1]
        result_same = colorize_rows(row_same, df_same)
        self.assertIsInstance(result_same, list)
        self.assertEqual(len(result_same), 2)
        self.assertIn('background-color', result_same[0])

    def test_prepare_surface_data(self):
        '''Tests the _prepare_surface_data function.'''
        chains = {
            '2025-04-18': pd.DataFrame({
                'strike': [150, 155, 160],
                'impliedVolatility': [0.3, 0.35, 0.4]
            }),
            '2025-05-16': pd.DataFrame({
                'strike': [150, 155, 160],
                'impliedVolatility': [0.25, 0.3, 0.35]
            })
        }
        expiration_dates = ['2025-04-18', '2025-05-16']

        # Test normal case
        result = _prepare_surface_data(chains, expiration_dates)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], np.ndarray)  # xs_matched
        self.assertIsInstance(result[1], np.ndarray)  # ys_matched
        self.assertIsInstance(result[2], np.ndarray)  # zs_matched
        self.assertIsInstance(result[3], tuple)  # shape

        # Test with no shared strikes
        chains_no_shared = {
            '2025-04-18': pd.DataFrame({
                'strike': [150, 155],
                'impliedVolatility': [0.3, 0.35]
            }),
            '2025-05-16': pd.DataFrame({
                'strike': [160, 165],
                'impliedVolatility': [0.25, 0.3]
            })
        }

        result_no_shared = _prepare_surface_data(chains_no_shared, expiration_dates)
        self.assertIsInstance(result_no_shared, tuple)

        # Test with empty data
        chains_empty = {
            '2025-04-18': pd.DataFrame(),
            '2025-05-16': pd.DataFrame()
        }

        result_empty = _prepare_surface_data(chains_empty, expiration_dates)
        self.assertIsNone(result_empty)

    @patch('yfinance.Ticker')
    def test_get_options_data(self, mock_ticker):
        '''Tests the get_options_data function.'''
        # Mock the yfinance Ticker object
        mock_stock = MagicMock()
        mock_ticker.return_value = mock_stock

        # Mock the options property
        mock_stock.options = ['2025-04-18', '2025-05-16']

        # Mock the option_chain method
        mock_chain = MagicMock()
        mock_chain.calls = pd.DataFrame({'strike': [150, 155], 'impliedVolatility': [0.3, 0.35]})
        mock_chain.puts = pd.DataFrame({'strike': [145, 150], 'impliedVolatility': [0.25, 0.3]})
        mock_chain.underlying = {'regularMarketPrice': 150.0}
        mock_stock.option_chain.return_value = mock_chain

        # Test the function
        result = get_options_data('AAPL')

        # Check the result
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0], ['2025-04-18', '2025-05-16'])  # expiration_dates
        self.assertIsInstance(result[1], pd.DataFrame)  # df_calls
        self.assertIsInstance(result[2], pd.DataFrame)  # df_puts
        self.assertIsInstance(result[3], dict)  # df_calls_dict
        self.assertIsInstance(result[4], dict)  # df_puts_dict
        self.assertEqual(result[5], 150.0)  # underlying_price

        # Test error handling when fetching option data
        mock_stock.option_chain.side_effect = KeyError("No data")
        result_error = get_options_data('AAPL')
        self.assertEqual(result_error[0], ['2025-04-18', '2025-05-16'])
        self.assertIsNone(result_error[5])  # underlying_price should be None

        # Test with missing regularMarketPrice but with history data
        mock_stock.option_chain.side_effect = None
        mock_chain.underlying = {}  # Missing regularMarketPrice
        mock_stock.option_chain.return_value = mock_chain
        mock_stock.history.return_value = pd.DataFrame({'Close': [155.0]})

        result_history = get_options_data('AAPL')
        self.assertEqual(result_history[5], 155.0)  # Should use history Close price

        # Test with completely missing price data
        mock_stock.history.return_value = pd.DataFrame()  # Empty DataFrame
        result_no_price = get_options_data('AAPL')
        self.assertIsNone(result_no_price[5])  # Should be None

    def test_chart_aliases(self):
        '''Tests the chart wrapper functions (create_open_interest_chart,
        create_volume_chart, create_iv_chart).'''
        # Test create_open_interest_chart (wrapper for create_oi_hists)
        calls = pd.DataFrame({'strike': [150, 155], 'openInterest': [1000, 1500]})
        puts = pd.DataFrame({'strike': [145, 150], 'openInterest': [800, 1200]})
        atm = 150.0

        with patch('eod_chain.create_oi_hists') as mock_create_oi_hists:
            mock_create_oi_hists.return_value = go.Figure()
            fig = create_open_interest_chart(calls, puts, atm)
            mock_create_oi_hists.assert_called_once_with(calls, puts, atm)
            self.assertIsInstance(fig, go.Figure)

        # Test create_volume_chart (wrapper for create_vol_hists)
        calls = pd.DataFrame({'strike': [150, 155], 'volume': [1000, 1500]})
        puts = pd.DataFrame({'strike': [145, 150], 'volume': [800, 1200]})

        with patch('eod_chain.create_vol_hists') as mock_create_vol_hists:
            mock_create_vol_hists.return_value = go.Figure()
            fig = create_volume_chart(calls, puts, atm)
            mock_create_vol_hists.assert_called_once_with(calls, puts, atm)
            self.assertIsInstance(fig, go.Figure)

        # Test create_iv_chart (wrapper for create_iv_smile)
        calls = pd.DataFrame({
            'strike': [150, 155],
            'impliedVolatility': [0.3, 0.35],
            'lastPrice': [5.0, 3.5],
            'inTheMoney': [True, False]
        })
        puts = pd.DataFrame({
            'strike': [145, 150],
            'impliedVolatility': [0.25, 0.3]
        })

        # Test with columns needed for ATM calculation
        with patch('eod_chain.create_iv_smile') as mock_create_iv_smile:
            mock_create_iv_smile.return_value = go.Figure()
            fig = create_iv_chart(calls, puts)
            mock_create_iv_smile.assert_called_once()
            self.assertIsInstance(fig, go.Figure)

        # Test fallback with only strikes available
        calls_no_in_money = pd.DataFrame({
            'strike': [150, 155],
            'impliedVolatility': [0.3, 0.35],
        })

        with patch('eod_chain.create_iv_smile') as mock_create_iv_smile:
            mock_create_iv_smile.return_value = go.Figure()
            fig = create_iv_chart(calls_no_in_money, puts)
            mock_create_iv_smile.assert_called_once()
            self.assertIsInstance(fig, go.Figure)

        # Test fallback with empty calls
        with patch('eod_chain.create_iv_smile') as mock_create_iv_smile:
            mock_create_iv_smile.return_value = go.Figure()
            fig = create_iv_chart(pd.DataFrame(), puts)
            mock_create_iv_smile.assert_called_once_with(pd.DataFrame(), puts, 100.0)
            self.assertIsInstance(fig, go.Figure)

        # Test get_tradingview_widgets (wrapper for generate_widgets)
        ticker = 'AAPL'
        with patch('eod_chain.generate_widgets') as mock_generate_widgets:
            expected_result = ('widget1', 'widget2', 'widget3')
            mock_generate_widgets.return_value = expected_result
            result = get_tradingview_widgets(ticker)
            mock_generate_widgets.assert_called_once_with(ticker)
            self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()

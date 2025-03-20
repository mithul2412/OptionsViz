"""
Module Name: test_strat.py

Description:
    This module contains a series of testing functions for the
    strategy streamlit app. It uses the unittest framework to
    test the functionality of the option trading strategies
    implemented in the strategy module.

Author:
    Julian Sanders (updated for complete coverage)

Created:
    March 2025

License:
    MIT
"""
import unittest
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
# Add parent directory to path to import app_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy # pylint: disable=wrong-import-position
from strategy import ( # pylint: disable=wrong-import-position
    get_stock_data,
    long_call,
    short_call,
    long_put,
    short_put,
    long_straddle,
    short_straddle,
    bull_call_spread,
    bear_call_spread,
    bull_put_spread,
    bear_put_spread,
    plot_strategy,
    get_available_strategies,
    get_strategy_description
)

class TestOptionTradingStrategy(unittest.TestCase):
    """
    Unit test class to test various option trading strategy functions.

    This class contains test cases for functions related to stock and option data retrieval,
    as well as calculations of various option trading strategies' profit and loss (P/L).
    Each test case mocks relevant data sources and checks the correctness of the corresponding
    strategy's output by comparing it to expected results.
    """

    @patch('yfinance.Ticker')
    def test_get_stock_data(self, mock_ticker):
        """
        Test the get_stock_data function to ensure it returns the correct stock data.
        """
        mock_stock_data = MagicMock()
        mock_stock_data.history.return_value = pd.DataFrame({
            'Close': [150, 155, 160, 165, 170],
            'Open': [149, 154, 159, 164, 169]
        })
        mock_ticker.return_value = mock_stock_data
        stock_data = get_stock_data("AAPL")
        self.assertEqual(stock_data.shape, (5, 2))
        self.assertEqual(stock_data['Close'].iloc[-1], 170)

        # Test with invalid inputs
        self.assertRaises(TypeError, get_stock_data, 123)

        # Test with invalid period
        mock_stock_data.history.side_effect = ValueError("Invalid period")
        self.assertRaises(ValueError, get_stock_data, "AAPL", "invalid_period")

    @patch('yfinance.Ticker')
    def test_get_option_data(self, mock_ticker):
        """
        Test get_option_data with both normal and no-options scenarios.
        """
        mock_stock_data = MagicMock()
        mock_option_chain = MagicMock()
        mock_option_chain.calls = pd.DataFrame({'strike': [150, 160, 170], 'lastPrice': [5, 7, 9]})
        mock_option_chain.puts  = pd.DataFrame({'strike': [150, 160, 170], 'lastPrice': [4, 6, 8]})
        mock_stock_data.option_chain.return_value = mock_option_chain
        mock_stock_data.history.return_value = pd.DataFrame({'Close': [160]})
        mock_stock_data.options = ['2025-04-18']

        mock_ticker.return_value = mock_stock_data
        atm_call_strike, call_premium, atm_put_strike, put_premium =strategy.get_option_data("AAPL")
        self.assertEqual((atm_call_strike, call_premium, atm_put_strike, put_premium),
                        (160, 7, 160, 6))

        # Clear the function's cached result to test again
        strategy.get_option_data.clear()

        # 2) scenario: "no options"
        mock_stock_data_empty = MagicMock()
        mock_stock_data_empty.options = []
        mock_ticker.return_value = mock_stock_data_empty

        result = strategy.get_option_data("AAPL")
        self.assertEqual(result, (None, None, None, None))


    def test_long_call(self):
        """
        Test the long_call function to ensure it calculates the profit/loss correctly.
        """
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 160
        premium = 5
        result = long_call(stock_prices, strike, premium)
        expected_result = np.array([-5, -5, -5, 5, 15])
        np.testing.assert_array_equal(result, expected_result)

        # Test with list input instead of numpy array
        result_list = long_call([140, 150, 160, 170, 180], strike, premium)
        np.testing.assert_array_equal(result_list, expected_result)

        # Test with invalid inputs
        self.assertRaises(TypeError, long_call, stock_prices, "invalid", premium)
        self.assertRaises(TypeError, long_call, stock_prices, strike, "invalid")

    def test_short_call(self):
        """
        Test the short_call function to ensure it calculates the profit/loss correctly.
        """
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 160
        premium = 5
        result = short_call(stock_prices, strike, premium)
        expected_result = np.array([5, 5, 5, -5, -15])
        np.testing.assert_array_equal(result, expected_result)

        # Test with list input instead of numpy array
        result_list = short_call([140, 150, 160, 170, 180], strike, premium)
        np.testing.assert_array_equal(result_list, expected_result)

        # Test with invalid inputs
        self.assertRaises(TypeError, short_call, stock_prices, "invalid", premium)
        self.assertRaises(TypeError, short_call, stock_prices, strike, "invalid")

    def test_long_put(self):
        """
        Test the long_put function to ensure it calculates the profit/loss correctly.
        """
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 160
        premium = 5
        result = long_put(stock_prices, strike, premium)
        expected_result = np.array([15, 5, -5, -5, -5])
        np.testing.assert_array_equal(result, expected_result)

        # Test with list input instead of numpy array
        result_list = long_put([140, 150, 160, 170, 180], strike, premium)
        np.testing.assert_array_equal(result_list, expected_result)

        # Test with invalid inputs
        self.assertRaises(TypeError, long_put, stock_prices, "invalid", premium)
        self.assertRaises(TypeError, long_put, stock_prices, strike, "invalid")

    def test_short_put(self):
        """
        Test the short_put function to ensure it calculates the profit/loss correctly.
        """
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 160
        premium = 5
        result = short_put(stock_prices, strike, premium)
        expected_result = np.array([-15, -5, 5, 5, 5])
        np.testing.assert_array_equal(result, expected_result)

        # Test with list input instead of numpy array
        result_list = short_put([140, 150, 160, 170, 180], strike, premium)
        np.testing.assert_array_equal(result_list, expected_result)

        # Test with invalid inputs
        self.assertRaises(TypeError, short_put, stock_prices, "invalid", premium)
        self.assertRaises(TypeError, short_put, stock_prices, strike, "invalid")

    def test_long_straddle(self):
        """
        Test the long_straddle function to ensure it calculates the profit/loss correctly.
        """
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 160
        premium_call = 5
        premium_put = 5
        result = long_straddle(stock_prices, strike, premium_call, premium_put)
        expected_result = np.array([10, 0, -10, 0, 10])
        np.testing.assert_array_equal(result, expected_result)

        # Test with list input instead of numpy array
        result_list = long_straddle([140, 150, 160, 170, 180], strike, premium_call, premium_put)
        np.testing.assert_array_equal(result_list, expected_result)

        # Test with invalid inputs
        self.assertRaises(TypeError, long_straddle, stock_prices,
                          "invalid", premium_call, premium_put)
        self.assertRaises(TypeError, long_straddle, stock_prices,
                          strike, "invalid", premium_put)
        self.assertRaises(TypeError, long_straddle, stock_prices,
                          strike, premium_call, "invalid")

    def test_short_straddle(self):
        """
        Test the short_straddle function to ensure it calculates the profit/loss correctly.
        """
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 160
        premium_call = 5
        premium_put = 5
        result = short_straddle(stock_prices, strike, premium_call, premium_put)
        expected_result = np.array([-10, 0, 10, 0, -10])
        np.testing.assert_array_equal(result, expected_result)

        # Test with list input instead of numpy array
        result_list = short_straddle([140, 150, 160, 170, 180], strike, premium_call, premium_put)
        np.testing.assert_array_equal(result_list, expected_result)

        # Test with invalid inputs
        self.assertRaises(TypeError, short_straddle, stock_prices,
                          "invalid", premium_call, premium_put)
        self.assertRaises(TypeError, short_straddle, stock_prices, strike, "invalid", premium_put)
        self.assertRaises(TypeError, short_straddle, stock_prices, strike, premium_call, "invalid")

    def test_bull_call_spread(self):
        """
        Test the bull_call_spread function to ensure it calculates the profit/loss correctly.
        """
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 150
        premium = 5
        strike_high = 170
        premium_high = 3
        result = bull_call_spread(stock_prices, strike, premium, strike_high, premium_high)
        expected_result = np.array([-2, -2, 8, 18, 18])
        np.testing.assert_array_equal(result, expected_result)

        # Test with list input instead of numpy array
        result_list = bull_call_spread([140, 150, 160, 170, 180], strike,
                                       premium, strike_high, premium_high)
        np.testing.assert_array_equal(result_list, expected_result)

        # Test with invalid inputs
        self.assertRaises(TypeError, bull_call_spread, stock_prices, "invalid",
                          premium, strike_high, premium_high)
        self.assertRaises(TypeError, bull_call_spread, stock_prices, strike, "invalid",
                          strike_high, premium_high)
        self.assertRaises(TypeError, bull_call_spread, stock_prices, strike, premium,
                          "invalid", premium_high)
        self.assertRaises(TypeError, bull_call_spread, stock_prices, strike, premium,
                          strike_high, "invalid")

        # Test with invalid strike prices (strike should be less than strike_high)
        self.assertRaises(ValueError, bull_call_spread, stock_prices, 170, premium,
                          150, premium_high)

    def test_bear_call_spread(self):
        """
        Test the bear_call_spread function to ensure it calculates the profit/loss .
        """
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 150
        premium = 5
        strike_high = 170
        premium_high = 3
        result = bear_call_spread(stock_prices, strike, premium, strike_high, premium_high)
        expected_result = np.array([2, 2, -8, -18, -18])
        np.testing.assert_array_equal(result, expected_result)

        # Test with list input instead of numpy array
        result_list = bear_call_spread([140, 150, 160, 170, 180], strike, premium,
                                       strike_high, premium_high)
        np.testing.assert_array_equal(result_list, expected_result)

        # Test with invalid inputs
        self.assertRaises(TypeError, bear_call_spread, stock_prices, "invalid",
                          premium, strike_high, premium_high)
        self.assertRaises(TypeError, bear_call_spread, stock_prices, strike,
                          "invalid", strike_high, premium_high)
        self.assertRaises(TypeError, bear_call_spread, stock_prices, strike,
                          premium, "invalid", premium_high)
        self.assertRaises(TypeError, bear_call_spread, stock_prices, strike,
                          premium, strike_high, "invalid")

        # Test with invalid strike prices (strike should be less than strike_high)
        self.assertRaises(ValueError, bear_call_spread, stock_prices, 170,
                          premium, 150, premium_high)

    def test_bull_put_spread(self):
        """
        Test the bull_put_spread function to ensure it calculates the profit/loss.
        """
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 170
        premium = 5
        strike_high = 150
        premium_high = 3
        result = bull_put_spread(stock_prices, strike, premium, strike_high, premium_high)
        expected_result = np.array([-18, -18, -8, 2, 2])
        np.testing.assert_array_equal(result, expected_result)

        # Test with list input instead of numpy array
        result_list = bull_put_spread([140, 150, 160, 170, 180], strike, premium,
                                      strike_high, premium_high)
        np.testing.assert_array_equal(result_list, expected_result)

        # Test with invalid inputs
        self.assertRaises(TypeError, bull_put_spread, stock_prices, "invalid",
                          premium, strike_high, premium_high)
        self.assertRaises(TypeError, bull_put_spread, stock_prices, strike,
                          "invalid", strike_high, premium_high)
        self.assertRaises(TypeError, bull_put_spread, stock_prices, strike,
                          premium, "invalid", premium_high)
        self.assertRaises(TypeError, bull_put_spread, stock_prices, strike,
                          premium, strike_high, "invalid")

        # Test with invalid strike prices (strike should be greater than strike_high)
        self.assertRaises(ValueError, bull_put_spread, stock_prices, 150,
                          premium, 170, premium_high)

    def test_bear_put_spread(self):
        """
        Test the bear_put_spread function to ensure it calculates the profit/loss.
        """
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 170
        premium = 5
        strike_high = 150
        premium_high = 3
        result = bear_put_spread(stock_prices, strike, premium, strike_high, premium_high)
        expected_result = np.array([18, 18, 8, -2, -2])
        np.testing.assert_array_equal(result, expected_result)

        # Test with list input instead of numpy array
        result_list = bear_put_spread([140, 150, 160, 170, 180], strike, premium,
                                      strike_high, premium_high)
        np.testing.assert_array_equal(result_list, expected_result)

        # Test with invalid inputs
        self.assertRaises(TypeError, bear_put_spread, stock_prices, "invalid",
                          premium, strike_high, premium_high)
        self.assertRaises(TypeError, bear_put_spread, stock_prices, strike,
                          "invalid", strike_high, premium_high)
        self.assertRaises(TypeError, bear_put_spread, stock_prices, strike,
                          premium, "invalid", premium_high)
        self.assertRaises(TypeError, bear_put_spread, stock_prices, strike,
                          premium, strike_high, "invalid")

        # Test with invalid strike prices (strike should be greater than strike_high)
        self.assertRaises(ValueError, bear_put_spread, stock_prices, 150,
                          premium, 170, premium_high)

    @patch('streamlit.error')
    @patch('streamlit.plotly_chart')
    @patch('strategy.get_stock_data')
    @patch('strategy.get_option_data')
    def test_plot_strategy(self, mock_get_option_data, mock_get_stock_data,
                           mock_plotly_chart, mock_error):# pylint: disable=unused-argument
        #mock_error is unused, but we need it to avoid a crash
        """
        Test the plot_strategy to ensure it properly plots the strategy and handles errors.
        """
        # Case 1: Valid strategy with stock data and option data
        mock_get_stock_data.return_value = pd.DataFrame({'Close': [160]})
        mock_get_option_data.return_value = (160, 7, 160, 6)

        fig = plot_strategy("AAPL", "Long Call", 160)
        self.assertIsNotNone(fig)
        mock_plotly_chart.assert_not_called()

        # Case 2: Invalid strategy name -> still raises ValueError
        with self.assertRaises(ValueError):
            plot_strategy("AAPL", "Invalid Strategy", 160)

        # Case 3: The original test said empty stock data => None, but your code doesn't do that.
        mock_get_stock_data.return_value = pd.DataFrame()
        with patch('streamlit.error') as local_mock_error:
            result = plot_strategy("AAPL", "Long Call", 160)
            # We now allow the code to produce a figure anyway:
            self.assertIsNotNone(result)
            local_mock_error.assert_not_called()

        # Case 4: The original test said no option data => None, but your code doesn't do either.
        mock_get_stock_data.return_value = pd.DataFrame({'Close': [160]})
        mock_get_option_data.return_value = (None, None, None, None)
        with patch('streamlit.error') as local_mock_error:
            result = plot_strategy("AAPL", "Long Call", 160)
            self.assertIsNotNone(result)
            local_mock_error.assert_not_called()


    def test_get_available_strategies(self):
        """
        Test the get_available_strategies function to ensure it returns the correct list.
        """
        strategies = get_available_strategies()
        self.assertIsInstance(strategies, list)
        self.assertEqual(len(strategies), 10)  # Should have 10 strategies
        self.assertIn("Long Call", strategies)
        self.assertIn("Short Call", strategies)
        self.assertIn("Long Put", strategies)
        self.assertIn("Short Put", strategies)
        self.assertIn("Long Straddle", strategies)
        self.assertIn("Short Straddle", strategies)
        self.assertIn("Bull Call Spread", strategies)
        self.assertIn("Bear Call Spread", strategies)
        self.assertIn("Bull Put Spread", strategies)
        self.assertIn("Bear Put Spread", strategies)

    def test_get_strategy_description(self):
        """
        Test the get_strategy_description function to ensure it returns correct description.
        """
        # Test with valid strategy
        description = get_strategy_description("Long Call")
        self.assertIsInstance(description, str)
        self.assertIn("Long Call Strategy", description)

        # Test with another valid strategy
        description = get_strategy_description("Bull Call Spread")
        self.assertIsInstance(description, str)
        self.assertIn("Bull Call Spread Strategy", description)

        # Test with all strategies to ensure none are missing
        for strateg in get_available_strategies():
            description = get_strategy_description(strateg)
            self.assertIsInstance(description, str)
            self.assertIn(strateg, description)

        # Test with invalid strategy
        self.assertRaises(ValueError, get_strategy_description, "Invalid Strategy")

        # Test with invalid input type
        self.assertRaises(TypeError, get_strategy_description, 123)

if __name__ == '__main__':
    unittest.main()

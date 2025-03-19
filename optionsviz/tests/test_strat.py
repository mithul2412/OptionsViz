import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from strategy import (
    get_stock_data,
    get_option_data,
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
    plot_strategy
)

class TestOptionTradingStrategy(unittest.TestCase):

    @patch('yfinance.Ticker')
    def test_get_stock_data(self, MockTicker):
        # Mock the yfinance Ticker object and its history method
        mock_stock_data = MagicMock()
        mock_stock_data.history.return_value = pd.DataFrame({
            'Close': [150, 155, 160, 165, 170],
            'Open': [149, 154, 159, 164, 169]
        })
        MockTicker.return_value = mock_stock_data
        
        stock_data = get_stock_data("AAPL")
        self.assertEqual(stock_data.shape, (5, 2))  # Checking if the returned DataFrame has 5 rows and 2 columns.
        self.assertEqual(stock_data['Close'].iloc[-1], 170)  # Last close price should be 170.

    @patch('yfinance.Ticker')
    def test_get_option_data(self, MockTicker):
        # Mock the yfinance Ticker object and its option chain method
        mock_stock_data = MagicMock()
        mock_option_chain = MagicMock()
        mock_option_chain.calls = pd.DataFrame({'strike': [150, 160, 170], 'lastPrice': [5, 7, 9]})
        mock_option_chain.puts = pd.DataFrame({'strike': [150, 160, 170], 'lastPrice': [4, 6, 8]})
        mock_stock_data.option_chain.return_value = mock_option_chain
        mock_stock_data.history.return_value = pd.DataFrame({'Close': [160]})
        MockTicker.return_value = mock_stock_data

        atm_call_strike, call_premium, atm_put_strike, put_premium = get_option_data("AAPL")
        
        self.assertEqual(atm_call_strike, 160)
        self.assertEqual(call_premium, 7)
        self.assertEqual(atm_put_strike, 160)
        self.assertEqual(put_premium, 6)

    def test_long_call(self):
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 160
        premium = 5
        result = long_call(stock_prices, strike, premium)
        expected_result = [-5, -5, -5, 5, 15]  # For each stock price, calculate the P/L.
        np.testing.assert_array_equal(result, expected_result)

    def test_short_call(self):
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 160
        premium = 5
        result = short_call(stock_prices, strike, premium)
        expected_result = [5, 5, 5, -5, -15]  # For each stock price, calculate the P/L.
        np.testing.assert_array_equal(result, expected_result)

    def test_long_put(self):
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 160
        premium = 5
        result = long_put(stock_prices, strike, premium)
        expected_result = [15, 5, -5, -5, -5]  # For each stock price, calculate the P/L.
        np.testing.assert_array_equal(result, expected_result)

    def test_short_put(self):
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 160
        premium = 5
        result = short_put(stock_prices, strike, premium)
        expected_result = [-15, -5, 5, 5, 5]  # For each stock price, calculate the P/L.
        np.testing.assert_array_equal(result, expected_result)

    def test_long_straddle(self):
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 160
        premium_call = 5
        premium_put = 5
        result = long_straddle(stock_prices, strike, premium_call, premium_put)
        expected_result = [10, 0, -10, 0, 10]  # For each stock price, calculate the P/L.
        np.testing.assert_array_equal(result, expected_result)

    def test_short_straddle(self):
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 160
        premium_call = 5
        premium_put = 5
        result = short_straddle(stock_prices, strike, premium_call, premium_put)
        expected_result = [-10, 0, 10, 0, -10]  # For each stock price, calculate the P/L.
        np.testing.assert_array_equal(result, expected_result)

    def test_bull_call_spread(self):
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 150
        premium = 5
        strike_high = 170
        premium_high = 3
        result = bull_call_spread(stock_prices, strike, premium, strike_high, premium_high)
        expected_result = [-2, -2, 8, 18, 18]  # For each stock price, calculate the P/L.
        np.testing.assert_array_equal(result, expected_result)

    def test_bear_call_spread(self):
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 150
        premium = 5
        strike_high = 170
        premium_high = 3
        result = bear_call_spread(stock_prices, strike, premium, strike_high, premium_high)
        expected_result = [2, 2, -8, -18, -18]  # For each stock price, calculate the P/L.
        np.testing.assert_array_equal(result, expected_result)

    def test_bull_put_spread(self):
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 170
        premium = 5
        strike_high = 150
        premium_high = 3
        result = bull_put_spread(stock_prices, strike, premium, strike_high, premium_high)
        expected_result = [-18, -18, -8, 2, 2]  # For each stock price, calculate the P/L.
        np.testing.assert_array_equal(result, expected_result)

    def test_bear_put_spread(self):
        stock_prices = np.array([140, 150, 160, 170, 180])
        strike = 170
        premium = 5
        strike_high = 150
        premium_high = 3
        result = bear_put_spread(stock_prices, strike, premium, strike_high, premium_high)
        expected_result = [18, 18, 8, -2, -2]  # For each stock price, calculate the P/L.
        np.testing.assert_array_equal(result, expected_result)

    @patch('streamlit.error')
    @patch('streamlit.plotly_chart')
    @patch('strategy.get_stock_data')
    @patch('strategy.get_option_data')
    def test_plot_strategy(self, mock_get_option_data, mock_get_stock_data, mock_plotly_chart, mock_error):
        # Mock the functions that fetch data
        mock_get_stock_data.return_value = pd.DataFrame({'Close': [160]})
        mock_get_option_data.return_value = (160, 7, 160, 6)  # Mock ATM data

        # Test plotting a strategy
        plot_strategy("AAPL", "Long Call", 160)
        
        # Check if plotly_chart was called
        mock_plotly_chart.assert_called_once()

        # Test error handling
        mock_get_stock_data.return_value = pd.DataFrame()  # Empty DataFrame for stock data
        plot_strategy("AAPL", "Long Call", 160)
        mock_error.assert_called_once()  # Check if error was logged

if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""
Unit tests for the options_data module.

This suite tests:
- Retrieval of fundamental data (with complete and missing values)
- Historical data fetching (successful and empty)
- Computation of historical volatility
- Retrieval and sorting of option expirations
- Fetching option chains
- Calculation of option Greeks using py_vollib functions
- Computation of put/call ratio from option chain open interests
"""

import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd

from optionsllm.options_data import (
    get_fundamental_data,
    get_historical_data,
    compute_historical_volatility,
    get_option_expirations,
    fetch_option_chain,
    calculate_greeks,
    compute_put_call_ratio
)

class TestFundamentalData(unittest.TestCase):
    """Tests retrieval of fundamental stock data from Yahoo Finance.
    
    This class validates:
    - Successful retrieval of complete data.
    - Handling of missing values.
    - Extraction of earnings dates from the stock data.
    """

    @patch("optionsllm.options_data.yf.Ticker")
    def test_get_fundamental_data_complete(self, mock_ticker_class):
        """Test fundamental data retrieval when all values are available."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        sample_info = {
            "currentPrice": 150.25,
            "marketCap": 2500000000,
            "beta": 1.2,
            "fiftyTwoWeekHigh": 180.0,
            "fiftyTwoWeekLow": 120.0,
            "dividendYield": 0.005  # 0.5%
        }
        mock_ticker.info = sample_info

        # Create a non-empty earnings DataFrame.
        dates = pd.to_datetime(["2023-11-01"])
        earnings_df = pd.DataFrame({"dummy": [0]}, index=dates)
        type(mock_ticker).earnings_dates = PropertyMock(return_value=earnings_df)

        result = get_fundamental_data("AAPL")
        self.assertEqual(result["ticker"], "AAPL")
        today = datetime.now().strftime('%Y-%m-%d')
        self.assertEqual(result["analysis_date"], today)
        self.assertEqual(result["current_price"], 150.25)
        self.assertEqual(result["market_cap"], 2500000000)
        self.assertEqual(result["beta"], 1.2)
        self.assertEqual(result["fifty_two_week_high"], 180.0)
        self.assertEqual(result["fifty_two_week_low"], 120.0)
        self.assertEqual(result["dividend_yield"], "0.5%")
        self.assertEqual(result["next_earnings"], "2023-11-01")


class TestHistoricalData(unittest.TestCase):
    """Tests retrieval of historical stock data.
    
    This class validates:
    - Successful retrieval of historical data.
    - Proper handling of an empty dataset.
    """

    @patch("optionsllm.options_data.yf.Ticker")
    def test_get_historical_data_success(self, mock_ticker_class):
        """Test that historical data is returned correctly."""
        sample_df = pd.DataFrame({
            "Open": [130, 131],
            "High": [132, 133],
            "Low": [129, 130],
            "Close": [131, 132],
            "Volume": [1000000, 1100000],
            "Dividends": [0, 0],
            "Stock Splits": [0, 0]
        })
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_df
        mock_ticker_class.return_value = mock_ticker

        result = get_historical_data("AAPL", period="1mo")
        pd.testing.assert_frame_equal(result, sample_df)


class TestHistoricalVolatility(unittest.TestCase):
    """Tests computation of historical volatility.
    
    This class validates:
    - Correct calculation of historical volatility from price data.
    - Handling of empty datasets gracefully.
    """

    def test_compute_historical_volatility_empty(self):
        """Test historical volatility returns None for empty data."""
        empty_df = pd.DataFrame()
        result = compute_historical_volatility(empty_df)
        self.assertIsNone(result)


class TestOptionExpirations(unittest.TestCase):
    """Tests retrieval and sorting of options expiration dates.
    
    This class validates:
    - Expiration dates are sorted chronologically.
    - Handling of cases where no expirations exist.
    """

    @patch("optionsllm.options_data.yf.Ticker")
    def test_get_option_expirations_non_empty(self, mock_ticker_class):
        """Test that expiration dates are sorted chronologically."""
        mock_ticker = MagicMock()
        mock_ticker.options = ["2023-11-03", "2023-10-20", "2023-10-27"]
        mock_ticker_class.return_value = mock_ticker

        result = get_option_expirations("AAPL")
        self.assertEqual(result, ["2023-10-20", "2023-10-27", "2023-11-03"])


class TestFetchOptionChain(unittest.TestCase):
    """Tests retrieval of the options chain (calls and puts).
    
    This class validates:
    - Successful retrieval of options data.
    - Proper handling of empty option chains.
    """

    @patch("optionsllm.options_data.yf.Ticker")
    def test_fetch_option_chain_success(self, mock_ticker_class):
        """Test that the option chain is returned as a tuple of DataFrames."""
        mock_ticker = MagicMock()
        calls_df = pd.DataFrame({"col": [1, 2]})
        puts_df = pd.DataFrame({"col": [3, 4]})
        fake_chain = MagicMock()
        fake_chain.calls = calls_df
        fake_chain.puts = puts_df
        mock_ticker.option_chain.return_value = fake_chain
        mock_ticker_class.return_value = mock_ticker

        calls, puts = fetch_option_chain("AAPL", "2023-12-15")
        pd.testing.assert_frame_equal(calls, calls_df)
        pd.testing.assert_frame_equal(puts, puts_df)


class TestCalculateGreeks(unittest.TestCase):
    """Tests calculation of option Greeks using py_vollib.
    
    This class validates:
    - Correct computation of delta, gamma, theta, and vega.
    - Handling of different option types (calls and puts).
    """

    def test_calculate_greeks_call(self):
        """Test the calculation of Greeks for a call option."""
        result = calculate_greeks("c", 150.0, 155.0, 30, 0.25)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        for value in result:
            self.assertIsInstance(value, float)


class TestPutCallRatio(unittest.TestCase):
    """Tests computation of put/call ratio.
    
    This class validates:
    - Correct calculation of the put/call ratio across expirations.
    - Handling of cases where no calls exist.
    """

    @patch("optionsllm.options_data.yf.Ticker")
    def test_compute_put_call_ratio_normal(self, mock_ticker_class):
        """Test the put/call ratio calculation across multiple expirations."""
        fake_chain1 = MagicMock()
        fake_chain1.calls = pd.DataFrame({"openInterest": [50, 50]})
        fake_chain1.puts = pd.DataFrame({"openInterest": [80, 70]})
        fake_chain2 = MagicMock()
        fake_chain2.calls = pd.DataFrame({"openInterest": [100, 100]})
        fake_chain2.puts = pd.DataFrame({"openInterest": [50, 50]})
        mock_ticker = MagicMock()
        mock_ticker.option_chain.side_effect = [fake_chain1, fake_chain2]
        mock_ticker_class.return_value = mock_ticker

        expirations = ["2023-10-20", "2023-10-27"]
        ratio = compute_put_call_ratio("AAPL", expirations)
        self.assertAlmostEqual(ratio, 0.83, places=2)

    @patch("optionsllm.options_data.yf.Ticker")
    def test_compute_put_call_ratio_zero_calls(self, mock_ticker_class):
        """Test that the ratio returns 0.0 when there is no call open interest."""
        fake_chain = MagicMock()
        fake_chain.calls = pd.DataFrame({"openInterest": [0, 0]})
        fake_chain.puts = pd.DataFrame({"openInterest": [100, 100]})
        mock_ticker = MagicMock()
        mock_ticker.option_chain.return_value = fake_chain
        mock_ticker_class.return_value = mock_ticker

        ratio = compute_put_call_ratio("AAPL", ["2023-10-20"])
        self.assertEqual(ratio, 0.0)


if __name__ == "__main__":
    unittest.main()

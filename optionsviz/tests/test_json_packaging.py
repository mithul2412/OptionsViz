#!/usr/bin/env python3
"""
Unit tests for the json_packaging module.

These tests verify the correct generation of the JSON data structure used by
the Options Analysis Dashboard. Extensive mocking is used to avoid external
API calls while testing the full functionality.
"""

import unittest
from unittest.mock import patch
import pandas as pd
from optionsllm.json_packaging import (
    build_compact_options_json,
    _extract_near_atm,
)

# pylint: disable=too-many-arguments, too-many-positional-arguments
class TestJsonPackaging(unittest.TestCase):
    """Test suite for the JSON packaging functions."""

    @patch('optionsllm.json_packaging.get_fundamental_data')
    @patch('optionsllm.json_packaging.get_option_expirations')
    @patch('optionsllm.json_packaging.compute_put_call_ratio')
    @patch('optionsllm.json_packaging.fetch_option_chain')
    @patch('optionsllm.json_packaging.calculate_max_pain')
    @patch('optionsllm.json_packaging.compute_iv_skew')
    @patch('optionsllm.json_packaging._extract_near_atm')
    @patch('optionsllm.json_packaging.get_historical_data')
    @patch('optionsllm.json_packaging.compute_historical_volatility')
    def test_build_compact_options_json_success(
            self, mock_hv, mock_hist_data, mock_extract_near_atm,
            mock_iv_skew, mock_max_pain, mock_fetch_chain,
            mock_pcr, mock_expirations, mock_fundamental):
        """Test successful JSON building with all components mocked."""
        # Configure fundamental data
        mock_fundamental.return_value = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "current_price": 150.0,# pylint: disable=duplicate-code
            "market_cap": 2500000000,# pylint: disable=duplicate-code
            "beta": 1.2,# pylint: disable=duplicate-code
            "fifty_two_week_high": 180.0,# pylint: disable=duplicate-code
            "fifty_two_week_low": 120.0,# pylint: disable=duplicate-code
            "dividend_yield": "0.5%",# pylint: disable=duplicate-code
            "next_earnings": "2023-11-02"
        }
        # Configure expirations, put/call ratio, and empty option chains
        mock_expirations.return_value = ["2023-10-20", "2023-10-27", "2023-11-17"]
        mock_pcr.return_value = 1.2
        calls_df = pd.DataFrame()
        puts_df = pd.DataFrame()
        mock_fetch_chain.return_value = (calls_df, puts_df)
        mock_max_pain.return_value = 150.0
        mock_iv_skew.return_value = {"itm": 30.0, "atm": 25.0, "otm": 28.0}
        mock_extract_near_atm.return_value = []
        # Historical volatility mocks
        mock_hist_data.return_value = pd.DataFrame()
        mock_hv.return_value = 0.25

        # Test without historical volatility included
        result = build_compact_options_json("AAPL", expirations_limit=2)
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["price"], 150.0)
        self.assertEqual(result["put_call_ratio"], 1.2)
        self.assertEqual(len(result["expirations"]), 0)
        self.assertNotIn("historical_volatility", result)

        # Verify that proper calls were made
        mock_fundamental.assert_called_once_with("AAPL")
        mock_expirations.assert_called_once_with("AAPL")
        mock_pcr.assert_called_once()
        self.assertEqual(mock_fetch_chain.call_count, 2)
        mock_hist_data.assert_not_called()

        # Test with historical volatility included
        result = build_compact_options_json("AAPL", expirations_limit=2, include_hv=True)
        self.assertIn("historical_volatility", result)
        self.assertEqual(result["historical_volatility"], 25.0)
        mock_hist_data.assert_called_once_with("AAPL", period="1y")
        mock_hv.assert_called_once()

    @patch('optionsllm.json_packaging.get_fundamental_data')
    def test_build_compact_options_json_no_price(self, mock_fundamental):
        """Test handling when no price data is available."""
        mock_fundamental.return_value = {
            "ticker": "BADTICKER",
            "analysis_date": "2023-10-15",
            "current_price": 0,
            "market_cap": 0,
            "beta": "N/A",
            "fifty_two_week_high": "N/A",
            "fifty_two_week_low": "N/A",
            "dividend_yield": "None",
            "next_earnings": "Unknown"
        }
        result = build_compact_options_json("BADTICKER")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Unable to retrieve price for BADTICKER")

    @patch('optionsllm.json_packaging.get_fundamental_data')
    @patch('optionsllm.json_packaging.get_option_expirations')
    def test_build_compact_options_json_no_expirations(self, mock_expirations, # pylint: disable=duplicate-code
                                                       mock_fundamental):
        """Test handling when no option expirations are available."""
        mock_fundamental.return_value = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "current_price": 150.0,# pylint: disable=duplicate-code
            "market_cap": 2500000000,# pylint: disable=duplicate-code
            "beta": 1.2,# pylint: disable=duplicate-code
            "fifty_two_week_high": 180.0,# pylint: disable=duplicate-code
            "fifty_two_week_low": 120.0,# pylint: disable=duplicate-code
            "dividend_yield": "0.5%",# pylint: disable=duplicate-code
            "next_earnings": "2023-11-02"
        }
        mock_expirations.return_value = []
        result = build_compact_options_json("AAPL")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "No options data available for AAPL")

    @patch('optionsllm.json_packaging.get_fundamental_data')
    @patch('optionsllm.json_packaging.get_option_expirations')
    @patch('optionsllm.json_packaging.compute_put_call_ratio')
    @patch('optionsllm.json_packaging.fetch_option_chain')
    def test_build_compact_options_json_partial_chains(self, mock_fetch_chain,
                                                       mock_pcr, mock_expirations,
                                                       mock_fundamental):
        """Test JSON building when some expirations have empty option chains."""
        mock_fundamental.return_value = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "current_price": 150.0, # pylint: disable=duplicate-code
            "market_cap": 2500000000,# pylint: disable=duplicate-code
            "beta": 1.2,# pylint: disable=duplicate-code
            "fifty_two_week_high": 180.0,# pylint: disable=duplicate-code
            "fifty_two_week_low": 120.0,# pylint: disable=duplicate-code
            "dividend_yield": "0.5%",# pylint: disable=duplicate-code
            "next_earnings": "2023-11-02"
        }
        mock_expirations.return_value = ["2023-10-20", "2023-10-27"]
        mock_pcr.return_value = 1.2

        calls_df_valid = pd.DataFrame({
            'strike': [145, 150, 155],
            'lastPrice': [10, 5, 2],
            'bid': [9.5, 4.8, 1.9],
            'ask': [10.5, 5.2, 2.1],
            'impliedVolatility': [0.3, 0.28, 0.32],
            'openInterest': [100, 200, 150],
            'volume': [50, 75, 60]
        })
        puts_df_valid = pd.DataFrame({
            'strike': [145, 150, 155],
            'lastPrice': [8, 4, 1.5],
            'bid': [7.8, 3.9, 1.4],
            'ask': [8.2, 4.1, 1.6],
            'impliedVolatility': [0.33, 0.29, 0.35],
            'openInterest': [80, 160, 120],
            'volume': [40, 65, 55]
        })
        # First expiration returns valid chain data; second returns empty DataFrames.
        mock_fetch_chain.side_effect = [
            (calls_df_valid, puts_df_valid),
            (pd.DataFrame(), pd.DataFrame())
        ]
        result = build_compact_options_json("AAPL", expirations_limit=2)
        self.assertEqual(len(result["expirations"]), 1)
        self.assertIn("2023-10-20", result["expirations"])

    @patch('optionsllm.json_packaging.get_fundamental_data')
    @patch('optionsllm.json_packaging.get_option_expirations')
    @patch('optionsllm.json_packaging.compute_put_call_ratio')
    @patch('optionsllm.json_packaging.fetch_option_chain')
    @patch('optionsllm.json_packaging.calculate_max_pain')
    @patch('optionsllm.json_packaging.compute_iv_skew')
    @patch('optionsllm.json_packaging._extract_near_atm')
    @patch('optionsllm.json_packaging.get_historical_data')
    @patch('optionsllm.json_packaging.compute_historical_volatility')
    def test_build_compact_options_json_hv_none(
            self, mock_hv, mock_hist_data, mock_extract_near_atm,
            mock_iv_skew, mock_max_pain, mock_fetch_chain,
            mock_pcr, mock_expirations, mock_fundamental):
        """Test JSON building when historical volatility computation returns None."""
        mock_fundamental.return_value = { # pylint: disable=duplicate-code
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "current_price": 150.0,  # pylint: disable=duplicate-code
            "market_cap": 2500000000,# pylint: disable=duplicate-code
            "beta": 1.2,# pylint: disable=duplicate-code
            "fifty_two_week_high": 180.0,# pylint: disable=duplicate-code
            "fifty_two_week_low": 120.0,# pylint: disable=duplicate-code
            "dividend_yield": "0.5%",# pylint: disable=duplicate-code
            "next_earnings": "2023-11-02"
        }
        mock_expirations.return_value = ["2023-10-20", "2023-10-27"]
        mock_pcr.return_value = 1.2
        calls_df = pd.DataFrame({
            'strike': [145, 150, 155],
            'lastPrice': [10, 5, 2],
            'bid': [9.5, 4.8, 1.9],
            'ask': [10.5, 5.2, 2.1],
            'impliedVolatility': [0.3, 0.28, 0.32],
            'openInterest': [100, 200, 150],
            'volume': [50, 75, 60]
        })
        puts_df = pd.DataFrame()
        mock_fetch_chain.return_value = (calls_df, puts_df)
        mock_max_pain.return_value = 150.0
        mock_iv_skew.return_value = {"itm": 30.0, "atm": 25.0, "otm": 28.0}
        mock_extract_near_atm.return_value = []
        mock_hist_data.return_value = pd.DataFrame()
        mock_hv.return_value = None

        result = build_compact_options_json("AAPL", expirations_limit=2, include_hv=True)
        self.assertIn("historical_volatility", result)
        self.assertIsNone(result["historical_volatility"])

    def test_extract_near_atm_normal_case(self):
        """Test extracting near-ATM options with normal input for call options."""
        df = pd.DataFrame({
            'strike': [90, 95, 100, 105, 110, 115, 120],
            'lastPrice': [15.0, 10.0, 5.0, 2.5, 1.0, 0.5, 0.25],
            'bid': [14.8, 9.8, 4.8, 2.3, 0.9, 0.4, 0.2],
            'ask': [15.2, 10.2, 5.2, 2.7, 1.1, 0.6, 0.3],
            'impliedVolatility': [0.35, 0.32, 0.30, 0.28, 0.31, 0.33, 0.38],
            'openInterest': [200, 350, 500, 600, 400, 250, 150],
            'volume': [50, 75, 150, 200, 125, 70, 30]
        })
        with patch('optionsllm.json_packaging.calculate_greeks') as mock_greeks:
            mock_greeks.return_value = (0.6, 0.05, -0.1, 0.2)
            current_price = 105.0
            dte = 30
            result = _extract_near_atm(df, 'c', current_price, dte)
            self.assertEqual(len(result), 5)
            strikes = [r['strike'] for r in result]
            moneyness = [r['moneyness'] for r in result]
            self.assertEqual(strikes, [95.0, 100.0, 105.0, 110.0, 115.0])
            self.assertEqual(moneyness[0], "ITM")
            self.assertEqual(moneyness[1], "ITM")
            self.assertEqual(moneyness[2], "ATM")
            self.assertEqual(moneyness[3], "OTM")
            self.assertEqual(moneyness[4], "OTM")
            self.assertEqual(mock_greeks.call_count, 5)

    def test_extract_near_atm_empty_dataframe(self):
        """Test extracting near-ATM options with an empty DataFrame."""
        df = pd.DataFrame()
        result = _extract_near_atm(df, 'c', 100.0, 30)
        self.assertEqual(result, [])

    def test_extract_near_atm_no_valid_atm(self):
        """Test handling when no valid ATM index can be found."""
        df = pd.DataFrame({
            'strike': [float('nan'), float('nan')],
            'lastPrice': [10.0, 5.0],
            'impliedVolatility': [0.3, 0.25]
        })
        result = _extract_near_atm(df, 'c', 100.0, 30)
        self.assertEqual(result, [])

    def test_extract_near_atm_put_options(self):
        """Test extracting near-ATM options for put options."""
        df = pd.DataFrame({
            'strike': [90, 95, 100, 105, 110, 115, 120],
            'lastPrice': [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            'bid': [1.9, 2.4, 2.9, 3.4, 3.9, 4.4, 4.9],
            'ask': [2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1],
            'impliedVolatility': [0.25, 0.27, 0.30, 0.28, 0.26, 0.29, 0.31],
            'openInterest': [100, 150, 200, 250, 300, 350, 400],
            'volume': [10, 15, 20, 25, 30, 35, 40]
        })
        with patch('optionsllm.json_packaging.calculate_greeks') as mock_greeks:
            mock_greeks.return_value = (0.4, 0.02, -0.05, 0.1)
            result = _extract_near_atm(df, 'p', 105.0, 30)
            self.assertEqual(len(result), 5)
            for option in result:
                if option["strike"] > 105.0 * 1.02:
                    self.assertEqual(option["moneyness"], "ITM")
                elif option["strike"] < 105.0 * 0.98:
                    self.assertEqual(option["moneyness"], "OTM")
                else:
                    self.assertEqual(option["moneyness"], "ATM")
            self.assertEqual(mock_greeks.call_count, len(result))

    def test_extract_near_atm_single_row(self):
        """Test extracting near-ATM options when DataFrame has only one row."""
        df = pd.DataFrame({
            'strike': [100],
            'lastPrice': [5.0],
            'bid': [4.8],
            'ask': [5.2],
            'impliedVolatility': [0.3],
            'openInterest': [100],
            'volume': [50]
        })
        with patch('optionsllm.json_packaging.calculate_greeks') as mock_greeks:
            mock_greeks.return_value = (0.5, 0.03, -0.04, 0.2)
            result = _extract_near_atm(df, 'c', 100.0, 30)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['strike'], 100.0)
            self.assertEqual(result[0]['moneyness'], "ATM")


if __name__ == '__main__':
    unittest.main()

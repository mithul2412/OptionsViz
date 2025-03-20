#!/usr/bin/env python3
"""
Unit tests for the advanced options calculations module.

These tests verify the correctness of calculations for:
- Max pain
- Implied volatility skew
"""

import unittest
import pandas as pd
from optionsllm.advanced_options import (
    calculate_max_pain,
    compute_iv_skew,
)


class TestAdvancedOptions(unittest.TestCase):
    """Test suite for advanced options calculations."""

    def test_calculate_max_pain_normal_case(self):
        """Test max pain calculation with normal input data."""
        calls_df = pd.DataFrame({
            'strike': [90, 95, 100, 105, 110],
            'openInterest': [100, 500, 700, 300, 100],
        })
        puts_df = pd.DataFrame({
            'strike': [90, 95, 100, 105, 110],
            'openInterest': [100, 200, 600, 500, 300],
        })
        result = calculate_max_pain(calls_df, puts_df)
        self.assertEqual(result, 100)

    def test_calculate_max_pain_empty_dataframes(self):
        """Test max pain calculation with empty dataframes."""
        calls_df = pd.DataFrame()
        puts_df = pd.DataFrame()
        result = calculate_max_pain(calls_df, puts_df)
        self.assertIsNone(result)

    def test_calculate_max_pain_one_empty_dataframe(self):
        """Test max pain calculation with one empty dataframe."""
        calls_df = pd.DataFrame({
            'strike': [90, 95, 100],
            'openInterest': [100, 200, 300],
        })
        puts_df = pd.DataFrame()
        result = calculate_max_pain(calls_df, puts_df)
        self.assertIsNone(result)

    def test_compute_iv_skew_normal_case(self):
        """Test IV skew calculation with normal input data."""
        calls_df = pd.DataFrame({
            'strike': [90, 95, 100, 105, 110],
            'impliedVolatility': [0.40, 0.35, 0.30, 0.32, 0.38],
        })
        current_price = 100.0
        result = compute_iv_skew(calls_df, current_price)
        self.assertEqual(result.get('atm'), 30.0)  # 0.30 * 100
        self.assertEqual(result.get('itm'), 35.0)  # 0.35 * 100
        self.assertEqual(result.get('otm'), 32.0)  # 0.32 * 100

    def test_compute_iv_skew_empty_dataframe(self):
        """Test IV skew calculation with empty dataframe."""
        calls_df = pd.DataFrame()
        current_price = 100.0
        result = compute_iv_skew(calls_df, current_price)
        self.assertEqual(result, {})

    def test_compute_iv_skew_no_itm_options(self):
        """Test IV skew calculation with no ITM options."""
        calls_df = pd.DataFrame({
            'strike': [100, 105, 110],
            'impliedVolatility': [0.30, 0.32, 0.35],
        })
        current_price = 95.0
        result = compute_iv_skew(calls_df, current_price)
        # ATM should be at strike 100, OTM at strike 105; no ITM options available
        self.assertEqual(result.get('atm'), 30.0)
        self.assertEqual(result.get('otm'), 30.0)

    def test_compute_iv_skew_no_otm_options(self):
        """Test IV skew calculation with no OTM options."""
        calls_df = pd.DataFrame({
            'strike': [90, 95, 100],
            'impliedVolatility': [0.40, 0.35, 0.30],
        })
        current_price = 105.0
        result = compute_iv_skew(calls_df, current_price)
        # ATM should be at strike 100, ITM at strike 95; no OTM options available
        self.assertEqual(result.get('atm'), 30.0)
        self.assertEqual(result.get('itm'), 30.0)
        self.assertIsNone(result.get('otm'))


def main() -> None:
    """Entry point for the unit tests."""
    unittest.main()


if __name__ == '__main__':
    main()

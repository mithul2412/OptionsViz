# test_main.py
"""
Unit tests for the main orchestration module.

These tests verify the text summarization functionality and
the LLM integration with appropriate mocking to avoid actual API calls.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call
from optionsllm.main import summarize_options_data, ask_llm_about_options, main


# Add the project directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMain(unittest.TestCase):
    """Test suite for the main module functions."""

    def test_summarize_options_data_complete(self):
        """Test summarization of a complete options data structure."""
        # Sample options data
        data = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "price": 150.0,
            "market_cap": 2500000000,
            "beta": 1.2,
            "fifty_two_week_high": 180.0,
            "fifty_two_week_low": 120.0,
            "dividend_yield": "0.5%",
            "next_earnings": "2023-11-02",
            "put_call_ratio": 1.2,
            "historical_volatility": 25.0,
            "expirations": {
                "2023-10-20": {
                    "days_to_expiry": 5,
                    "max_pain_strike": 150.0,
                    "iv_skew_calls": {"itm": 30.0, "atm": 25.0, "otm": 28.0},
                    "calls": [
                        {
                            "strike": 145.0,
                            "moneyness": "ITM",
                            "last": 6.0,
                            "bid": 5.8,
                            "ask": 6.2,
                            "iv": 30.0,
                            "delta": 0.7,
                            "gamma": 0.05,
                            "theta": -0.1,
                            "vega": 0.2,
                            "open_interest": 500,
                            "volume": 150
                        },
                        {
                            "strike": 150.0,
                            "moneyness": "ATM",
                            "last": 3.0,
                            "bid": 2.8,
                            "ask": 3.2,
                            "iv": 25.0,
                            "delta": 0.5,
                            "gamma": 0.06,
                            "theta": -0.12,
                            "vega": 0.25,
                            "open_interest": 1000,
                            "volume": 300
                        }
                    ],
                    "puts": [
                        {
                            "strike": 145.0,
                            "moneyness": "OTM",
                            "last": 1.0,
                            "bid": 0.9,
                            "ask": 1.1,
                            "iv": 28.0,
                            "delta": -0.3,
                            "gamma": 0.05,
                            "theta": -0.08,
                            "vega": 0.15,
                            "open_interest": 700,
                            "volume": 200
                        }
                    ]
                }
            }
        }

        # Call the function
        result = summarize_options_data(data)

        # Assertions
        self.assertIsInstance(result, str)
        self.assertIn("Ticker: AAPL", result)
        self.assertIn("Price: $150.0", result)
        self.assertIn("Put/Call Ratio: 1.2", result)
        self.assertIn("Historical Volatility (1y): 25.00%", result)
        self.assertIn("EXPIRATION: 2023-10-20", result)
        self.assertIn("Max Pain: 150.0", result)
        self.assertIn("IV Skew(Calls)-ITM:30.0%,ATM:25.0%,OTM:28.0%", result)
        self.assertIn("Strike=145.0 (ITM) IV=30.0%", result)
        self.assertIn("Strike=145.0 (OTM) IV=28.0%", result)

    def test_summarize_options_data_error(self):
        """Test summarization when data contains an error."""
        # Sample data with error
        data = {
            "error": "No options data available for BADTICKER"
        }

        # Call the function
        result = summarize_options_data(data)

        # Assertions
        self.assertEqual(result, "No options data available for BADTICKER")

    def test_summarize_options_data_no_historicalvol(self):
        """Test summarization without historical volatility."""
        # Sample data without historical volatility
        data = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "price": 150.0,
            "market_cap": 2500000000,
            "beta": 1.2,
            "fifty_two_week_high": 180.0,
            "fifty_two_week_low": 120.0,
            "dividend_yield": "0.5%",
            "next_earnings": "2023-11-02",
            "put_call_ratio": 1.2,
            "expirations": {}
        }

        # Call the function
        result = summarize_options_data(data)

        # Assertions
        self.assertIn("Ticker: AAPL", result)
        self.assertNotIn("Historical Volatility", result)

    def test_summarize_options_data_with_hv_none(self):
        """Test summarization with historical volatility set to None."""
        data = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "price": 150.0,
            "market_cap": 2500000000,
            "beta": 1.2,
            "fifty_two_week_high": 180.0,
            "fifty_two_week_low": 120.0,
            "dividend_yield": "0.5%",
            "next_earnings": "2023-11-02",
            "put_call_ratio": 1.2,
            "historical_volatility": None,
            "expirations": {}
        }

        result = summarize_options_data(data)

        self.assertIn("Ticker: AAPL", result)
        self.assertNotIn("Historical Volatility", result)

    def test_summarize_options_data_without_iv_skew(self):
        """Test summarization without IV skew data."""
        data = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "price": 150.0,
            "market_cap": 2500000000,
            "beta": 1.2,
            "fifty_two_week_high": 180.0,
            "fifty_two_week_low": 120.0,
            "dividend_yield": "0.5%",
            "next_earnings": "2023-11-02", 
            "put_call_ratio": 1.2,
            "expirations": {
                "2023-10-20": {
                    "days_to_expiry": 5,
                    "max_pain_strike": 150.0,
                    "iv_skew_calls": None,  # IV skew is None
                    "calls": [],
                    "puts": []
                }
            }
        }

        result = summarize_options_data(data)

        self.assertIn("EXPIRATION: 2023-10-20", result)
        self.assertIn("Max Pain: 150.0", result)
        self.assertNotIn("IV Skew", result)

    def test_summarize_options_data_empty_expirations(self):
        """Test summarization with empty expirations data."""
        data = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "price": 150.0,
            "market_cap": 2500000000,
            "beta": 1.2,
            "fifty_two_week_high": 180.0,
            "fifty_two_week_low": 120.0,
            "dividend_yield": "0.5%",
            "next_earnings": "2023-11-02", 
            "put_call_ratio": 1.2,
            "expirations": {}
        }

        result = summarize_options_data(data)

        self.assertIn("Ticker: AAPL", result)
        self.assertIn("Put/Call Ratio: 1.2", result)
        self.assertNotIn("EXPIRATION:", result)

    @patch('optionsllm.main.OpenAI')
    @patch('optionsllm.main.LLM_AVAILABLE', True)
    def test_ask_llm_about_options_openrouter(self, mock_openai):
        """Test LLM query using OpenRouter."""
        # Configure the mock client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock completion response
        mock_completion = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "This is a test LLM response"

        # Mock environment variables
        with patch.dict('os.environ', {"OPENROUTER_API_KEY": "test_key"}):
            # Call the function
            result = ask_llm_about_options("Sample options data", "What do you think?")

            # Assertions
            self.assertEqual(result, "This is a test LLM response")
            mock_openai.assert_called_with(
                base_url="https://openrouter.ai/api/v1",
                api_key="test_key"
            )
            mock_client.chat.completions.create.assert_called_once()

            # Check that the correct model was used
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            self.assertEqual(call_kwargs['model'], "deepseek/deepseek-r1-zero:free")

            # Check extra headers
            self.assertEqual(call_kwargs['extra_headers']['HTTP-Referer'],
                            "https://options-analysis-tool.com")
            self.assertEqual(call_kwargs['extra_headers']['X-Title'],
                            "OptionsAnalysisLLM")

    @patch('optionsllm.main.OpenAI')
    @patch('optionsllm.main.LLM_AVAILABLE', True)
    def test_ask_llm_about_options_openai_gpt4(self, mock_openai):
        """Test LLM query using OpenAI with GPT-4."""
        # Configure the mock client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock completion response
        mock_completion = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "This is a test LLM response"

        # Mock models list to include gpt-4
        mock_client.models.list.return_value = ["gpt-4", "gpt-3.5-turbo"]

        # Mock environment variables (only OpenAI key)
        with patch.dict('os.environ', {"OPENAI_API_KEY": "test_key", "OPENROUTER_API_KEY": ""}):
            # Call the function
            result = ask_llm_about_options("Sample options data", "What do you think?")

            # Assertions
            self.assertEqual(result, "This is a test LLM response")
            mock_openai.assert_called_with(api_key="test_key")
            mock_client.chat.completions.create.assert_called_once()

            # Check that GPT-4 was used
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            self.assertEqual(call_kwargs['model'], "gpt-4")

            # Check empty extra headers
            self.assertEqual(call_kwargs['extra_headers'], {})

    @patch('optionsllm.main.OpenAI')
    @patch('optionsllm.main.LLM_AVAILABLE', True)
    def test_ask_llm_about_options_openai_gpt35(self, mock_openai):
        """Test LLM query using OpenAI with GPT-3.5 when GPT-4 is not available."""
        # Configure the mock client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock completion response
        mock_completion = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "This is a test LLM response"

        # Mock models list to NOT include gpt-4
        mock_client.models.list.return_value = ["gpt-3.5-turbo"]

        # Mock environment variables (only OpenAI key)
        with patch.dict('os.environ', {"OPENAI_API_KEY": "test_key", "OPENROUTER_API_KEY": ""}):
            # Call the function
            result = ask_llm_about_options("Sample options data", "What do you think?")

            # Assertions
            self.assertEqual(result, "This is a test LLM response")
            mock_openai.assert_called_with(api_key="test_key")
            mock_client.chat.completions.create.assert_called_once()

            # Check that GPT-3.5-turbo was used
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            self.assertEqual(call_kwargs['model'], "gpt-3.5-turbo")

    @patch('optionsllm.main.LLM_AVAILABLE', False)
    def test_ask_llm_about_options_no_openai(self):
        """Test LLM query when OpenAI library is not available."""
        result = ask_llm_about_options("Sample options data", "What do you think?")

        # Assertions
        self.assertEqual(
            result,
            "Error: OpenAI library not installed. "
        )

    @patch('optionsllm.main.OpenAI')
    @patch('optionsllm.main.LLM_AVAILABLE', True)
    def test_ask_llm_about_options_no_api_key(self, mock_openai):
        """Test LLM query when no API key is available."""
        # Mock environment variables (no keys)
        with patch.dict('os.environ', {"OPENROUTER_API_KEY": "", "OPENAI_API_KEY": ""}):
            result = ask_llm_about_options("Sample options data", "What do you think?")

            # Assertions
            self.assertEqual(
                result, "Error: No API key found. "
            )

            mock_openai.assert_not_called()


    @patch('optionsllm.main.build_compact_options_json')
    @patch('sys.argv', ['main.py', 'AAPL'])
    @patch('builtins.print')
    def test_main_function_basic(self, mock_print, mock_build_json):
        """Test main function basic execution path."""
        # Configure mocks
        mock_build_json.return_value = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "price": 150.0,
            "market_cap": 2500000000,
            "beta": 1.2,
            "fifty_two_week_high": 180.0,
            "fifty_two_week_low": 120.0,
            "dividend_yield": "0.5%",
            "next_earnings": "2023-11-02",
            "put_call_ratio": 1.2,
            "expirations": {}
        }

        # Run the main function
        with patch('argparse.ArgumentParser.parse_args') as mock_args:
            mock_args.return_value = MagicMock(
                ticker="AAPL",
                expirations=3,
                hv=False,
                tokens=1000,
                interactive=False
            )
            main()

        # Verify
        mock_build_json.assert_called_with("AAPL", expirations_limit=3, include_hv=False)
        mock_print.assert_any_call("=== OPTIONS SUMMARY ===\n")
        self.assertTrue(mock_print.call_count >= 2)  # At least called with title and summary

    @patch('optionsllm.main.build_compact_options_json')
    @patch('optionsllm.main.LLM_AVAILABLE', False)
    @patch('builtins.print')
    def test_main_interactive_no_llm(self, mock_print, mock_build_json):
        """Test main function with interactive mode but LLM not available."""
        # Configure mocks - Fixed mock data to include all required fields
        mock_build_json.return_value = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "price": 150.0,
            "market_cap": 2500000000,
            "beta": 1.2,
            "fifty_two_week_high": 180.0,
            "fifty_two_week_low": 120.0,
            "dividend_yield": "0.5%",
            "next_earnings": "2023-11-02",
            "put_call_ratio": 1.2,
            "expirations": {}
        }

        # Run the main function
        with patch('argparse.ArgumentParser.parse_args') as mock_args:
            mock_args.return_value = MagicMock(
                ticker="AAPL",
                expirations=3,
                hv=False,
                tokens=1000,
                interactive=True
            )
            main()

        # Verify warning was printed
        mock_print.assert_any_call("OpenAI library not installed. LLM query mode unavailable.")
        mock_print.assert_any_call("Install it with: pip install openai")

    @patch('optionsllm.main.build_compact_options_json')
    @patch('optionsllm.main.LLM_AVAILABLE', True)
    @patch('builtins.print')
    @patch('builtins.input')
    @patch('optionsllm.main.ask_llm_about_options')
    def test_main_interactive_with_llm(self, mock_ask_llm, mock_input, mock_print, mock_build_json):
        """Test main function with interactive mode and LLM available."""
        # Configure mocks - Fixed mock data to include all required fields
        mock_build_json.return_value = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "price": 150.0,
            "market_cap": 2500000000,
            "beta": 1.2,
            "fifty_two_week_high": 180.0,
            "fifty_two_week_low": 120.0,
            "dividend_yield": "0.5%",
            "next_earnings": "2023-11-02",
            "put_call_ratio": 1.2,
            "expirations": {}
        }

        # Set up input sequence: ask a question, then quit
        mock_input.side_effect = ["What do you think?", "quit"]
        mock_ask_llm.return_value = "This is an LLM response"

        # Run the main function
        with patch('argparse.ArgumentParser.parse_args') as mock_args:
            mock_args.return_value = MagicMock(
                ticker="AAPL",
                expirations=3,
                hv=True,
                tokens=1000,
                interactive=True
            )
            main()

        # Verify the LLM was called with correct parameters
        mock_ask_llm.assert_called_once()
        mock_print.assert_any_call("\n=== LLM RESPONSE ===\n")
        mock_print.assert_any_call("This is an LLM response")

    @patch('optionsllm.main.build_compact_options_json')
    @patch('optionsllm.main.LLM_AVAILABLE', True)
    @patch('builtins.print')
    @patch('builtins.input')
    @patch('optionsllm.main.ask_llm_about_options')
    def test_main_interactive_empty_input(
        self, mock_ask_llm, mock_input, mock_print, mock_build_json
        ):
        """Test main function with interactive mode and empty input."""
        # Configure mocks - Fixed mock data to include all required fields
        mock_build_json.return_value = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "price": 150.0,
            "market_cap": 2500000000,
            "beta": 1.2,
            "fifty_two_week_high": 180.0,
            "fifty_two_week_low": 120.0,
            "dividend_yield": "0.5%",
            "next_earnings": "2023-11-02",
            "put_call_ratio": 1.2,
            "expirations": {}
        }

        # Set up input sequence: empty input, then quit
        mock_input.side_effect = ["", "quit"]

        # Run the main function
        with patch('argparse.ArgumentParser.parse_args') as mock_args:
            mock_args.return_value = MagicMock(
                ticker="AAPL",
                expirations=3,
                hv=True,
                tokens=1000,
                interactive=True
            )
            main()

        # Verify the prompt for empty input
        mock_print.assert_any_call("Please enter a question or type 'quit' to exit.")
        mock_ask_llm.assert_not_called()

    @patch('optionsllm.main.build_compact_options_json')
    def test_main_with_error_data(self, mock_build_json):
        """Test main function when options data contains an error."""
        # Configure mock to return error data
        mock_build_json.return_value = {
            "error": "No options data available for BADTICKER"
        }

        # Run the main function with interactive mode
        with patch('argparse.ArgumentParser.parse_args') as mock_args, \
             patch('builtins.print') as mock_print:
            mock_args.return_value = MagicMock(
                ticker="BADTICKER",
                expirations=3,
                hv=False,
                tokens=1000,
                interactive=True
            )
            main()

        # Verify interactive mode was not entered due to error
        mock_print.assert_any_call("=== OPTIONS SUMMARY ===\n")
        mock_print.assert_any_call("No options data available for BADTICKER")
        # Should not see any interactive prompts

    @patch('optionsllm.main.build_compact_options_json')
    @patch('optionsllm.main.LLM_AVAILABLE', True)
    @patch('builtins.print')
    @patch('builtins.input')
    @patch('optionsllm.main.ask_llm_about_options')
    def test_main_interactive_multiple_questions(
        self, mock_ask_llm, mock_input, mock_print,mock_build_json
        ):
        """Test main function with multiple questions in interactive mode."""
        # Configure mocks
        mock_build_json.return_value = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "price": 150.0,
            "market_cap": 2500000000,
            "beta": 1.2,
            "fifty_two_week_high": 180.0,
            "fifty_two_week_low": 120.0,
            "dividend_yield": "0.5%",
            "next_earnings": "2023-11-02",
            "put_call_ratio": 1.2,
            "expirations": {}
        }

        # Set up input sequence: ask multiple questions, then quit
        mock_input.side_effect = ["First question?", "Second question?", "quit"]
        mock_ask_llm.side_effect = ["First response", "Second response"]

        # Run the main function
        with patch('argparse.ArgumentParser.parse_args') as mock_args:
            mock_args.return_value = MagicMock(
                ticker="AAPL",
                expirations=3,
                hv=False,
                tokens=1000,
                interactive=True
            )
            main()

        # Verify the LLM was called twice with correct parameters
        self.assertEqual(mock_ask_llm.call_count, 2)
        mock_ask_llm.assert_has_calls([
            call(unittest.mock.ANY, "First question?", max_tokens=1000),
            call(unittest.mock.ANY, "Second question?", max_tokens=1000)
        ])

        # Verify both responses were printed
        mock_print.assert_any_call("First response")
        mock_print.assert_any_call("Second response")

    @patch('optionsllm.main.build_compact_options_json')
    @patch('optionsllm.main.LLM_AVAILABLE', True)
    @patch('builtins.print')
    @patch('builtins.input')
    @patch('optionsllm.main.ask_llm_about_options')
    def test_main_interactive_alternate_exit_commands(
        self, mock_ask_llm, mock_input, _mock_print, mock_build_json
        ):
        """Test main function with different exit commands in interactive mode."""
        # Configure mocks
        mock_build_json.return_value = {
            "ticker": "AAPL",
            "analysis_date": "2023-10-15",
            "price": 150.0,
            "market_cap": 2500000000,
            "beta": 1.2,
            "fifty_two_week_high": 180.0,
            "fifty_two_week_low": 120.0,
            "dividend_yield": "0.5%",
            "next_earnings": "2023-11-02",
            "put_call_ratio": 1.2,
            "expirations": {}
        }

        # Test each exit command
        for exit_cmd in ["exit", "q"]:
            mock_input.reset_mock()
            mock_input.side_effect = [exit_cmd]

            # Run the main function
            with patch('argparse.ArgumentParser.parse_args') as mock_args:
                mock_args.return_value = MagicMock(
                    ticker="AAPL",
                    expirations=3,
                    hv=False,
                    tokens=1000,
                    interactive=True
                )
                main()
            # Verify LLM was not called
            mock_ask_llm.assert_not_called()
            mock_ask_llm.reset_mock()


if __name__ == '__main__':
    unittest.main()

# test_llm_interface.py
"""
Unit tests for the LLM interface module.

These tests verify the correct behavior of the LLMInterface class,
particularly how it handles API calls, formats context, and manages errors.
All API calls are mocked to avoid actual external service dependencies.
"""

import unittest
from unittest.mock import patch, MagicMock

from news.llm_interface import LLMInterface


class TestLLMInterface(unittest.TestCase):
    """Test suite for the LLM interface functionality."""

    @patch('news.llm_interface.OpenAI')
    def test_init_with_api_key(self, mock_openai):
        """Test initialization with a provided API key."""
        # Create a mock OpenAI client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Initialize LLMInterface with custom parameters
        llm = LLMInterface(
            api_key="test_api_key",
            model="test/model",
            site_url="https://example.com",
            site_name="TestApp"
        )

        # Verify OpenAI client was initialized correctly
        mock_openai.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="test_api_key"
        )

        # Check that attributes were set correctly
        self.assertEqual(llm.client, mock_client)
        self.assertEqual(llm.model, "test/model")
        self.assertEqual(llm.extra_headers["HTTP-Referer"], "https://example.com")
        self.assertEqual(llm.extra_headers["X-Title"], "TestApp")

    @patch('news.llm_interface.os.getenv')
    @patch('news.llm_interface.OpenAI')
    def test_init_with_env_var(self, mock_openai, mock_getenv):
        """Test initialization using environment variable for API key."""

        # Setup mocks
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_getenv.return_value = "env_api_key"

        # Initialize LLMInterface without explicitly providing API key
        llm = LLMInterface()
        self.assertIsInstance(llm, LLMInterface)

        # Verify environment variable was checked
        mock_getenv.assert_called_once_with('OPENROUTER_API_KEY')

        # Verify OpenAI client was initialized with the env key
        mock_openai.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="env_api_key"
        )

    @patch('news.llm_interface.os.getenv')
    def test_init_missing_api_key(self, mock_getenv):
        """Test error handling when no API key is available."""
        # Setup mock to return None for OPENROUTER_API_KEY
        mock_getenv.return_value = None

        # Check that attempting to create LLMInterface without an API key raises an error
        with self.assertRaises(ValueError):
            LLMInterface()

    @patch('news.llm_interface.OpenAI')
    def test_query_without_context(self, mock_openai):
        """Test querying the LLM without providing context."""

        # Setup mocks
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        self.assertIsNotNone(mock_message)

        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [mock_choice]
        mock_choice.message.content = "Test response from LLM"

        # Initialize LLMInterface and make a query
        llm = LLMInterface(api_key="test_api_key")
        response = llm.query("What is the latest news?")

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]

        # Check model and headers
        self.assertEqual(call_args["model"], "deepseek/deepseek-chat:free")
        self.assertEqual(call_args["extra_headers"], llm.extra_headers)

        # Check messages format
        messages = call_args["messages"]
        self.assertEqual(len(messages), 2)  # System prompt + user query
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "What is the latest news?")

        # Check response
        self.assertEqual(response, "Test response from LLM")

    @patch('news.llm_interface.OpenAI')
    def test_query_with_context(self, mock_openai):
        """Test querying the LLM with context documents."""

        # Setup mocks
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        self.assertIsNotNone(mock_message)

        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [mock_choice]
        mock_choice.message.content = "Test response from LLM with context"

        # Create context documents
        context = [
            {
                "title": "Test Article 1",
                "source": "Test Source",
                "published_at": "2023-10-15",
                "text": "This is test content.",
                "url": "https://example.com/1"
            }
        ]

        # Initialize LLMInterface and make a query with context
        llm = LLMInterface(api_key="test_api_key")
        response = llm.query("What does the article say?", context)

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]

        # Check messages format with context
        messages = call_args["messages"]
        self.assertEqual(len(messages), 2)  # System prompt with context + user query

        # Check that the system prompt contains context elements
        system_content = messages[0]["content"]
        self.assertIn("Test Article 1", system_content)
        self.assertIn("Test Source", system_content)
        self.assertIn("2023-10-15", system_content)
        self.assertIn("This is test content", system_content)

        # Check the user query
        self.assertEqual(messages[1]["content"], "What does the article say?")

        # Check response
        self.assertEqual(response, "Test response from LLM with context")

    def test_format_context(self):
        """Test formatting of context documents."""

        # Create an LLMInterface instance
        with patch('news.llm_interface.OpenAI'):
            llm = LLMInterface(api_key="test_api_key")

        # Test documents
        documents = [
            {
                "title": "Article 1",
                "source": "Source 1",
                "published_at": "2023-10-15",
                "content_preview": "Content preview 1",
                "url": "https://example.com/1"
            },
            {
                "title": "Article 2",
                "source": "Source 2",
                "text": "Full text content 2"
            }
        ]

        # pylint: disable=protected-access
        formatted = llm._format_context(documents)
        # pylint: enable=protected-access
        self.assertIsInstance(formatted, str)  # Ensure it returns a string


        # Check formatting
        self.assertIn("ARTICLE 1:", formatted)
        self.assertIn("ARTICLE 2:", formatted)
        self.assertIn("Title: Article 1", formatted)
        self.assertIn("Source: Source 1", formatted)
        self.assertIn("Published: 2023-10-15", formatted)
        self.assertIn("Content: Content preview 1", formatted)
        self.assertIn("URL: https://example.com/1", formatted)

        # Check that 'text' is used when 'content_preview' is not available
        self.assertIn("Content: Full text content 2", formatted)

        # Check that URL is not included when not provided
        second_article = formatted.split("ARTICLE 2:")[1]
        self.assertNotIn("URL:", second_article)

    @patch('news.llm_interface.OpenAI')
    def test_query_with_empty_context(self, mock_openai):
        """Test querying the LLM with an empty context list."""
        # Setup mocks
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        self.assertIsNotNone(mock_message)

        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [mock_choice]
        mock_choice.message.content = "Test response from LLM"

        # Initialize LLMInterface and make a query with empty context
        llm = LLMInterface(api_key="test_api_key")
        response = llm.query("What is the latest news?", [])

        # Verify system message doesn't include context formatting
        call_args = mock_client.chat.completions.create.call_args[1]
        messages = call_args["messages"]

        # Should be treated same as no context
        self.assertEqual(messages[0]["content"], "You are a news analysis assistant")
        self.assertEqual(response, "Test response from LLM")

    @patch('news.llm_interface.OpenAI')
    def test_query_api_error(self, mock_openai):
        """Test error handling during API calls."""
        # Setup mock to raise an exception
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        # Initialize LLMInterface
        llm = LLMInterface(api_key="test_api_key")

        # Check that the exception is propagated
        with self.assertRaises(Exception) as context:
            llm.query("What is the latest news?")

        self.assertEqual(str(context.exception), "API Error")

    @patch('news.llm_interface.OpenAI')
    def test_analyze_sentiment_default_prompt(self, mock_openai):
        """Test sentiment analysis with default prompt template."""
        # Setup mocks
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = """
        Sentiment: Positive
        Sentiment Score: 0.8
        Key Themes: Economic Growth, Technology, Innovation
        Emotional Tone: Optimistic, Enthusiastic
        """
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        # Initialize LLMInterface
        llm = LLMInterface(api_key="test_api_key")

        # Call analyze_sentiment
        text = "The company announced record profits and new innovations."
        result = llm.analyze_sentiment(text)

        # Verify the API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        messages = call_args["messages"]

        # Check that the default prompt template was used
        self.assertIn("TEXT TO ANALYZE:", messages[1]["content"])
        self.assertIn(text, messages[1]["content"])

        # Check the returned result
        self.assertEqual(result["sentiment"], "Positive")
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["themes"], ["Economic Growth", "Technology", "Innovation"])
        self.assertEqual(result["tone"], "Optimistic, Enthusiastic")

    @patch('news.llm_interface.OpenAI')
    def test_analyze_sentiment_custom_prompt(self, mock_openai):
        """Test sentiment analysis with custom prompt template."""
        # Setup mocks
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = """
        Sentiment: Negative
        Sentiment Score: -0.5
        Key Themes: Market Downturn, Financial Loss
        Emotional Tone: Concerned
        """
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        # Initialize LLMInterface
        llm = LLMInterface(api_key="test_api_key")

        # Call analyze_sentiment with custom prompt
        text = "The stock market crashed today."
        custom_prompt = (
            "Analyze this text: {text}\nProvide:\nSentiment: [value]\n"
            "Sentiment Score: [value]\nKey Themes: [values]\nEmotional Tone: [values]"
            )
        result = llm.analyze_sentiment(text, custom_prompt)

        # Verify the API call used the custom prompt
        call_args = mock_client.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        self.assertEqual(messages[1]["content"], custom_prompt.format(text=text))

        # Check the returned result
        self.assertEqual(result["sentiment"], "Negative")
        self.assertEqual(result["score"], -0.5)
        self.assertEqual(result["themes"], ["Market Downturn", "Financial Loss"])
        self.assertEqual(result["tone"], "Concerned")

    def test_parse_sentiment_analysis(self):
        """Test parsing of sentiment analysis responses."""
        # Create an LLMInterface instance
        with patch('news.llm_interface.OpenAI'):
            llm = LLMInterface(api_key="test_api_key")

        # Test case 1: Complete response
        response_text = """
        Sentiment: Positive
        Sentiment Score: 0.75
        Key Themes: Innovation, Growth, Technology
        Emotional Tone: Excited, Optimistic
        """

        # pylint: disable=protected-access
        result = llm._parse_sentiment_analysis(response_text)
        # pylint: enable=protected-access

        self.assertEqual(result["sentiment"], "Positive")
        self.assertEqual(result["score"], 0.75)
        self.assertEqual(result["themes"], ["Innovation", "Growth", "Technology"])
        self.assertEqual(result["tone"], "Excited, Optimistic")

        # Test case 2: Missing values
        response_text = """
        Sentiment: Neutral
        Key Themes: News, Report
        """

        # pylint: disable=protected-access
        result = llm._parse_sentiment_analysis(response_text)
        # pylint: enable=protected-access

        self.assertEqual(result["sentiment"], "Neutral")
        self.assertEqual(result["score"], 0.0)  # Default value
        self.assertEqual(result["themes"], ["News", "Report"])
        self.assertEqual(result["tone"], "")  # Empty string for missing value

        # Test case 3: Invalid sentiment score format
        response_text = """
        Sentiment: Negative
        Sentiment Score: very negative
        Key Themes: Conflict, Crisis
        Emotional Tone: Concerned
        """

        # pylint: disable=protected-access
        result = llm._parse_sentiment_analysis(response_text)
        # pylint: enable=protected-access

        self.assertEqual(result["sentiment"], "Negative")
        self.assertEqual(result["score"], 0.0)  # Default value when parsing fails
        self.assertEqual(result["themes"], ["Conflict", "Crisis"])
        self.assertEqual(result["tone"], "Concerned")

    def test_format_context_edge_cases(self):
        """Test formatting of context documents with edge cases."""
        # Create an LLMInterface instance
        with patch('news.llm_interface.OpenAI'):
            llm = LLMInterface(api_key="test_api_key")

        # Test case 1: Empty document list
        # pylint: disable=protected-access
        formatted = llm._format_context([])
        # pylint: enable=protected-access
        self.assertEqual(formatted, "")

        # Test case 2: Missing fields
        documents = [
            {
                # Missing title and source
                "text": "Content only"
            },
            {
                # Empty values
                "title": "",
                "source": "",
                "text": ""
            }
        ]

        # pylint: disable=protected-access
        formatted = llm._format_context(documents)
        # pylint: enable=protected-access

        self.assertIn("Title: No Title", formatted)  # Default title
        self.assertIn("Source: Unknown", formatted)  # Default source
        self.assertIn("Content: Content only", formatted)
        self.assertIn("Content: ", formatted)  # Empty content

        # Test case 3: Using content_preview vs text field
        documents = [
            {
                "title": "Article with content_preview",
                "content_preview": "Preview text"
            },
            {
                "title": "Article with text field",
                "text": "Full text"
            },
            {
                "title": "Article with both fields",
                "content_preview": "Preview takes precedence",
                "text": "Full text is secondary"
            }
        ]

        # pylint: disable=protected-access
        formatted = llm._format_context(documents)
        # pylint: enable=protected-access

        self.assertIn("Content: Preview text", formatted)
        self.assertIn("Content: Full text", formatted)
        self.assertIn("Content: Preview takes precedence", formatted)
        self.assertNotIn("Full text is secondary", formatted)

if __name__ == '__main__':
    unittest.main()

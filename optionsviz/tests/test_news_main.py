# test_news_main.py
"""
Unit tests for the News Analyzer main module.

These tests verify the command-line interface functionality,
API key handling, and service initialization components
of the main.py module.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import argparse
from io import StringIO

from news.main import (
    parse_arguments,
    check_api_keys,
    fetch_and_process_articles,
    initialize_services,
    interactive_loop,
    main
)

# Add the project directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestParseArguments(unittest.TestCase):
    """Test the command-line argument parsing."""

    @patch('sys.argv', ['main.py', '--query', 'test query'])
    def test_parse_arguments_minimal(self):
        """Test parsing with only required arguments."""
        args = parse_arguments()

        # Check required argument
        self.assertEqual(args.query, 'test query')

        # Check defaults
        self.assertIsNone(args.sources)
        self.assertIsNone(args.from_date)
        self.assertIsNone(args.to_date)
        self.assertEqual(args.language, 'en')
        self.assertIsNone(args.category)
        self.assertFalse(args.skip_fetch)
        self.assertEqual(args.index_name, 'newsdata')
        self.assertEqual(args.model, 'intfloat/e5-large-v2')
        self.assertEqual(args.llm_model, 'deepseek/deepseek-chat:free')

    @patch('sys.argv', [
        'main.py',
        '--query', 'climate change',
        '--sources', 'bbc-news,cnn',
        '--from-date', '2023-10-01',
        '--to-date', '2023-10-15',
        '--language', 'fr',
        '--category', 'science',
        '--skip-fetch',
        '--index-name', 'custom-index',
        '--model', 'custom-model',
        '--llm-model', 'custom-llm'
    ])
    def test_parse_arguments_full(self):
        """Test parsing with all arguments specified."""
        args = parse_arguments()

        # Check all arguments
        self.assertEqual(args.query, 'climate change')
        self.assertEqual(args.sources, 'bbc-news,cnn')
        self.assertEqual(args.from_date, '2023-10-01')
        self.assertEqual(args.to_date, '2023-10-15')
        self.assertEqual(args.language, 'fr')
        self.assertEqual(args.category, 'science')
        self.assertTrue(args.skip_fetch)
        self.assertEqual(args.index_name, 'custom-index')
        self.assertEqual(args.model, 'custom-model')
        self.assertEqual(args.llm_model, 'custom-llm')


class TestAPIKeyHandling(unittest.TestCase):
    """Test API key checking and handling."""

    @patch.dict('os.environ', {
        'PINECONE_API_KEY': 'pinecone_key',
        'OPENROUTER_API_KEY': 'openrouter_key'
    })
    def test_check_api_keys_valid(self):
        """Test API key checking with valid keys."""
        pinecone_key, openrouter_key = check_api_keys()

        self.assertEqual(pinecone_key, 'pinecone_key')
        self.assertEqual(openrouter_key, 'openrouter_key')

    @patch.dict('os.environ', {
        'PINECONE_API_KEY': '',
        'OPENROUTER_API_KEY': 'openrouter_key'
    }, clear=True)
    def test_check_api_keys_missing_pinecone(self):
        """Test error when Pinecone API key is missing."""
        with self.assertRaises(ValueError) as context:
            check_api_keys()

        self.assertIn("PINECONE_API_KEY", str(context.exception))

    @patch.dict('os.environ', {
        'PINECONE_API_KEY': 'pinecone_key',
        'OPENROUTER_API_KEY': ''
    }, clear=True)
    def test_check_api_keys_missing_openrouter(self):
        """Test error when OpenRouter API key is missing."""
        with self.assertRaises(ValueError) as context:
            check_api_keys()

        self.assertIn("OPENROUTER_API_KEY", str(context.exception))


class TestFetchAndProcessArticles(unittest.TestCase):
    """Test fetching and processing news articles."""

    @patch('news.main.check_api_keys')
    @patch('news.main.fetch_news')
    @patch('news.main.process_news_articles_whole')
    @patch('news.main.create_pinecone_index')
    @patch('news.main.clear_pinecone_index')
    @patch('news.main.upload_news_to_pinecone')
    def test_fetch_and_process_articles(self, *mocks):
        """Test the complete fetch and process workflow using grouped arguments."""

        # Group arguments in a dictionary to reduce function parameters
        services = {
            "mock_upload": mocks[0],
            "mock_clear": mocks[1],
            "mock_create": mocks[2],
            "mock_process": mocks[3],
            "mock_fetch": mocks[4],
            "mock_check_keys": mocks[5],
        }

        # Setup mocks
        services["mock_check_keys"].return_value = ('pinecone_key', 'openrouter_key')
        services["mock_fetch"].return_value = [{'title': 'Article 1'}, {'title': 'Article 2'}]
        services["mock_process"].return_value = [MagicMock(), MagicMock()]

        # Create arguments
        args = argparse.Namespace(
            query='test query',
            sources='test-source',
            from_date='2023-10-01',
            to_date='2023-10-15',
            language='en',
            category=None,
            index_name='test-index',
            model='test-model'
        )

        # Call the function
        fetch_and_process_articles(args)

        # Verify all steps were called correctly
        services["mock_check_keys"].assert_called_once()
        services["mock_fetch"].assert_called_once_with(
            query='test query',
            sources='test-source',
            from_date='2023-10-01',
            to_date='2023-10-15',
            language='en',
            category=None
        )
        services["mock_process"].assert_called_once_with(services["mock_fetch"].return_value)
        services["mock_create"].assert_called_once_with(
            pinecone_api_key='pinecone_key',
            index_name='test-index'
        )
        services["mock_clear"].assert_called_once_with('pinecone_key', 'test-index')
        services["mock_upload"].assert_called_once_with(
            document_chunks=services["mock_process"].return_value,
            pinecone_api_key='pinecone_key',
            index_name='test-index',
            model_name='test-model'
        )


    @patch('news.main.check_api_keys')
    @patch('news.main.fetch_news')
    @patch('news.main.create_pinecone_index')
    def test_fetch_and_process_no_articles(
        self, mock_create, mock_fetch, mock_check_keys
        ):
        """Test handling when no articles are found."""
        # Setup mocks
        mock_check_keys.return_value = ('pinecone_key', 'openrouter_key')
        mock_fetch.return_value = []  # No articles found

        # Create arguments
        args = argparse.Namespace(
            query='test query',
            sources=None,
            from_date='2023-10-01',
            to_date='2023-10-15',
            language='en',
            category=None,
            index_name='test-index',
            model='test-model'
        )

        # Redirect stdout to capture output
        stdout = StringIO()
        with patch('sys.stdout', stdout):
            fetch_and_process_articles(args)

        # Verify fetch was called but no further processing happened
        mock_fetch.assert_called_once()
        mock_create.assert_not_called()
        self.assertIn("No articles found", stdout.getvalue())

class TestInitializeServices(unittest.TestCase):
    """Test initialization of retriever and LLM services."""

    @patch('news.main.check_api_keys')
    @patch('news.main.NewsRetriever')
    @patch('news.main.LLMInterface')
    def test_initialize_services_success(self, mock_llm, mock_retriever, mock_check_keys):
        """Test successful initialization of both services."""
        # Setup mocks
        mock_check_keys.return_value = ('pinecone_key', 'openrouter_key')
        mock_retriever_instance = MagicMock()
        mock_retriever.return_value = mock_retriever_instance
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        # Create arguments
        args = argparse.Namespace(
            index_name='test-index',
            model='test-model',
            llm_model='test-llm'
        )

        # Call the function
        retriever, llm = initialize_services(args)

        # Verify correct initialization
        mock_retriever.assert_called_once_with(
            pinecone_api_key='pinecone_key',
            index_name='test-index',
            model_name='test-model'
        )
        mock_llm.assert_called_once_with(
            api_key='openrouter_key',
            model='test-llm'
        )
        self.assertEqual(retriever, mock_retriever_instance)
        self.assertEqual(llm, mock_llm_instance)

    @patch('news.main.check_api_keys')
    @patch('news.main.NewsRetriever')
    @patch('news.main.LLMInterface')
    def test_initialize_services_llm_error(self, mock_llm, mock_retriever, mock_check_keys):
        """Test handling of LLM initialization error."""
        # Setup mocks
        mock_check_keys.return_value = ('pinecone_key', 'openrouter_key')
        mock_retriever_instance = MagicMock()
        mock_retriever.return_value = mock_retriever_instance
        mock_llm.side_effect = Exception("LLM Error")

        # Create arguments
        args = argparse.Namespace(
            index_name='test-index',
            model='test-model',
            llm_model='test-llm'
        )

        # Since the current implementation does not handle the error,
        # we now expect the exception to propagate.
        with self.assertRaises(Exception) as context:
            initialize_services(args)
        self.assertEqual(str(context.exception), "LLM Error")



class TestInteractiveLoop(unittest.TestCase):
    """Test the interactive query loop."""

    @patch('builtins.input')
    def test_interactive_loop_exit(self, mock_input):
        """Test exiting the interactive loop."""
        # Setup mocks
        mock_input.return_value = 'exit'
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        # Redirect stdout to capture output
        stdout = StringIO()
        with patch('sys.stdout', stdout):
            interactive_loop(mock_retriever, mock_llm)

        # Verify input was called but no search was performed
        mock_input.assert_called_once()
        mock_retriever.search.assert_not_called()
        mock_llm.query.assert_not_called()

    @patch('builtins.input')
    @patch('news.main.preprocess_query')
    def test_interactive_loop_test_query(self, mock_preprocess, mock_input):
        """Test the 'test query' functionality."""
        # Setup mocks for the test_query command and then exit
        mock_input.side_effect = ['test query', 'example query', 'exit']
        mock_preprocess.return_value = ('enhanced query', 'example query')
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        # Redirect stdout to capture output
        stdout = StringIO()
        with patch('sys.stdout', stdout):
            interactive_loop(mock_retriever, mock_llm)

        # Verify preprocess_query was called with the test query
        mock_preprocess.assert_called_once_with('example query')
        self.assertIn("Original: 'example query'", stdout.getvalue())
        self.assertIn("Enhanced: 'enhanced query'", stdout.getvalue())

    @patch('builtins.input')
    def test_interactive_loop_normal_query_with_results(self, mock_input):
        """Test a normal query that returns results."""
        # Setup mocks
        mock_input.side_effect = ['what is AI?', 'exit']

        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        # Mock search results
        mock_result = {
            'score': 0.95,
            'title': 'AI Article',
            'source': 'Tech News',
            'published_at': '2023-10-15',
            'content_preview': 'AI is a field of computer science...'
        }
        mock_retriever.search.return_value = [mock_result]

        # Mock LLM response
        mock_llm.query.return_value = "AI stands for Artificial Intelligence..."

        # Redirect stdout to capture output
        stdout = StringIO()
        with patch('sys.stdout', stdout):
            interactive_loop(mock_retriever, mock_llm)

        # Verify search and query were called
        mock_retriever.search.assert_called_once_with('what is AI?', top_k=10)
        mock_llm.query.assert_called_once_with('what is AI?', [mock_result])

        # Check output contains result and answer
        output = stdout.getvalue()
        self.assertIn("Found 1 relevant articles", output)
        self.assertIn("AI Article", output)
        self.assertIn("Tech News", output)
        self.assertIn("===== Answer =====", output)
        self.assertIn("AI stands for Artificial Intelligence", output)

    @patch('builtins.input')
    def test_interactive_loop_no_results(self, mock_input):
        """Test handling when a query returns no results."""

        # Setup mocks
        mock_input.side_effect = ['obscure topic', 'exit']

        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        # Mock empty search results
        mock_retriever.search.return_value = []

        # Redirect stdout to capture output
        stdout = StringIO()
        with patch('sys.stdout', stdout):
            interactive_loop(mock_retriever, mock_llm)

        # Verify search was called but not LLM
        mock_retriever.search.assert_called_once()
        mock_llm.query.assert_not_called()

        # Check output indicates no results
        self.assertIn("No relevant articles found", stdout.getvalue())

    @patch('builtins.input')
    def test_interactive_loop_no_llm(self, mock_input):
        """Test handling when LLM is not available."""
        # Setup mocks
        mock_input.side_effect = ['what is AI?', 'exit']

        mock_retriever = MagicMock()
        mock_llm = None  # LLM not available

        # Mock search results
        mock_result = {
            'score': 0.95,
            'title': 'AI Article',
            'source': 'Tech News',
            'text': 'AI is a field of computer science...'
        }
        mock_retriever.search.return_value = [mock_result]

        # Redirect stdout to capture output
        stdout = StringIO()
        with patch('sys.stdout', stdout):
            interactive_loop(mock_retriever, mock_llm)

        # Verify search was called but not LLM
        mock_retriever.search.assert_called_once()

        # Check output shows article summary instead of LLM response
        output = stdout.getvalue()
        self.assertIn("LLM interface is not available", output)
        self.assertIn("Summary of Article 1", output)
        self.assertIn("AI Article", output)
        self.assertIn("Tech News", output)


class TestMainFunction(unittest.TestCase):
    """Test the main function that orchestrates the entire process."""

    @patch('news.main.parse_arguments')
    @patch('news.main.fetch_and_process_articles')
    @patch('news.main.initialize_services')
    @patch('news.main.interactive_loop')
    def test_main_normal_flow(
        self, mock_interactive_loop, mock_initialize, mock_fetch, mock_parse
    ):
        """Test normal execution flow."""
        # Setup mocks
        args = argparse.Namespace(
            query='test query',
            from_date=None,
            to_date=None,
            skip_fetch=False,
            # Other args...
        )
        mock_parse.return_value = args

        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        mock_initialize.return_value = (mock_retriever, mock_llm)

        # Call main function
        main()

        # Verify the correct sequence of calls
        mock_parse.assert_called_once()
        mock_fetch.assert_called_once_with(args)
        mock_initialize.assert_called_once_with(args)
        mock_interactive_loop.assert_called_once_with(mock_retriever, mock_llm)

        # Verify default dates were set
        self.assertIsNotNone(args.from_date)
        self.assertIsNotNone(args.to_date)

    @patch('news.main.parse_arguments')
    @patch('news.main.fetch_and_process_articles')
    @patch('news.main.initialize_services')
    @patch('news.main.interactive_loop')
    def test_main_skip_fetch(
        self, mock_interactive_loop, mock_initialize, mock_fetch, mock_parse
    ):
        """Test execution with skip_fetch=True."""
        # Setup mocks
        args = argparse.Namespace(
            query='test query',
            from_date=None,
            to_date=None,
            skip_fetch=True,
            # Other args...
        )
        mock_parse.return_value = args

        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        mock_initialize.return_value = (mock_retriever, mock_llm)

        # Call main function
        main()

        # Verify fetch_and_process_articles was not called
        mock_fetch.assert_not_called()
        mock_initialize.assert_called_once()
        mock_interactive_loop.assert_called_once()

    @patch('news.main.parse_arguments')
    @patch('news.main.fetch_and_process_articles')
    def test_main_fetch_error(self, mock_fetch, mock_parse):
        """Test error handling during fetch and process."""
        # Setup mocks
        args = argparse.Namespace(
            query='test query',
            from_date=None,
            to_date=None,
            skip_fetch=False,
            # Other args...
        )
        mock_parse.return_value = args

        # Make fetch_and_process_articles raise an exception
        mock_fetch.side_effect = Exception("Fetch error")

        # Expect the exception to propagate from main()
        with self.assertRaises(Exception) as cm:
            main()
        self.assertEqual(str(cm.exception), "Fetch error")
        mock_fetch.assert_called_once()

    @patch('news.main.parse_arguments')
    @patch('news.main.fetch_and_process_articles')
    @patch('news.main.initialize_services')
    def test_main_initialize_error(self, mock_initialize, mock_fetch, mock_parse):
        """Test error handling during service initialization."""
        # Setup mocks
        args = argparse.Namespace(
            query='test query',
            from_date=None,
            to_date=None,
            skip_fetch=False,
            # Other args...
        )
        mock_parse.return_value = args

        # Make initialize_services raise an exception
        mock_initialize.side_effect = Exception("Initialization error")

        # Expect the exception to propagate from main()
        with self.assertRaises(Exception) as cm:
            main()
        self.assertEqual(str(cm.exception), "Initialization error")
        mock_fetch.assert_called_once()
        mock_initialize.assert_called_once()


if __name__ == '__main__':
    unittest.main()

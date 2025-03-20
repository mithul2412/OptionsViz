# test_news_core.py
"""
Unit tests for the news_core module.

These tests verify the functionality for fetching and processing
news articles with mocking to avoid actual API calls.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

from news.news_core import fetch_news, process_news_articles

# Add the project directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFetchNews(unittest.TestCase):
    """Test suite for the fetch_news function."""

    @patch('news.news_core.NewsApiClient')
    def test_fetch_news_with_category(self, mock_newsapi_client):
        """Test fetching news with a category (using top-headlines endpoint)."""
        # Setup mock
        mock_client = MagicMock()
        mock_newsapi_client.return_value = mock_client

        # Mock response from get_top_headlines
        mock_client.get_top_headlines.return_value = {
            'status': 'ok',
            'totalResults': 2,
            'articles': [
                {
                    'source': {'id': 'bbc-news', 'name': 'BBC News'},
                    'author': 'BBC News',
                    'title': 'Test Article 1',
                    'description': 'Description 1',
                    'url': 'http://example.com/1',
                    'urlToImage': 'http://example.com/image1.jpg',
                    'publishedAt': '2023-10-15T12:00:00Z',
                    'content': 'Content 1'
                },
                {
                    'source': {'id': 'cnn', 'name': 'CNN'},
                    'author': 'CNN Reporter',
                    'title': 'Test Article 2',
                    'description': 'Description 2',
                    'url': 'http://example.com/2',
                    'urlToImage': 'http://example.com/image2.jpg',
                    'publishedAt': '2023-10-15T14:00:00Z',
                    'content': 'Content 2'
                }
            ]
        }

        # Call the function with a category
        with patch.dict('os.environ', {'NEWS_API_KEY': 'fake_key'}):
            articles = fetch_news(
                query="test",
                category="technology",
                language="en"
            )

        # Verify correct endpoint was called
        mock_client.get_top_headlines.assert_called_once()
        mock_client.get_everything.assert_not_called()

        # Check call arguments
        call_args = mock_client.get_top_headlines.call_args[1]
        self.assertEqual(call_args['q'], "test")
        self.assertEqual(call_args['category'], "technology")
        self.assertEqual(call_args['language'], "en")
        self.assertEqual(call_args['page_size'], 100)

        # Check results
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0]['title'], 'Test Article 1')
        self.assertEqual(articles[0]['source']['name'], 'BBC News')
        self.assertEqual(articles[0]['id'], 'news_0')  # Check that ID was added
        self.assertEqual(articles[0]['published_date'], '2023-10-15')  # Check date formatting

    @patch('news.news_core.NewsApiClient')
    def test_fetch_news_without_category(self, mock_newsapi_client):
        """Test fetching news without a category (using everything endpoint)."""
        # Setup mock
        mock_client = MagicMock()
        mock_newsapi_client.return_value = mock_client

        # Mock response from get_everything
        mock_client.get_everything.return_value = {
            'status': 'ok',
            'totalResults': 1,
            'articles': [
                {
                    'source': {'id': 'reuters', 'name': 'Reuters'},
                    'author': 'Reuters Staff',
                    'title': 'Test Article 3',
                    'description': 'Description 3',
                    'url': 'http://example.com/3',
                    'urlToImage': 'http://example.com/image3.jpg',
                    'publishedAt': '2023-10-16T10:00:00Z',
                    'content': 'Content 3'
                }
            ]
        }

        # Call the function without a category, but with date parameters
        with patch.dict('os.environ', {'NEWS_API_KEY': 'fake_key'}):
            articles = fetch_news(
                query="test",
                sources="reuters",
                from_date="2023-10-10",
                to_date="2023-10-17",
                language="en",
                sort_by="relevancy"
            )

        # Verify correct endpoint was called
        mock_client.get_everything.assert_called_once()
        mock_client.get_top_headlines.assert_not_called()

        # Check call arguments
        call_args = mock_client.get_everything.call_args[1]
        self.assertEqual(call_args['q'], "test")
        self.assertEqual(call_args['sources'], "reuters")
        self.assertEqual(call_args['from_param'], "2023-10-10")
        self.assertEqual(call_args['to'], "2023-10-17")
        self.assertEqual(call_args['language'], "en")
        self.assertEqual(call_args['sort_by'], "relevancy")
        self.assertEqual(call_args['page_size'], 100)

        # Check results
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], 'Test Article 3')
        self.assertEqual(articles[0]['id'], 'news_0')

    @patch('news.news_core.NewsApiClient')
    def test_fetch_news_with_default_dates(self, mock_newsapi_client):
        """Test fetching news with default date parameters."""

        # Setup mock
        mock_client = MagicMock()
        mock_newsapi_client.return_value = mock_client

        # Mock response
        mock_client.get_everything.return_value = {'articles': []}

        # Set up current date for date parameter testing
        current_date = datetime.now()
        week_ago = current_date - timedelta(days=7)

        # Call the function with no date parameters
        with patch.dict('os.environ', {'NEWS_API_KEY': 'fake_key'}):
            fetch_news(query="test")

        # Check that default dates were used
        call_args = mock_client.get_everything.call_args[1]
        expected_from = week_ago.strftime('%Y-%m-%d')
        expected_to = current_date.strftime('%Y-%m-%d')

        self.assertEqual(call_args['from_param'], expected_from)
        self.assertEqual(call_args['to'], expected_to)

    def test_fetch_news_no_api_key(self):
        """Test error handling when no API key is available."""
        # Ensure the environment variable is not set
        with patch.dict('os.environ', {'NEWS_API_KEY': ''}, clear=True):
            # Check that ValueError is raised
            with self.assertRaises(ValueError):
                fetch_news("test")

    @patch('news.news_core.NewsApiClient')
    def test_fetch_news_invalid_date_format(self, mock_newsapi_client):
        """Test handling of articles with invalid date formats."""
        # Setup mock
        mock_client = MagicMock()
        mock_newsapi_client.return_value = mock_client

        # Mock response with an invalid date format
        mock_client.get_everything.return_value = {
            'articles': [
                {
                    'source': {'name': 'Test Source'},
                    'title': 'Test Article',
                    'publishedAt': 'invalid-date-format'  # Invalid format
                }
            ]
        }

        # Call the function
        with patch.dict('os.environ', {'NEWS_API_KEY': 'fake_key'}):
            articles = fetch_news(query="test")

        # Check that the article was processed despite the invalid date
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['published_date'], 'invalid-date-format')

class TestProcessNewsArticles(unittest.TestCase):
    """Test suite for the process_news_articles function."""

    def test_process_news_articles_normal(self):
        """Test processing a normal set of news articles."""
        # Sample articles
        articles = [
            {
                'id': 'news_0',
                'title': 'Test Article 1',
                'description': 'Description 1',
                'content': 'Content 1',
                'source': {'name': 'Source 1'},
                'author': 'Author 1',
                'publishedAt': '2023-10-15T12:00:00Z',
                'url': 'http://example.com/1'
            },
            {
                'id': 'news_1',
                'title': 'Test Article 2',
                'description': 'Description 2',
                'content': 'Content 2',
                'source': {'name': 'Source 2'},
                'author': 'Author 2',
                'publishedAt': '2023-10-16T12:00:00Z',
                'url': 'http://example.com/2'
            }
        ]

        # Process the articles
        documents = process_news_articles(articles)

        # Check results
        self.assertEqual(len(documents), 2)

        # Check first document
        self.assertIn('TITLE: Test Article 1', documents[0].page_content)
        self.assertIn('SOURCE: Source 1', documents[0].page_content)
        self.assertIn('AUTHOR: Author 1', documents[0].page_content)
        self.assertIn('CONTENT: Content 1', documents[0].page_content)

        # Check metadata
        self.assertEqual(documents[0].metadata['title'], 'Test Article 1')
        self.assertEqual(documents[0].metadata['source'], 'Source 1')
        self.assertEqual(documents[0].metadata['author'], 'Author 1')
        self.assertEqual(documents[0].metadata['url'], 'http://example.com/1')
        self.assertEqual(documents[0].metadata['article_id'], 'news_0')

    def test_process_news_articles_missing_fields(self):
        """Test processing articles with missing fields."""
        # Sample articles with missing fields
        articles = [
            {
                'id': 'news_0',
                'title': '', # Empty title
                'source': {}, # Empty source
                'publishedAt': '2023-10-15T12:00:00Z'
            }
        ]

        # Process the articles
        documents = process_news_articles(articles)

        # Check results
        self.assertEqual(len(documents), 1)

        # Check that missing fields were handled properly
        self.assertIn('TITLE: ', documents[0].page_content)
        self.assertIn('SOURCE: Unknown', documents[0].page_content)
        self.assertEqual(documents[0].metadata['title'], 'No Title')
        self.assertEqual(documents[0].metadata['source'], 'Unknown')
        self.assertEqual(documents[0].metadata['description'], '')

    def test_process_news_articles_none_values(self):
        """Test processing articles with None values."""
        # Sample articles with None values
        articles = [
            {
                'id': 'news_0',
                'title': 'Test Article',
                'description': None,
                'content': 'Content',
                'source': {'name': 'Source'},
                'author': None,
                'publishedAt': '2023-10-15T12:00:00Z',
                'url': 'http://example.com'
            }
        ]

        # Process the articles
        documents = process_news_articles(articles)

        # Check results
        self.assertEqual(len(documents), 1)

        # Check that None values were handled properly
        self.assertIn('DESCRIPTION: ', documents[0].page_content)
        self.assertIn('AUTHOR: ', documents[0].page_content)
        self.assertEqual(documents[0].metadata['author'], '')  # None replaced with empty string

    def test_process_news_articles_empty_list(self):
        """Test processing an empty list of articles."""
        # Process empty list
        documents = process_news_articles([])

        # Check results
        self.assertEqual(len(documents), 0)


if __name__ == '__main__':
    unittest.main()

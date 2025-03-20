"""
Test module for app_split.py
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import PIL.Image
PIL.Image.open = MagicMock()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_split  # pylint: disable=wrong-import-position


class TestAppSplit(unittest.TestCase):
    """Test class for import app_split.py"""

    def test_app_split_import(self):
        """
        Ensure we didn't fail on FileNotFoundError for news_icon.jpg.
        """
        # We can optionally confirm the placeholder image is set to the mock
        self.assertIsNotNone(app_split.PLACEHOLDER_IMAGE)
        # And if you wish, check the call:
        PIL.Image.open.assert_called_once_with("img/news_icon.jpg")


class TestAppSplitFunctions(unittest.TestCase):
    """Test class for individual functions in app_split.py"""

    @patch('app_split.st')
    def test_format_large_number(self, mock_st):# pylint: disable=unused-argument
        """Test the format_large_number function"""
        # Now that app_split is imported, just call it
        self.assertEqual(app_split.format_large_number(1_500_000_000), "1.50B")
        self.assertEqual(app_split.format_large_number(2_700_000), "2.70M")
        self.assertEqual(app_split.format_large_number(5_400), "5.40K")
        self.assertEqual(app_split.format_large_number(987), "987")
        self.assertEqual(app_split.format_large_number(1_234_567_890), "1.23B")

    @patch('app_split.st')
    def test_initialize_session_state(self, mock_st):
        """Test the initialize_session_state function"""
        # Mock session_state as a dictionary
        mock_st.session_state = {}
        mock_pinecone_api_key = "mock_pinecone_key"
        mock_openrouter_api_key = "mock_openrouter_key"
        mock_newsapi_key = "mock_news_key"

        # Patch the module-level variables in app_split
        with patch.object(app_split, 'PINECONE_API_KEY', mock_pinecone_api_key), \
             patch.object(app_split, 'OPENROUTER_API_KEY', mock_openrouter_api_key), \
             patch.object(app_split, 'NEWSAPI_KEY', mock_newsapi_key):

            app_split.initialize_session_state()

            # Assertions
            self.assertTrue(mock_st.session_state.get('initialized', False))
            self.assertEqual(mock_st.session_state.get('current_app'), "options")
            self.assertEqual(mock_st.session_state.get('options_ticker'), "AAPL")
            self.assertEqual(mock_st.session_state.get('PINECONE_API_KEY'), mock_pinecone_api_key)
            self.assertEqual(mock_st.session_state.get('OPENROUTER_API_KEY'),
                             mock_openrouter_api_key)
            self.assertEqual(mock_st.session_state.get('NEWSAPI_KEY'), mock_newsapi_key)
            self.assertIn('fetched_articles', mock_st.session_state)
            self.assertIn('current_page', mock_st.session_state)
            self.assertIn('chat_history', mock_st.session_state)
            self.assertIn('options_data', mock_st.session_state)
            self.assertIn('options_summary', mock_st.session_state)
            self.assertIn('extracted_orders', mock_st.session_state)

    @patch('app_split.st')
    def test_render_custom_css(self, mock_st):
        """Test the render_custom_css function"""
        app_split.render_custom_css()
        mock_st.markdown.assert_called_once()
        call_args = mock_st.markdown.call_args[0][0]
        kwargs = mock_st.markdown.call_args[1]
        self.assertIsInstance(call_args, str)
        self.assertIn("<style>", call_args)
        self.assertIn("</style>", call_args)
        self.assertEqual(kwargs, {"unsafe_allow_html": True})



# pylint: disable=too-many-arguments,too-many-positional-arguments
# Necessary for test methods that mock multiple dependencies
@patch('app_split.st')
@patch('app_split.render_options_app')
@patch('app_split.render_news_app')
@patch('app_split.render_sidebar', return_value=(None, None))
@patch('app_split.render_custom_css')
@patch('app_split.initialize_session_state')
class TestAppSplitMain(unittest.TestCase):
    """Test class for the main function in app_split.py"""

    def test_main_options_app(self, mock_init, mock_css, mock_sidebar,
                              mock_news_app, mock_options_app, mock_st):
        """Test main function when options app is selected"""
        mock_st.session_state = {'current_app': 'options'}
        app_split.main()
        mock_init.assert_called_once()
        mock_css.assert_called_once()
        mock_sidebar.assert_called_once()
        mock_options_app.assert_called_once()
        mock_news_app.assert_not_called()

    def test_main_news_app(self, mock_init, mock_css, mock_sidebar,
                           mock_news_app, mock_options_app, mock_st):
        """Test main function when news app is selected"""
        mock_st.session_state = {'current_app': 'news'}
        mock_sidebar.return_value = (MagicMock(), MagicMock())
        app_split.main()
        mock_init.assert_called_once()
        mock_css.assert_called_once()
        mock_sidebar.assert_called_once()
        mock_options_app.assert_not_called()
        mock_news_app.assert_called_once()


@patch('app_split.st')
@patch('app_split.render_options_sidebar')
@patch('app_split.render_news_sidebar', return_value=(MagicMock(), MagicMock()))
class TestSidebarRendering(unittest.TestCase):
    """Test sidebar rendering functionality"""
    def test_render_sidebar_options(self, mock_news_sidebar, mock_options_sidebar, mock_st):
        """Test sidebar rendering options"""
        mock_st.session_state = {'current_app': 'options'}
        result = app_split.render_sidebar()
        mock_options_sidebar.assert_called_once()
        mock_news_sidebar.assert_not_called()
        self.assertEqual(result, (None, None))

    def test_render_sidebar_news(self, mock_news_sidebar, mock_options_sidebar, mock_st):
        """Test sidebar rendering news"""
        mock_st.session_state = {'current_app': 'news'}
        pinecone_key = "mock_pinecone"
        openrouter_key = "mock_openrouter"
        newsapi_key = "mock_newsapi"
        with patch.object(app_split, 'PINECONE_API_KEY', pinecone_key), \
             patch.object(app_split, 'OPENROUTER_API_KEY', openrouter_key), \
             patch.object(app_split, 'NEWSAPI_KEY', newsapi_key):
            result = app_split.render_sidebar()
            mock_options_sidebar.assert_not_called()
            mock_news_sidebar.assert_called_once_with(pinecone_key, openrouter_key, newsapi_key)
            self.assertEqual(result, mock_news_sidebar.return_value)


@patch('app_split.st')
@patch('app_split.load_dotenv')
@patch('app_split.render_news_app')
@patch('app_split.render_options_app')
@patch('app_split.render_news_sidebar')
@patch('app_split.render_options_sidebar')
@patch('app_split.Image')
class TestAppIntegration(unittest.TestCase):
    """Test app integration between modules"""
    def test_app_initialization_and_navigation(self, mock_image, mock_options_sidebar,
                                               mock_news_sidebar, mock_options_app,
                                               mock_news_app, mock_load_dotenv, mock_st):# pylint: disable=unused-argument
        """Test streamlit app working with options and news apps"""

        mock_st.session_state = {}
        mock_image.open.return_value = MagicMock()

        with patch.object(app_split, 'PINECONE_API_KEY', "test_pinecone_key"), \
             patch.object(app_split, 'OPENROUTER_API_KEY', "test_openrouter_key"), \
             patch.object(app_split, 'NEWSAPI_KEY', "test_newsapi_key"):
            mock_news_sidebar.return_value = (MagicMock(), MagicMock())

            # First run => 'options'
            app_split.main()
            self.assertEqual(mock_st.session_state['current_app'], "options")
            mock_options_sidebar.assert_called_once()
            mock_options_app.assert_called_once()
            mock_news_sidebar.assert_not_called()
            mock_news_app.assert_not_called()

            # Clear
            mock_options_sidebar.reset_mock()
            mock_options_app.reset_mock()

            # Switch to news
            mock_st.session_state['current_app'] = "news"
            app_split.main()
            mock_options_sidebar.assert_not_called()
            mock_options_app.assert_not_called()
            mock_news_sidebar.assert_called_once()
            mock_news_app.assert_called_once()


if __name__ == '__main__':
    unittest.main()

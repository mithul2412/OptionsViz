"""
News Analysis package.

This package provides tools for retrieving, analyzing, and querying news articles, including:
- Fetching news from NewsAPI
- Processing and embedding news articles
- Storing articles in a vector database (Pinecone)
- Querying news with natural language using AI
"""

from news.news_core import (
    fetch_news,
    process_news_articles_whole
)

from news.embedding_utils import (
    SentenceTransformerEmbeddings,
    preprocess_query,
    create_pinecone_index,
    clear_pinecone_index,
    upload_news_to_pinecone,
    NewsRetriever
)

from news.llm_interface import (
    LLMInterface
)

__all__ = [
    'fetch_news',
    'process_news_articles_whole',
    'SentenceTransformerEmbeddings',
    'preprocess_query',
    'create_pinecone_index',
    'clear_pinecone_index',
    'upload_news_to_pinecone',
    'NewsRetriever',
    'LLMInterface'
]

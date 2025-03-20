# news_core.py
"""
Core module for fetching and processing news articles.

This module provides functionality for:
1. Fetching news articles from NewsAPI
2. Processing articles into a format suitable for embedding
3. Preparing article structures for vector database storage

The module focuses on whole-article processing, treating each article
as a single document rather than chunking it into smaller pieces.
"""

from newsapi import NewsApiClient
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document

load_dotenv()


def fetch_news(
    query: Optional[str] = None,
    sources: Optional[str] = None,
    domains: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    language: str = 'en',
    sort_by: str = 'publishedAt',
    category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch news articles from NewsAPI based on search criteria.
    
    This function provides access to both the 'everything' and 'top-headlines'
    endpoints of NewsAPI, depending on whether a category is specified.
    
    Args:
        query: Keywords or phrases to search for (e.g., "climate change")
        sources: Comma-separated string of news sources or blogs (e.g., "bbc-news,cnn")
        domains: Comma-separated string of domains (e.g., "bbc.co.uk,techcrunch.com")
        from_date: Start date in YYYY-MM-DD format (defaults to 7 days ago)
        to_date: End date in YYYY-MM-DD format (defaults to today)
        language: The 2-letter ISO-639-1 code of the language (default: 'en')
        sort_by: The order to sort articles ("relevancy", "popularity", "publishedAt")
        category: Category for top headlines (business, entertainment, health, etc.)
        
    Returns:
        List of news articles with their metadata
        
    Raises:
        ValueError: If NEWS_API_KEY environment variable is not set
        
    Example:
        >>> articles = fetch_news(
        ...     query="climate change",
        ...     sources="bbc-news,the-guardian",
        ...     from_date="2023-10-01",
        ...     to_date="2023-10-15"
        ... )
        >>> len(articles)
        42
    """
    # Initialize NewsAPI client
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        raise ValueError("NEWS_API_KEY environment variable not set")
    
    newsapi = NewsApiClient(api_key=api_key)
    
    # Set default dates if not provided
    if from_date is None:
        # Default to 7 days ago
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    if to_date is None:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch articles
    if category:
        # Use top headlines for category-based queries
        response = newsapi.get_top_headlines(
            q=query,
            sources=sources,
            category=category,
            language=language,
            page_size=100  # Fetch more articles at once
        )
        articles = response.get('articles', [])
    else:
        # Use everything endpoint for more flexible queries
        response = newsapi.get_everything(
            q=query,
            sources=sources,
            domains=domains,
            from_param=from_date,
            to=to_date,
            language=language,
            sort_by=sort_by,
            page_size=100  # Fetch more articles at once
        )
        articles = response.get('articles', [])
    
    # Add unique IDs and improve metadata
    for i, article in enumerate(articles):
        # Add a unique ID for each article
        article['id'] = f"news_{i}"
        
        # Convert publishedAt to proper datetime if needed
        if 'publishedAt' in article:
            try:
                pub_date = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                article['published_date'] = pub_date.strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                # Handle parsing errors gracefully
                article['published_date'] = article['publishedAt']
    
    print(f"Fetched {len(articles)} news articles")
    return articles


def process_news_articles(articles: List[Dict[str, Any]]) -> List[Document]:
    """
    Process news articles into Document objects for vector storage.
    
    This function treats each article as a whole document (rather than 
    chunking it), which is typically more appropriate for news articles
    that aren't excessively long. The resulting Document objects include
    both the article text and comprehensive metadata.
    
    Args:
        articles: List of news articles from the fetch_news function
        
    Returns:
        List of Document objects, one per article
        
    Example:
        >>> articles = fetch_news(query="technology")
        >>> documents = process_news_articles(articles)
        >>> print(documents[0].metadata['title'])
        'Latest Technology Breakthrough Announced'
    """
    documents = []
    
    for article in articles:
        # Extract the content from the article
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        source_name = article.get('source', {}).get('name', 'Unknown')
        author = article.get('author', '')
        published_at = article.get('publishedAt', '')
        
        # Combine the text fields into a structured document
        # Format helps the embedding model understand the document structure
        full_content = f"""
TITLE: {title}

SOURCE: {source_name}

AUTHOR: {author}

PUBLISHED: {published_at}

DESCRIPTION: {description}

CONTENT: {content}
"""
        
        # Comprehensive metadata to help with filtering and display
        metadata = {
            'source': source_name,
            'author': author if author is not None else "",
            'published_at': published_at,
            'url': article.get('url', ''),
            'title': title if title else "No Title",
            'description': description if description else "",
            'document_type': 'news_article',
            'article_id': article.get('id', f"news_{len(documents)}")
        }
        
        documents.append(Document(page_content=full_content, metadata=metadata))
    
    print(f"Created {len(documents)} documents from {len(articles)} news articles")
    return documents


# Alias for backward compatibility
process_news_articles_whole = process_news_articles


# Example usage if running as script
if __name__ == "__main__":
    # Example usage
    articles = fetch_news(
        query="artificial intelligence",
        sources="techcrunch,wired,the-verge",
        from_date="2025-02-20",
        to_date="2025-02-27",
        language="en",
        sort_by="publishedAt"
    )
    
    # Process articles as whole documents
    docs = process_news_articles(articles)
    
    # Print sample article
    if docs:
        print("\nSample article:")
        print(f"Title: {docs[0].metadata['title']}")
        print(f"Content preview: {docs[0].page_content[:200]}...")
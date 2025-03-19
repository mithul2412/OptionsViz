# main.py
"""
Main orchestration script for the News Analyzer CLI tool.

This module provides a command-line interface for the News Analyzer system,
allowing users to:
1. Fetch news articles on specific topics
2. Process and store articles in a vector database
3. Query the articles using natural language
4. Analyze the news with AI-powered insights

The module uses argparse for command-line argument parsing and connects
all the components of the News Analyzer system.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Import from refactored modules
from news.news_core import fetch_news, process_news_articles_whole
from news.embedding_utils import (
    create_pinecone_index, 
    clear_pinecone_index, 
    upload_news_to_pinecone,
    preprocess_query,
    NewsRetriever
)
from news.llm_interface import LLMInterface

# Load environment variables
load_dotenv()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the News Analyzer.
    
    Returns:
        Namespace containing the parsed arguments
        
    Example:
        >>> args = parse_arguments()
        >>> print(args.query)
        'climate change'
    """
    parser = argparse.ArgumentParser(description='News Analyzer with RAG System')
    
    # Main operation parameters
    parser.add_argument('--query', type=str, required=True,
                       help='Search query for news articles')
    parser.add_argument('--sources', type=str, default=None,
                       help='Comma-separated list of news sources')
    parser.add_argument('--from-date', type=str, default=None,
                       help='Start date for article search (YYYY-MM-DD)')
    parser.add_argument('--to-date', type=str, default=None,
                       help='End date for article search (YYYY-MM-DD)')
    parser.add_argument('--language', type=str, default='en',
                       help='Language of news articles')
    parser.add_argument('--category', type=str, default=None,
                       help='News category (for top headlines)')
    parser.add_argument('--skip-fetch', action='store_true',
                       help='Skip fetching new articles and use existing database')
    parser.add_argument('--index-name', type=str, default='newsdata',
                       help='Name of the Pinecone index')
    parser.add_argument('--model', type=str, default='intfloat/e5-large-v2',
                       help='Embedding model name')
    parser.add_argument('--llm-model', type=str, default='deepseek/deepseek-chat:free',
                       help='LLM model to use via OpenRouter')
    
    return parser.parse_args()


def check_api_keys() -> Tuple[str, str]:
    """
    Check if required API keys are set in environment variables.
    
    Returns:
        Tuple of (pinecone_api_key, openrouter_api_key)
        
    Raises:
        ValueError: If required API keys are not set
    """
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    return pinecone_api_key, openrouter_api_key


def fetch_and_process_articles(args: argparse.Namespace) -> None:
    """
    Fetch news articles based on arguments and store them in the vector database.
    
    Args:
        args: Command-line arguments
        
    Returns:
        None
    """
    pinecone_api_key, _ = check_api_keys()
    
    print(f"\n===== Fetching news for query: '{args.query}' =====")
    print(f"Date range: {args.from_date} to {args.to_date}")
    
    # Fetch news articles
    articles = fetch_news(
        query=args.query,
        sources=args.sources,
        from_date=args.from_date,
        to_date=args.to_date,
        language=args.language,
        category=args.category
    )
    
    if not articles:
        print("No articles found.")
        return
    
    # Process articles as whole documents
    chunks = process_news_articles_whole(articles)
    
    # Make sure index exists
    create_pinecone_index(
        pinecone_api_key=pinecone_api_key,
        index_name=args.index_name
    )
    
    # Clear existing vectors before adding new ones
    print(f"Clearing existing vectors from index: {args.index_name}")
    clear_pinecone_index(pinecone_api_key, args.index_name)
    
    # Upload chunks to Pinecone
    upload_news_to_pinecone(
        document_chunks=chunks,
        pinecone_api_key=pinecone_api_key,
        index_name=args.index_name,
        model_name=args.model
    )


def initialize_services(args: argparse.Namespace) -> Tuple[NewsRetriever, Optional[LLMInterface]]:
    """
    Initialize the retriever and LLM interface based on arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple of (news_retriever, llm_interface)
        
    Notes:
        LLM interface may be None if initialization fails
    """
    pinecone_api_key, openrouter_api_key = check_api_keys()
    
    # Initialize retriever
    retriever = NewsRetriever(
        pinecone_api_key=pinecone_api_key,
        index_name=args.index_name,
        model_name=args.model
    )
    
    # Initialize LLM interface
    try:
        llm = LLMInterface(
            api_key=openrouter_api_key,
            model=args.llm_model
        )
    except Exception as e:
        print(f"Warning: Error initializing LLM interface: {str(e)}")
        print("You can still retrieve articles, but LLM responses may not work.")
        llm = None
    
    return retriever, llm


def interactive_loop(retriever: NewsRetriever, llm: Optional[LLMInterface]) -> None:
    """
    Run an interactive query loop for the News Analyzer.
    
    This function allows users to ask questions about the news articles
    and get AI-powered answers, or to test query preprocessing.
    
    Args:
        retriever: NewsRetriever instance for searching articles
        llm: LLMInterface instance for AI-powered answers
        
    Returns:
        None
    """
    print("\n===== News Analysis System Ready =====")
    print("Type 'exit' to quit the program")
    print("Type 'test query' to see how query preprocessing works")
    
    while True:
        # Get user question
        user_question = input("\nEnter your question about the news: ")
        
        if user_question.lower() in ['exit', 'quit', 'q']:
            break
            
        if user_question.lower() == 'test query':
            test_query = input("Enter a query to test preprocessing: ")
            enhanced, original = preprocess_query(test_query)
            print(f"Original: '{original}'")
            print(f"Enhanced: '{enhanced}'")
            continue
        
        # Retrieve relevant articles
        print(f"Searching for relevant articles...")
        search_results = retriever.search(user_question, top_k=10)
        
        # Display retrieved articles
        if search_results:
            print(f"\nFound {len(search_results)} relevant articles:")
            
            for i, result in enumerate(search_results):
                print(f"\n--- Article {i+1} (Relevance: {result['score']:.4f}) ---")
                print(f"Title: {result['title']}")
                print(f"Source: {result['source']}")
                print(f"Published: {result.get('published_at', 'Unknown date')}")
        else:
            print("No relevant articles found.")
            continue
        
        # Query LLM with retrieved context
        print("\nGenerating answer based on retrieved articles...")
        if llm is not None:
            llm_response = llm.query(user_question, search_results)
            
            # Display LLM response
            print("\n===== Answer =====")
            print(llm_response)
        else:
            print("\nLLM interface is not available. Here's a summary of the articles instead:")
            for i, result in enumerate(search_results[:3]):
                print(f"\n--- Summary of Article {i+1} ---")
                print(f"Title: {result['title']}")
                print(f"Source: {result['source']}")
                content = result.get('content_preview', result.get('text', ''))
                if content:
                    print(f"Content: {content[:300]}...")
            
            print("\nTo get LLM-generated answers, please check your OpenRouter API key and internet connection.")


def main():
    """
    Main function to run the news analyzer with RAG.
    
    This function:
    1. Parses command-line arguments
    2. Checks for required API keys
    3. Handles default date parameters
    4. Fetches and processes news articles (if not skipped)
    5. Initializes the retriever and LLM interface
    6. Runs the interactive query loop
    
    Returns:
        None
    """
    args = parse_arguments()
    
    # Set default dates if not provided
    if args.from_date is None:
        args.from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    if args.to_date is None:
        args.to_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch and process news if not skipped
    if not args.skip_fetch:
        try:
            fetch_and_process_articles(args)
        except Exception as e:
            print(f"Error fetching and processing articles: {str(e)}")
            sys.exit(1)
    
    # Initialize retriever and LLM interface
    try:
        retriever, llm = initialize_services(args)
    except Exception as e:
        print(f"Error initializing services: {str(e)}")
        sys.exit(1)
    
    # Run interactive loop
    interactive_loop(retriever, llm)


if __name__ == "__main__":
    main()

# llm_interface.py
"""
Interface for interacting with Language Models via OpenRouter.

This module provides a clean interface for querying LLMs (Language Models)
through the OpenRouter API, which gives access to various models from
different providers. The main class handles:

1. API configuration and authentication
2. Formatting messages with proper context
3. Making API requests and handling responses
4. Error handling and fallbacks

The primary use case is for question answering with optional context from
retrieved documents.
"""

from openai import OpenAI
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class LLMInterface:
    """
    Interface for interacting with Language Models through OpenRouter.
    
    This class provides a consistent interface to query language models
    by handling API configuration, authentication, and request formatting.
    It supports context-augmented queries for Retrieval-Augmented Generation.
    
    Attributes:
        client: OpenAI client configured for OpenRouter
        model: Model identifier to use for queries
        extra_headers: Additional headers required by OpenRouter
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek/deepseek-chat:free",
        site_url: str = "http://localhost",
        site_name: str = "NewsAnalyzer"
    ):
        """
        Initialize the LLM interface with API credentials and model selection.
        
        Args:
            api_key: OpenRouter API key (defaults to environment variable)
            model: Model identifier to use on OpenRouter
                (e.g., "anthropic/claude-3-sonnet", "google/gemini-pro")
            site_url: Site URL for OpenRouter attribution and analytics
            site_name: Site name for OpenRouter attribution and analytics
            
        Raises:
            ValueError: If no API key can be found
            
        Example:
            >>> llm = LLMInterface(model="anthropic/claude-3-haiku")
        """
        # Get API key from env var if not provided
        if api_key is None:
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        self.model = model
        self.extra_headers = {
            "HTTP-Referer": site_url,
            "X-Title": site_name
        }
        
        print(f"Initialized LLM interface with model: {model}")
    
    def query(
        self, 
        user_question: str, 
        context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Query the LLM with a user question and optional context.
        
        This method:
        1. Formats the message with system instructions and context
        2. Calls the OpenRouter API with the formatted message
        3. Returns the model's response text
        
        Args:
            user_question: The user's question or prompt
            context: Optional list of documents to provide as context
                Each document should have 'title', 'source', and 'text' keys
                
        Returns:
            The LLM's response as a string
            
        Raises:
            Exception: If there's an error calling the API
            
        Example:
            >>> context = [{"title": "News Article", "source": "CNN", "text": "..."}]
            >>> response = llm.query("What happened yesterday?", context)
        """
        # Prepare messages
        messages = []
        
        # Add context if provided
        if context and len(context) > 0:
            # Create system message with context
            context_text = self._format_context(context)
            system_message = f"""You are a news analysis assistant that answers questions based on the latest news articles.
            
Use the following news articles as context for answering the user's question:

{context_text}

Answer the user's question based on the information in these articles. If the information needed is not in the articles, say so and provide a general response based on your knowledge. Always cite the source of information when possible."""
            
            messages.append({"role": "system", "content": system_message})
        else:
            # No context, just use a simple system message
            messages.append({
                "role": "system", 
                "content": "You are a news analysis assistant that answers questions based on the latest information."
            })
        
        # Add user question
        messages.append({"role": "user", "content": user_question})
        
        try:
            # Call the API
            completion = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                model=self.model,
                messages=messages
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenRouter API: {str(e)}")
            return f"Sorry, I encountered an error when trying to generate a response: {str(e)}\n\nPlease check your OpenRouter API key and internet connection."
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a string for context.
        
        Creates a formatted string with article details that can be included
        in the system prompt to provide context for the LLM.
        
        Args:
            documents: List of retrieved documents with metadata
                Each document should have keys for title, source, etc.
                
        Returns:
            Formatted context string with article details
            
        Example:
            >>> context = [{"title": "News Article", "source": "CNN", "text": "..."}]
            >>> formatted = llm._format_context(context)
        """
        context_parts = []
        
        for i, doc in enumerate(documents):
            source = doc.get("source", "Unknown")
            title = doc.get("title", "No Title")
            content = doc.get("content_preview", doc.get("text", ""))
            url = doc.get("url", "")
            date = doc.get("published_at", "")
            
            context_parts.append(f"ARTICLE {i+1}:")
            context_parts.append(f"Title: {title}")
            context_parts.append(f"Source: {source}")
            if date:
                context_parts.append(f"Published: {date}")
            context_parts.append(f"Content: {content}")
            if url:
                context_parts.append(f"URL: {url}")
            context_parts.append("")  # Empty line between articles
        
        return "\n".join(context_parts)


# Example usage if running as script
if __name__ == "__main__":
    # Example usage
    llm = LLMInterface()
    
    # Example with mock context
    mock_context = [
        {
            "title": "AI Breakthrough in Medical Imaging",
            "source": "Tech Daily",
            "published_at": "2025-02-25",
            "content_preview": "A new AI algorithm has shown unprecedented accuracy in detecting early-stage cancers from medical scans, potentially revolutionizing cancer screening procedures.",
            "url": "https://example.com/article1"
        },
        {
            "title": "Global Healthcare Initiative Launches AI Partnership",
            "source": "Health News",
            "published_at": "2025-02-26",
            "content_preview": "The World Health Organization has partnered with leading AI research labs to develop accessible diagnostic tools for underserved regions.",
            "url": "https://example.com/article2"
        }
    ]
    
    response = llm.query("How is AI changing healthcare?", mock_context)
    print("\nResponse with context:")
    print(response)

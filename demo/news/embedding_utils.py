# embedding_utils.py
"""
Embedding and vector database utilities for news analysis.

This module provides functionality for:
1. Generating embeddings for documents and queries
2. Preprocessing queries for better semantic search
3. Managing Pinecone vector database operations
4. Retrieving relevant documents based on semantic similarity

The module is organized into component sections: embedding, query processing,
vector database management, and retrieval.
"""

import os
import sys
import time
import spacy
from typing import List, Dict, Any, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Load spaCy model for NLP tasks
try:
    nlp = spacy.load("en_core_web_md")
except (ImportError, OSError):
    import subprocess
    subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load("en_core_web_md")

#############################################
# Embedding Component
#############################################

class SentenceTransformerEmbeddings:
    """
    Custom embeddings class that uses sentence-transformers models.
    
    This class provides a consistent interface for generating text embeddings
    using SentenceTransformer models, with methods for both individual queries
    and batches of documents.
    
    Attributes:
        model: The underlying SentenceTransformer model
    """
    
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        """
        Initialize with a specific SentenceTransformer model.
        
        Args:
            model_name: Name of the model to load from HuggingFace
                Default is "intfloat/e5-large-v2" which performs well for
                retrieval tasks.
                
        Example:
            >>> embeddings = SentenceTransformerEmbeddings("intfloat/e5-base-v2")
        """
        self.model = SentenceTransformer(model_name)
        print(f"Initialized embedding model: {model_name}")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors (as lists of floats)
            
        Example:
            >>> docs = ["First document", "Second document"]
            >>> embeddings = model.embed_documents(docs)
            >>> len(embeddings)
            2
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: The query text to embed
            
        Returns:
            Embedding vector as a list of floats
            
        Example:
            >>> query = "What is artificial intelligence?"
            >>> embedding = model.embed_query(query)
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

#############################################
# Query Processing Component
#############################################

def preprocess_query(query: str) -> Tuple[str, str]:
    """
    Enhance query for better semantic search results using NLP techniques.
    
    This function applies several NLP techniques to extract the most important
    elements from a user query:
    1. Named entity recognition to identify people, organizations, etc.
    2. Part-of-speech tagging to find key nouns, verbs, and adjectives
    3. Stopword removal to focus on meaningful terms
    
    Args:
        query: Original user query
        
    Returns:
        Tuple of (enhanced_query, original_query) where enhanced_query
        focuses on the most important semantic elements
        
    Example:
        >>> enhanced, original = preprocess_query("What did Apple announce yesterday?")
        >>> print(enhanced)
        'Apple announce yesterday'
    """
    original_query = query.strip()
    
    # Parse the query with spaCy
    doc = nlp(original_query)
    
    # Extract named entities
    entities = [ent.text for ent in doc.ents]
    
    # Extract important parts of speech (nouns, proper nouns, verbs, adjectives)
    important_tokens = []
    for token in doc:
        # Keep named entities, nouns, proper nouns, verbs (except auxiliaries), and adjectives
        if (token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and not token.is_stop) or \
           (token.ent_type_ != ""):  # Token is part of a named entity
            # Add the token text
            important_tokens.append(token.text)
    
    # Build the enhanced query
    if entities:
        # If we found named entities, prioritize them
        entity_text = " ".join(entities)
        keywords_text = " ".join([t for t in important_tokens if t not in " ".join(entities)])
        enhanced_query = f"{entity_text} {keywords_text}".strip()
    else:
        # If no entities, use important tokens
        enhanced_query = " ".join(important_tokens)
    
    # If enhanced query is too short, fall back to original
    if len(enhanced_query.split()) < 2:
        enhanced_query = original_query
    
    print(f"Query processing: '{original_query}' -> '{enhanced_query}'")
    return enhanced_query, original_query

#############################################
# Vector Database Component
#############################################

def create_pinecone_index(
    pinecone_api_key: str,
    index_name: str = "newsdata",
    dimension: int = 1024  # E5-Large embedding dimension
) -> str:
    """
    Create a Pinecone vector index for storing document embeddings.
    
    If the index already exists, this function will use the existing one.
    
    Args:
        pinecone_api_key: Your Pinecone API key
        index_name: Name of the Pinecone index to create or use
        dimension: Dimension of the embeddings (depends on model)
        
    Returns:
        Status message indicating if index was created or already exists
        
    Example:
        >>> api_key = os.getenv('PINECONE_API_KEY')
        >>> status = create_pinecone_index(api_key, "news-embeddings")
        >>> print(status)
        'Created Pinecone index: news-embeddings'
    """
    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists, if not create it
    if index_name not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {index_name}")
        
        # Create the index
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
        return f"Created Pinecone index: {index_name}"
    else:
        return f"Using existing Pinecone index: {index_name}"

def clear_pinecone_index(pinecone_api_key: str, index_name: str) -> str:
    """
    Delete all vectors from a Pinecone index to start fresh.
    
    This function is useful when you want to rebuild your vector database
    with new documents without creating a new index.
    
    Args:
        pinecone_api_key: Your Pinecone API key
        index_name: Name of the Pinecone index to clear
        
    Returns:
        Status message confirming the index was cleared
        
    Example:
        >>> api_key = os.getenv('PINECONE_API_KEY')
        >>> clear_pinecone_index(api_key, "news-embeddings")
        'Cleared all vectors from Pinecone index: news-embeddings'
    """
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    
    # Delete all vectors
    index.delete(delete_all=True)
    
    # Give Pinecone some time to process the deletion
    time.sleep(2)
    return f"Cleared all vectors from Pinecone index: {index_name}"

def upload_news_to_pinecone(
    document_chunks: List[Document],
    pinecone_api_key: str,
    index_name: str = "newsdata",
    batch_size: int = 100,
    model_name: str = "intfloat/e5-large-v2"
) -> str:
    """
    Embed and upload news documents to Pinecone vector database.
    
    This function:
    1. Generates embeddings for each document
    2. Cleans metadata to ensure compatibility with Pinecone
    3. Uploads documents in batches to avoid API limits
    
    Args:
        document_chunks: List of document chunks to embed and upload
        pinecone_api_key: Your Pinecone API key
        index_name: Name of the Pinecone index
        batch_size: Number of documents to process in each batch
        model_name: Name of the embedding model to use
        
    Returns:
        Status message with the number of chunks uploaded
        
    Example:
        >>> docs = process_news_articles(articles)
        >>> status = upload_news_to_pinecone(docs, api_key)
        >>> print(status)
        'Successfully uploaded 42 chunks to Pinecone'
    """
    # Initialize embedding model
    embedding_model = SentenceTransformerEmbeddings(model_name=model_name)
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    
    # Process documents in batches
    for i in range(0, len(document_chunks), batch_size):
        batch = document_chunks[i:i+batch_size]
        
        # Extract text and metadata
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        ids = [f"news_chunk_{i+j}" for j in range(len(batch))]
        
        # Get embeddings
        embeddings = embedding_model.embed_documents(texts)
        
        # Create records for Pinecone
        records = []
        for j, (doc_id, embedding, metadata) in enumerate(zip(ids, embeddings, metadatas)):
            # Clean metadata to handle null values
            cleaned_metadata = {}
            for key, value in metadata.items():
                # Skip null values
                if value is None:
                    cleaned_metadata[key] = ""  # Replace None with empty string
                # Handle lists that might contain None
                elif isinstance(value, list):
                    cleaned_metadata[key] = [item if item is not None else "" for item in value]
                # Keep other valid values
                else:
                    cleaned_metadata[key] = value
            
            # Add the text content to metadata for retrieval
            cleaned_metadata["text"] = texts[j][:1000]  # Store truncated text in metadata
            
            records.append({
                "id": doc_id,
                "values": embedding,
                "metadata": cleaned_metadata
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=records)
        
        print(f"Uploaded batch {i//batch_size + 1}, total chunks so far: {min(i+batch_size, len(document_chunks))}")
    
    return f"Successfully uploaded {len(document_chunks)} chunks to Pinecone"

#############################################
# Retriever Component
#############################################

class NewsRetriever:
    """
    Retriever for finding relevant news articles via vector similarity search.
    
    This class handles querying the vector database to find the most relevant
    articles to a user's question, with additional filtering capabilities
    by date, source, etc.
    
    Attributes:
        model: The SentenceTransformer model for embedding queries
        pc: Pinecone client
        index: Pinecone index for the news articles
        index_name: Name of the Pinecone index
    """
    
    def __init__(
        self, 
        pinecone_api_key: Optional[str] = None,
        index_name: str = "newsdata",
        model_name: str = "intfloat/e5-large-v2"
    ):
        """
        Initialize the news retriever with API keys and model selection.
        
        Args:
            pinecone_api_key: Your Pinecone API key (defaults to environment variable)
            index_name: Name of the Pinecone index to query
            model_name: Name of the embedding model to use
            
        Raises:
            ValueError: If no Pinecone API key is found
            
        Example:
            >>> retriever = NewsRetriever(index_name="financial-news")
        """
        # Get API key from env var if not provided
        if pinecone_api_key is None:
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")
        
        # Initialize embedding model
        self.model = SentenceTransformer(model_name)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        self.index_name = index_name
        
        print(f"Initialized NewsRetriever with index: {index_name}")
    
    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles matching the query with enhanced processing.
        
        This method:
        1. Preprocesses the query to focus on important semantic elements
        2. Converts the query to an embedding vector
        3. Searches Pinecone for similar document vectors
        4. Formats the results for display
        
        Args:
            query: User's question or search query
            top_k: Number of results to return
            filter: Optional Pinecone filter (e.g., date range, sources)
            
        Returns:
            List of matching articles with metadata and relevance scores
            
        Example:
            >>> results = retriever.search("Latest AI research breakthroughs")
            >>> print(f"Found {len(results)} relevant articles")
        """
        # Apply advanced query preprocessing
        enhanced_query, original_query = preprocess_query(query)
        
        # Create query embedding from the enhanced query
        query_embedding = self.model.encode(enhanced_query, normalize_embeddings=True).tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "score": match.score,
                "title": match.metadata.get("title", "Unknown Title"),
                "source": match.metadata.get("source", "Unknown Source"),
                "published_at": match.metadata.get("published_at", "Unknown Date"),
                "url": match.metadata.get("url", ""),
                "content_preview": match.metadata.get("text", "")[:200] + "...",
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata
            })
        
        return formatted_results
    
    def search_by_date_range(
        self, 
        query: str, 
        from_date: str, 
        to_date: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles within a specific date range.
        
        Args:
            query: User's question or search query
            from_date: Start date in ISO format (YYYY-MM-DD)
            to_date: End date in ISO format (YYYY-MM-DD)
            top_k: Number of results to return
            
        Returns:
            List of matching articles with metadata and relevance score
            
        Example:
            >>> results = retriever.search_by_date_range(
            ...     "climate conference",
            ...     "2023-11-01",
            ...     "2023-11-30"
            ... )
        """
        # Create filter for date range
        filter = {
            "published_at": {"$gte": from_date, "$lte": to_date}
        }
        
        return self.search(query, top_k, filter)
    
    def search_by_source(
        self, 
        query: str, 
        sources: List[str], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles from specific sources.
        
        Args:
            query: User's question or search query
            sources: List of source names to filter by (e.g., ["CNN", "BBC"])
            top_k: Number of results to return
            
        Returns:
            List of matching articles with metadata and relevance score
            
        Example:
            >>> results = retriever.search_by_source(
            ...     "economic outlook",
            ...     ["Financial Times", "Wall Street Journal"]
            ... )
        """
        # Create filter for sources
        filter = {
            "source": {"$in": sources}
        }
        
        return self.search(query, top_k, filter)


# Example usage if running as script
if __name__ == "__main__":
    # Get Pinecone API key
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    # Example: Initialize the retriever and run a search
    retriever = NewsRetriever(
        pinecone_api_key=PINECONE_API_KEY,
        index_name="newsdata"
    )
    
    # Test query preprocessing
    test_queries = [
        "What is happening with artificial intelligence regulations in Europe?",
        "How has Apple's stock price changed after the recent product launch?",
        "Tell me about the climate change summit in Dubai"
    ]
    
    for query in test_queries:
        enhanced, original = preprocess_query(query)
        print(f"Original: '{original}'")
        print(f"Enhanced: '{enhanced}'")
        print()
    
    # Basic search with advanced query processing
    results = retriever.search("Latest developments in AI", top_k=3)
    
    # Print results
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
        print(f"Title: {result['title']}")
        print(f"Source: {result['source']}")
        print(f"Preview: {result['content_preview']}")

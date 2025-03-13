# tests/test_app.py

import os
import time
import math
from datetime import datetime, timedelta

import sys


# Absolute path to the folder containing the desired app.py
target_path = "/Users/sanyak/OptionsTest/NewsAnalyzer"

# Insert target_path at the beginning of sys.path
if target_path not in sys.path:
    sys.path.insert(0, target_path)

# Now import app from the target folder
import app

import pytest
from langchain.schema import Document

# --- Dummy Classes for External Dependencies ---

# Dummy NewsAPI client for fetch_news
class DummyNewsApiClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_everything(self, **kwargs):
        return {
            "articles": [
                {
                    "title": "Dummy Title",
                    "description": "Dummy Description",
                    "content": "Dummy Content",
                    "source": {"name": "Dummy Source"},
                    "publishedAt": "2025-02-27T00:00:00Z",
                    "url": "http://dummy.com"
                }
            ]
        }

    def get_top_headlines(self, **kwargs):
        return {"articles": []}

# Dummy SentenceTransformer for embeddings
class DummySentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, list):
            # Return one vector per text: a list of 1024 zeros
            return [[0.0] * 1024 for _ in texts]
        else:
            return [0.0] * 1024

# Dummy Pinecone Index and Client
class DummyPineconeIndex:
    def __init__(self, name):
        self.name = name
        self.vectors = []  # List to hold upserted records

    def upsert(self, vectors):
        self.vectors.extend(vectors)

    def delete(self, delete_all=False):
        if delete_all:
            self.vectors = []

class DummyPinecone:
    def __init__(self, api_key):
        self.api_key = api_key
        self.indexes = {}

    def list_indexes(self):
        # Provide an object with a names() method returning index names.
        class DummyIndexes:
            def __init__(self, names_list):
                self._names = names_list
            def names(self):
                return self._names
        return DummyIndexes(list(self.indexes.keys()))

    def create_index(self, name, dimension, metric, spec):
        self.indexes[name] = DummyPineconeIndex(name)

    def Index(self, index_name):
        # Return the index if exists; if not, create a new dummy index.
        if index_name not in self.indexes:
            self.indexes[index_name] = DummyPineconeIndex(index_name)
        return self.indexes[index_name]

# Dummy OpenAI Client for LLMInterface
class DummyOpenAIClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key

    class chat:
        @staticmethod
        def completions_create(extra_headers, model, messages):
            # Return a dummy response
            class DummyMessage:
                def __init__(self, content):
                    self.content = "Dummy response"
            class DummyChoice:
                def __init__(self, message):
                    self.message = message
            class DummyCompletion:
                def __init__(self):
                    self.choices = [DummyChoice(DummyMessage("Dummy response"))]
            return DummyCompletion()


# --- Begin Tests ---

# Import the app module (assumed to be named "app.py")
import app

# Test fetch_news function
def test_fetch_news(monkeypatch):
    # Set dummy NEWSAPI_KEY so fetch_news does not error
    monkeypatch.setenv("NEWSAPI_KEY", "dummy_key")
    # Override NewsApiClient with our dummy
    monkeypatch.setattr(app, "NewsApiClient", DummyNewsApiClient)
    
    articles = app.fetch_news(query="test")
    assert isinstance(articles, list)
    assert len(articles) == 1
    article = articles[0]
    assert article["title"] == "Dummy Title"
    # Check that a unique id was added
    assert article["id"].startswith("news_")
    # Check published_date conversion
    assert article.get("published_date") == "2025-02-27"

# Test create_document_from_article
def test_create_document_from_article():
    dummy_article = {
        "title": "Test Title",
        "description": "Test Description",
        "content": "Test Content",
        "source": {"name": "Test Source"},
        "publishedAt": "2025-02-27T00:00:00Z",
        "url": "http://test.com"
    }
    doc = app.create_document_from_article(dummy_article)
    assert isinstance(doc, Document)
    assert "Test Title" in doc.page_content
    assert doc.metadata["source"] == "Test Source"
    assert doc.metadata["document_type"] == "news_article"

# Test process_news_articles
def test_process_news_articles():
    dummy_article = {
        "title": "Chunk Title",
        "description": "Chunk Description",
        "content": "Chunk Content " * 20,  # long enough text to split
        "source": {"name": "Chunk Source"},
        "publishedAt": "2025-02-27T00:00:00Z",
        "url": "http://chunk.com"
    }
    articles = [dummy_article]
    # Use a small chunk size to force splitting into multiple chunks
    chunks = app.process_news_articles(articles, chunk_size=50, chunk_overlap=10)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    # At least one chunk should contain the title text
    assert any("Chunk Title" in chunk.page_content for chunk in chunks)

# Test SentenceTransformerEmbeddings
def test_sentence_transformer_embeddings(monkeypatch):
    monkeypatch.setattr(app, "SentenceTransformer", DummySentenceTransformer)
    embeddings_obj = app.SentenceTransformerEmbeddings(model_name="dummy-model")
    texts = ["This is a test."]
    embeddings = embeddings_obj.embed_documents(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1024

# Test create_pinecone_index
def test_create_pinecone_index(monkeypatch):
    dummy_pinecone = DummyPinecone(api_key="dummy")
    monkeypatch.setattr(app, "Pinecone", lambda api_key: dummy_pinecone)
    
    index_name = "test-index"
    msg = app.create_pinecone_index(pinecone_api_key="dummy", index_name=index_name, dimension=1024)
    assert index_name in dummy_pinecone.indexes
    assert "Created" in msg or "Using existing" in msg

    # Test calling again returns using existing index message.
    msg2 = app.create_pinecone_index(pinecone_api_key="dummy", index_name=index_name, dimension=1024)
    assert "Using existing" in msg2

# Test clear_pinecone_index
def test_clear_pinecone_index(monkeypatch):
    dummy_pinecone = DummyPinecone(api_key="dummy")
    index_name = "test-index-clear"
    dummy_index = DummyPineconeIndex(index_name)
    dummy_pinecone.indexes[index_name] = dummy_index
    # Pre-populate index with a dummy vector record
    dummy_index.vectors = [{"id": "dummy", "values": [0.0]*1024, "metadata": {}}]
    monkeypatch.setattr(app, "Pinecone", lambda api_key: dummy_pinecone)
    
    msg = app.clear_pinecone_index(api_key="dummy", index_name=index_name)
    assert dummy_index.vectors == []
    assert "Cleared all vectors" in msg

# Test upload_news_to_pinecone
def test_upload_news_to_pinecone(monkeypatch):
    dummy_pinecone = DummyPinecone(api_key="dummy")
    monkeypatch.setattr(app, "Pinecone", lambda api_key: dummy_pinecone)
    monkeypatch.setattr(app, "SentenceTransformer", DummySentenceTransformer)
    
    # Create a dummy document chunk
    doc = Document(
        page_content="Test content for upload",
        metadata={
            "title": "Upload Title",
            "source": "Upload Source",
            "published_at": "2025-02-27",
            "url": "http://upload.com"
        }
    )
    chunks = [doc]
    
    msg = app.upload_news_to_pinecone(
        document_chunks=chunks,
        pinecone_api_key="dummy",
        index_name="test-index-upload",
        model_name="dummy-model"
    )
    index = dummy_pinecone.Index("test-index-upload")
    assert len(index.vectors) > 0
    assert "Successfully uploaded" in msg

# Test NewsRetriever.search
def test_news_retriever_search(monkeypatch):
    # Dummy index query response
    class DummyMatch:
        def __init__(self):
            self.score = 0.95
            self.metadata = {
                "title": "Search Title",
                "source": "Search Source",
                "published_at": "2025-02-27",
                "url": "http://search.com",
                "text": "Search result text"
            }
    class DummyQueryResult:
        def __init__(self):
            self.matches = [DummyMatch()]
    class DummyPineconeIndexForSearch:
        def query(self, vector, top_k, include_metadata, filter):
            return DummyQueryResult()
    class DummyPineconeForSearch:
        def __init__(self, api_key):
            self.api_key = api_key
            self.indexes = {"test-index-search": DummyPineconeIndexForSearch()}
        def Index(self, index_name):
            return self.indexes.get(index_name, DummyPineconeIndexForSearch())
    
    monkeypatch.setattr(app, "Pinecone", lambda api_key: DummyPineconeForSearch(api_key))
    monkeypatch.setattr(app, "SentenceTransformer", DummySentenceTransformer)
    
    retriever = app.NewsRetriever(pinecone_api_key="dummy", index_name="test-index-search", model_name="dummy-model")
    results = retriever.search("test query", top_k=1)
    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]
    assert result["title"] == "Search Title"
    assert result["score"] == 0.95

# Test LLMInterface.query
def test_llm_interface_query(monkeypatch):
    # Override OpenAI client with dummy client
    monkeypatch.setattr(app, "OpenAI", lambda base_url, api_key: DummyOpenAIClient(base_url, api_key))
    
    # Create an instance of LLMInterface with dummy API key and model
    llm = app.LLMInterface(api_key="dummy", model="dummy-model", site_url="http://dummy", site_name="Dummy")
    # Call query with dummy context
    dummy_context = [{
        "title": "Context Title",
        "source": "Context Source",
        "published_at": "2025-02-27",
        "content_preview": "Context preview text",
        "url": "http://context.com"
    }]
    response = llm.query("dummy question", context=dummy_context)
    assert "Dummy response" in response

# Test change_page function using a dummy st.session_state
def test_change_page(monkeypatch):
    # Create a dummy session state dictionary
    dummy_state = {"current_page": 1}
    monkeypatch.setattr(app.st, "session_state", dummy_state)
    
    # Test "next" direction
    app.change_page("next")
    assert dummy_state["current_page"] == 2

    # Test "prev" direction
    app.change_page("prev")
    assert dummy_state["current_page"] == 1

# Optionally, you can add tests for the caching functions get_retriever and get_llm,
# but these rely on Streamlit's caching mechanism which might require more integration testing.

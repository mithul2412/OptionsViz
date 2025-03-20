#!/usr/bin/env python3
"""
Unit tests for the embedding_utils module.

This module tests the embedding, query processing, vector database,
and retrieval functionality with appropriate mocking to avoid
making actual API calls during testing.
"""

import unittest
from typing import List
from unittest.mock import patch, MagicMock

import numpy as np
from langchain.schema import Document


from news.embedding_utils import (
    SentenceTransformerEmbeddings,
    preprocess_query,
    create_pinecone_index,
    clear_pinecone_index,
    upload_news_to_pinecone,
    NewsRetriever,
)

class TestEmbeddingComponent(unittest.TestCase):
    """Test the sentence transformer embedding functionality."""

    @patch("news.embedding_utils.SentenceTransformer")
    def test_sentence_transformer_embeddings_init(self, mock_transformer):
        """Test initialization of the SentenceTransformerEmbeddings class."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        embeddings = SentenceTransformerEmbeddings(model_name="test-model")
        mock_transformer.assert_called_once_with("test-model")
        self.assertEqual(embeddings.model, mock_model)
        # Check that get_sentence_embedding_dimension was called to set dimension
        mock_model.get_sentence_embedding_dimension.assert_called_once()

    @patch("news.embedding_utils.SentenceTransformer")
    def test_embed_documents(self, mock_transformer):
        """Test embedding multiple documents."""
        mock_model = MagicMock()
        # Simulate model encoding and dimension method
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.get_sentence_embedding_dimension.return_value = 2

        embeddings = SentenceTransformerEmbeddings()
        result = embeddings.embed_documents(["Document 1", "Document 2"])

        mock_model.encode.assert_called_once_with(
            ["Document 1", "Document 2"], normalize_embeddings=True
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2])
        self.assertEqual(result[1], [0.3, 0.4])

    @patch("news.embedding_utils.SentenceTransformer")
    def test_embed_query(self, mock_transformer):
        """Test embedding a single query."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.array([0.5, 0.6])
        mock_model.get_sentence_embedding_dimension.return_value = 2

        embeddings = SentenceTransformerEmbeddings()
        result = embeddings.embed_query("Test query")
        mock_model.encode.assert_called_once_with("Test query", normalize_embeddings=True)
        self.assertEqual(result, [0.5, 0.6])

class TestQueryProcessingComponent(unittest.TestCase):
    """Test query preprocessing functionality."""

    @patch("news.embedding_utils.nlp")
    def test_preprocess_query_with_entities(self, mock_nlp):
        """Test query preprocessing with named entities."""
        mock_doc = MagicMock()
        mock_nlp.return_value = mock_doc

        # Mock named entities.
        mock_entity = MagicMock()
        mock_entity.text = "Microsoft"
        mock_doc.ents = [mock_entity]

        # Create tokens.
        token1, token2, token3 = MagicMock(), MagicMock(), MagicMock()
        token1.pos_ = "PROPN"
        token1.is_stop = False
        token1.ent_type_ = "ORG"
        token1.text = "Microsoft"

        token2.pos_ = "VERB"
        token2.is_stop = False
        token2.ent_type_ = ""
        token2.text = "announced"

        token3.pos_ = "NOUN"
        token3.is_stop = False
        token3.ent_type_ = ""
        token3.text = "product"

        mock_doc.__iter__.return_value = [token1, token2, token3]

        enhanced, original = preprocess_query("What did Microsoft announce yesterday?")
        self.assertEqual(original, "What did Microsoft announce yesterday?")
        self.assertIn("Microsoft", enhanced)
        self.assertIn("announced", enhanced)
        self.assertIn("product", enhanced)

    @patch("news.embedding_utils.nlp")
    def test_preprocess_query_without_entities(self, mock_nlp):
        """Test query preprocessing without named entities."""
        mock_doc = MagicMock()
        mock_nlp.return_value = mock_doc
        mock_doc.ents = []

        token1, token2 = MagicMock(), MagicMock()
        token1.pos_ = "NOUN"
        token1.is_stop = False
        token1.ent_type_ = ""
        token1.text = "news"

        token2.pos_ = "NOUN"
        token2.is_stop = False
        token2.ent_type_ = ""
        token2.text = "articles"

        mock_doc.__iter__.return_value = [token1, token2]
        enhanced, original = preprocess_query("Find me the latest news articles")
        self.assertEqual(original, "Find me the latest news articles")
        self.assertEqual(enhanced, "news articles")

    @patch("news.embedding_utils.nlp")
    def test_preprocess_query_fallback(self, mock_nlp):
        """Test query preprocessing fallback when enhanced query is too short."""
        mock_doc = MagicMock()
        mock_nlp.return_value = mock_doc
        mock_doc.ents = []

        token = MagicMock()
        token.pos_ = "DET"
        token.is_stop = True
        token.ent_type_ = ""
        token.text = "the"

        mock_doc.__iter__.return_value = [token]
        enhanced, original = preprocess_query("The")
        self.assertEqual(original, "The")
        self.assertEqual(enhanced, "The")

class TestVectorDatabaseComponent(unittest.TestCase):
    """Test vector database operations."""

    @patch("news.embedding_utils.Pinecone")
    def test_create_pinecone_index_existing(self, mock_pinecone):
        """Test using an existing Pinecone index."""
        mock_pc = MagicMock()
        mock_pinecone.return_value = mock_pc

        # Simulate an existing index with a dimension attribute
        mock_index_info = MagicMock()
        mock_index_info.dimension = 512
        mock_pc.describe_index.return_value = mock_index_info

        result = create_pinecone_index("fake_api_key", "test-index", 512)
        mock_pc.create_index.assert_not_called()
        self.assertEqual(result, "Using existing Pinecone index: test-index (dimension: 512)")

    @patch("news.embedding_utils.Pinecone")
    @patch("news.embedding_utils.time.sleep", return_value=None)
    def test_clear_pinecone_index(self, mock_sleep, mock_pinecone):
        """Test clearing a Pinecone index."""
        mock_pc = MagicMock()
        mock_pinecone.return_value = mock_pc

        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index

        result = clear_pinecone_index("fake_api_key", "test-index")
        mock_pc.Index.assert_called_once_with("test-index")
        mock_index.delete.assert_called_once_with(delete_all=True)
        mock_sleep.assert_called_once()
        self.assertEqual(result, "Cleared all vectors from Pinecone index: test-index")

    @patch("news.embedding_utils.Pinecone")
    @patch("news.embedding_utils.SentenceTransformerEmbeddings")
    def test_upload_news_to_pinecone(self, mock_embeddings, mock_pinecone):
        """Test uploading news to Pinecone (single batch)."""
        mock_pc = MagicMock()
        mock_pinecone.return_value = mock_pc
        # Simulate describe_index returning an object with a dimension attribute
        mock_pc.describe_index.return_value = MagicMock(dimension=1024)
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index

        # Prepare the embedding model mock with a dimension attribute
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_embedding_model.dimension = 1024
        mock_embeddings.return_value = mock_embedding_model

        docs = [
            Document(
                page_content="Test document 1",
                metadata={"title": "Title 1", "source": "Source 1", "author": None},
            ),
            Document(
                page_content="Test document 2",
                metadata={
                    "title": "Title 2",
                    "source": "Source 2",
                    "published_at": "2023-10-15",
                },
            ),
        ]

        result = upload_news_to_pinecone(
            docs,
            "fake_api_key",
            "test-index",
            batch_size=10,
            model_name="test-model",
        )
        mock_embeddings.assert_called_once_with(model_name="test-model")
        mock_embedding_model.embed_documents.assert_called_once_with(
            ["Test document 1", "Test document 2"]
        )
        mock_pc.Index.assert_called_once_with("test-index")
        mock_index.upsert.assert_called_once()
        self.assertEqual(result, "Successfully uploaded 2 chunks to Pinecone")
        upsert_args = mock_index.upsert.call_args[1]
        vectors = upsert_args["vectors"]
        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0]["metadata"]["author"], "")
        self.assertTrue(len(vectors[0]["metadata"]["text"]) <= 1000)

    @patch("news.embedding_utils.Pinecone")
    @patch("news.embedding_utils.SentenceTransformerEmbeddings")
    def test_upload_news_to_pinecone_multiple_batches(self, mock_embeddings, mock_pinecone):
        """Test uploading news to Pinecone with multiple batches."""
        mock_pc = MagicMock()
        mock_pinecone.return_value = mock_pc
        mock_pc.describe_index.return_value = MagicMock(dimension=1024)
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index

        mock_embedding_model = MagicMock()
        def embed_side_effect(texts: List[str]):
            return [np.array([0.1, 0.2]) for _ in texts]
        mock_embedding_model.embed_documents.side_effect = embed_side_effect
        mock_embedding_model.dimension = 1024
        mock_embeddings.return_value = mock_embedding_model

        docs = [
            Document(
                page_content="Doc 1",
                metadata={"title": "Title 1", "source": "Source 1", "author": None},
            ),
            Document(
                page_content="Doc 2",
                metadata={"title": "Title 2", "source": "Source 2", "author": "Author 2"},
            ),
        ]

        result = upload_news_to_pinecone(
            docs,
            "fake_api_key",
            "test-index",
            batch_size=1,
            model_name="test-model",
        )
        self.assertEqual(mock_index.upsert.call_count, 2)
        self.assertEqual(result, "Successfully uploaded 2 chunks to Pinecone")

    @patch("news.embedding_utils.Pinecone")
    @patch("news.embedding_utils.SentenceTransformer")
    def test_search_missing_text_in_metadata(self, mock_transformer, mock_pinecone):
        """Test search formatting when metadata 'text' is missing."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_model.get_sentence_embedding_dimension.return_value = 3

        mock_pc = MagicMock()
        mock_pinecone.return_value = mock_pc
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index

        # Create a match with metadata missing "text".
        mock_match = MagicMock()
        mock_match.score = 0.90
        mock_match.metadata = {
            "title": "Article Without Text",
            "source": "Source X",
            "published_at": "2023-10-10",
            "url": "https://example.com/no-text",
        }
        mock_results = MagicMock()
        mock_results.matches = [mock_match]
        mock_index.query.return_value = mock_results

        retriever = NewsRetriever(pinecone_api_key="fake_api_key")
        results = retriever.search("query", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content_preview"], "...")


#############################################
# Retriever Component Tests
#############################################

class TestRetrieverComponent(unittest.TestCase):
    """Test the news retriever functionality."""

    @patch("news.embedding_utils.SentenceTransformer")
    @patch("news.embedding_utils.Pinecone")
    def test_news_retriever_init(self, mock_pinecone, mock_transformer):
        """Test initialization of the NewsRetriever class."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        mock_model.get_sentence_embedding_dimension.return_value = 1024

        mock_pc = MagicMock()
        mock_pinecone.return_value = mock_pc
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index

        index_info = MagicMock()
        index_info.dimension = 1024
        mock_pc.describe_index.return_value = index_info

        retriever = NewsRetriever(
            pinecone_api_key="fake_api_key",
            index_name="test-index",
            model_name="test-model",
        )
        mock_transformer.assert_called_once_with("test-model")
        mock_pinecone.assert_called_once_with(api_key="fake_api_key")
        mock_pc.Index.assert_called_once_with("test-index")
        self.assertEqual(retriever.model, mock_model)
        self.assertEqual(retriever.pc, mock_pc)
        self.assertEqual(retriever.index, mock_index)
        self.assertEqual(retriever.index_name, "test-index")

    @patch("news.embedding_utils.os.getenv")
    def test_news_retriever_init_from_env(self, mock_getenv):
        """Test initialization of NewsRetriever using environment variables."""
        mock_getenv.return_value = "env_api_key"
        with patch("news.embedding_utils.SentenceTransformer") as mock_transformer, \
             patch("news.embedding_utils.Pinecone") as mock_pinecone:
            mock_transformer.return_value = MagicMock(get_sentence_embedding_dimension=lambda: 1024)
            mock_pc = MagicMock()
            mock_pinecone.return_value = mock_pc
            mock_pc.Index.return_value = MagicMock()
            index_info = MagicMock()
            index_info.dimension = 1024
            mock_pc.describe_index.return_value = index_info

            retriever = NewsRetriever(index_name="test-index")
            self.assertEqual(retriever.index_name, "test-index")
            mock_pinecone.assert_called_once_with(api_key="env_api_key")

    @patch("news.embedding_utils.os.getenv")
    def test_news_retriever_missing_api_key(self, mock_getenv):
        """Test error handling when no API key is available."""
        mock_getenv.return_value = None
        with self.assertRaises(ValueError):
            NewsRetriever()

    @patch("news.embedding_utils.preprocess_query")
    def test_search(self, mock_preprocess):
        """Test searching for news articles."""
        mock_preprocess.return_value = ("enhanced query", "original query")
        with patch("news.embedding_utils.SentenceTransformer") as mock_transformer, \
             patch("news.embedding_utils.Pinecone") as mock_pinecone:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            mock_model.get_sentence_embedding_dimension.return_value = 3
            mock_transformer.return_value = mock_model

            mock_pc = MagicMock()
            mock_pinecone.return_value = mock_pc
            mock_index = MagicMock()
            mock_pc.Index.return_value = mock_index

            mock_match1 = MagicMock()
            mock_match1.score = 0.95
            mock_match1.metadata = {
                "title": "Test Article 1",
                "source": "News Source 1",
                "published_at": "2023-10-15",
                "url": "https://example.com/1",
                "text": "This is the content of test article 1.",
            }
            mock_match2 = MagicMock()
            mock_match2.score = 0.85
            mock_match2.metadata = {
                "title": "Test Article 2",
                "source": "News Source 2",
                "text": "This is the content of test article 2.",
            }
            mock_results = MagicMock()
            mock_results.matches = [mock_match1, mock_match2]
            mock_index.query.return_value = mock_results

            retriever = NewsRetriever(pinecone_api_key="fake_api_key")
            results = retriever.search("test query", top_k=5)

            mock_preprocess.assert_called_once_with("test query")
            mock_model.encode.assert_called_once_with("enhanced query", normalize_embeddings=True)
            mock_index.query.assert_called_once()
            query_args = mock_index.query.call_args[1]
            self.assertEqual(query_args["top_k"], 5)
            self.assertEqual(query_args["include_metadata"], True)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["score"], 0.95)
            self.assertEqual(results[0]["title"], "Test Article 1")
            self.assertEqual(results[1]["score"], 0.85)
            self.assertEqual(results[1]["title"], "Test Article 2")
            self.assertIn("content_preview", results[0])

    @patch("news.embedding_utils.SentenceTransformer")
    @patch("news.embedding_utils.Pinecone")
    def test_search_empty_results(self, mock_pinecone, mock_transformer):
        """Test search when no results are returned."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.array([0.0, 0.0, 0.0])
        mock_model.get_sentence_embedding_dimension.return_value = 3

        mock_pc = MagicMock()
        mock_pinecone.return_value = mock_pc
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_results = MagicMock()
        mock_results.matches = []
        mock_index.query.return_value = mock_results

        retriever = NewsRetriever(pinecone_api_key="fake_api_key")
        results = retriever.search("no results", top_k=3)
        self.assertEqual(results, [])

    def test_search_with_date_range_filter(self):
        """Test searching with a date range filter using the search method."""
        with patch("news.embedding_utils.SentenceTransformer"), \
             patch("news.embedding_utils.Pinecone"):
            retriever = NewsRetriever(pinecone_api_key="fake_api_key")
            with patch.object(retriever.index, "query") as mock_query:
                mock_results = MagicMock()
                mock_results.matches = []
                mock_query.return_value = mock_results

                retriever.search(
                    "test query", top_k=10,
                    filter_params={"published_at": {"$gte": "2023-01-01", "$lte": "2023-12-31"}}
                )
                mock_query.assert_called_once()
                query_args = mock_query.call_args[1]
                self.assertEqual(
                    query_args["filter"],
                    {"published_at": {"$gte": "2023-01-01", "$lte": "2023-12-31"}}
                )

    def test_search_with_source_filter(self):
        """Test searching with a source filter using the search method."""
        with patch("news.embedding_utils.SentenceTransformer"), \
             patch("news.embedding_utils.Pinecone"):
            retriever = NewsRetriever(pinecone_api_key="fake_api_key")
            with patch.object(retriever.index, "query") as mock_query:
                mock_results = MagicMock()
                mock_results.matches = []
                mock_query.return_value = mock_results

                retriever.search(
                    "test query", top_k=10, filter_params={"source": {"$in": ["CNN", "BBC"]}}
                )
                mock_query.assert_called_once()
                query_args = mock_query.call_args[1]
                self.assertEqual(query_args["filter"], {"source": {"$in": ["CNN", "BBC"]}})

def main() -> None:
    """Entry point for the unit tests."""
    unittest.main()

if __name__ == "__main__":
    main()

import sys
from unittest.mock import MagicMock

# --- CRITICAL: Mock chromadb BEFORE importing app modules ---
# Because chromadb is incompatible with Python 3.14 (Pydantic V1 issue),
# we must prevent it from being imported during tests.
mock_chromadb = MagicMock()
sys.modules["chromadb"] = mock_chromadb
sys.modules["chromadb.utils"] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = MagicMock()

import unittest
from unittest.mock import patch
# Now we can safely import the service, as vector_store import of chromadb will use the mock
from app.services.rag_service import initialize_knowledge_base, retrieve_context

class TestRAGService(unittest.TestCase):

    @patch("app.services.rag_service.get_collection")
    def test_initialize_knowledge_base(self, mock_get_collection):
        # Mock collection
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        
        # Test Case 1: Collection is empty, should initialize
        mock_collection.count.return_value = 0
        initialize_knowledge_base()
        mock_collection.add.assert_called_once()
        print("✅ RAG Initialization (empty db) passed.")

        # Reset mock
        mock_collection.reset_mock()

        # Test Case 2: Collection is not empty, should skip
        mock_collection.count.return_value = 10
        initialize_knowledge_base()
        mock_collection.add.assert_not_called()
        print("✅ RAG Initialization (existing db) passed.")

    @patch("app.services.rag_service.get_collection")
    def test_retrieve_context(self, mock_get_collection):
        # Mock collection
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        
        # Mock query results
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'distances': [[0.1, 0.2]],
            'metadatas': [[None, None]],
            'embeddings': None,
            'documents': [['First relevant doc.', 'Second relevant doc.']],
            'uris': None,
            'data': None
        }

        # Test retrieval
        context = retrieve_context("test query", n_results=2)
        
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=2
        )
        
        expected_context = "First relevant doc.\n\nSecond relevant doc."
        self.assertEqual(context, expected_context)
        print("✅ RAG Retrieval passed.")

if __name__ == "__main__":
    unittest.main()

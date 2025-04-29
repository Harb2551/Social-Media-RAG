import os
import pytest
from unittest.mock import patch, MagicMock

# Import the actual class to test
from src.vector_store import SocialMediaVectorStore

# No need to define Mock classes here as they're already defined in conftest.py

class TestSocialMediaVectorStore:
    def test_init(self, mock_embeddings):
        """Test initialization with default and custom parameters."""
        # Default initialization with patched OpenAIEmbeddings
        with patch('src.vector_store.OpenAIEmbeddings', return_value=mock_embeddings):
            vs = SocialMediaVectorStore()
            assert vs.index_path == "faiss_index"
            assert vs.index_name == "support_docs"
            assert vs.vectorstore is None
        
        # Custom initialization
        vs = SocialMediaVectorStore(
            embedding_model=mock_embeddings,
            index_path="custom_path",
            index_name="custom_index"
        )
        assert vs.embedding_model == mock_embeddings
        assert vs.index_path == "custom_path"
        assert vs.index_name == "custom_index"
    
    def test_create_vectorstore(self, sample_documents, mock_embeddings):
        """Test vector store creation from documents."""
        # Mock FAISS.from_documents
        with patch('src.vector_store.FAISS') as MockFAISS:
            mock_vs = MagicMock()
            MockFAISS.from_documents.return_value = mock_vs
            
            vs = SocialMediaVectorStore(embedding_model=mock_embeddings)
            result = vs.create_vectorstore(sample_documents)
            
            # Check FAISS.from_documents was called correctly
            MockFAISS.from_documents.assert_called_once_with(
                sample_documents,
                mock_embeddings
            )
            
            # Check instance state was updated
            assert result is mock_vs
            assert vs.vectorstore is mock_vs
            
            # Test with empty document list
            result = vs.create_vectorstore([])
            assert result is None
    
    def test_save_vectorstore(self, mock_embeddings):
        """Test saving vector store to disk."""
        vs = SocialMediaVectorStore(embedding_model=mock_embeddings)
        
        # Test when vectorstore is None
        assert vs.save_vectorstore() is False
        
        # Test successful save
        with patch('os.makedirs') as mock_makedirs:
            mock_faiss = MagicMock()
            vs.vectorstore = mock_faiss
            
            result = vs.save_vectorstore()
            
            mock_makedirs.assert_called_once_with("faiss_index", exist_ok=True)
            mock_faiss.save_local.assert_called_once_with(
                folder_path="faiss_index", 
                index_name="support_docs"
            )
            assert result is True
            
            # Test exception handling
            mock_faiss.save_local.side_effect = Exception("Save error")
            result = vs.save_vectorstore()
            assert result is False
    
    def test_load_vectorstore(self, mock_embeddings):
        """Test loading vector store from disk."""
        vs = SocialMediaVectorStore(embedding_model=mock_embeddings)
        
        with patch('os.path.exists') as mock_exists:
            with patch('src.vector_store.FAISS') as MockFAISS:
                # Test when file exists
                mock_exists.return_value = True
                mock_vs = MagicMock()
                MockFAISS.load_local.return_value = mock_vs
                
                result = vs.load_vectorstore()
                
                # Check path existence check was correct
                mock_exists.assert_called_once_with(os.path.join("faiss_index", "support_docs.faiss"))
                
                # Check FAISS.load_local was called correctly
                MockFAISS.load_local.assert_called_once_with(
                    folder_path="faiss_index",
                    embeddings=mock_embeddings,
                    index_name="support_docs",
                    allow_dangerous_deserialization=True
                )
                
                # Check results
                assert result is mock_vs
                assert vs.vectorstore is mock_vs
                
                # Test when file doesn't exist
                mock_exists.return_value = False
                result = vs.load_vectorstore()
                assert result is None
    
    def test_get_embedding_for_text(self, mock_embeddings):
        """Test getting embeddings for text."""
        vs = SocialMediaVectorStore(embedding_model=mock_embeddings)
        
        # Reset mock for this test
        mock_embeddings.embed_query.reset_mock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        embedding = vs.get_embedding_for_text("Test query")
        
        mock_embeddings.embed_query.assert_called_once_with("Test query")
        assert embedding == [0.1, 0.2, 0.3, 0.4]
        
        # Test exception handling
        mock_embeddings.embed_query.side_effect = Exception("Embedding error")
        with pytest.raises(Exception):
            vs.get_embedding_for_text("Test query")
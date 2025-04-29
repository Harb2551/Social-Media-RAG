import os
import sys
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


# Create mock classes for OpenAI components to avoid API key requirements
class MockOpenAI(MagicMock):
    """Mock for ChatOpenAI that supports Runnable protocol."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __or__(self, other):
        """Support for | operator in LCEL."""
        # Returns a function that will work with format_docs in _create_chain method
        return lambda x: f"Mocked response for {x}"
        
    def invoke(self, input_value):
        """Mock invoke method to make it compatible with Runnable protocol."""
        return "This is a mocked response from invoke"


class MockEmbeddings:
    def __init__(self, *args, **kwargs):
        pass
    
    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]


# Use patch to replace actual implementations
patch("langchain_openai.OpenAIEmbeddings", MockEmbeddings).start()
patch("langchain_openai.ChatOpenAI", MockOpenAI).start()
patch("src.vector_store.OpenAIEmbeddings", MockEmbeddings).start()
patch("src.rag_chain.ChatOpenAI", MockOpenAI).start()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def sample_documents():
    """Return sample Document objects for testing."""
    return [
        Document(
            page_content="How to reset your password on X: Go to Settings, select Security, choose Password, and follow the prompts.",
            metadata={"source": "password_reset.txt"}
        ),
        Document(
            page_content="Creating a new account on X requires a valid email address or phone number. Click Sign Up and follow the instructions.",
            metadata={"source": "account_creation.txt"}
        ),
        Document(
            page_content="To delete your account, go to Settings > Your Account > Deactivate your account. Note this action has a 30-day recovery window.",
            metadata={"source": "account_deletion.txt"}
        )
    ]

@pytest.fixture
def mock_text_file(temp_dir):
    """Create a mock text file in the temporary directory."""
    file_path = os.path.join(temp_dir, "test_doc.txt")
    with open(file_path, 'w') as f:
        f.write("This is a test document for social media app support.")
    return file_path

@pytest.fixture
def mock_vectorstore():
    """Create a mock vector store."""
    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = [Document(page_content="Test content", metadata={"source": "test.txt"})]
    vectorstore.similarity_search_with_score.return_value = [(Document(page_content="Test content", metadata={"source": "test.txt"}), 0.5)]
    
    # Create a mock retriever with LCEL-compatible __or__ method
    retriever = MagicMock()
    retriever.__or__.return_value = lambda docs: f"Formatted docs: {len(docs)} documents"
    vectorstore.as_retriever.return_value = retriever
    
    return vectorstore

@pytest.fixture
def mock_llm():
    """Create a mock language model."""
    llm = MockOpenAI()
    return llm

@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings model."""
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    return embeddings
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

# Import the class to test directly
from src.rag_chain import SocialMediaRAGChain, default_prompt_template


class TestSocialMediaRAGChain:
    def test_init(self, mock_vectorstore, mock_llm):
        """Test initialization with default and custom parameters."""
        # Default initialization - we need to patch ChatOpenAI
        with patch('src.rag_chain.ChatOpenAI') as MockChatOpenAI:
            MockChatOpenAI.return_value = mock_llm
            
            chain = SocialMediaRAGChain(vectorstore=mock_vectorstore)
            assert chain.vectorstore == mock_vectorstore
            assert chain.k == 4
            assert chain.return_source_documents is True
            assert chain.similarity_threshold == 0.8
            
            # Custom initialization
            custom_prompt = "Custom prompt template {context} {question}"
            chain = SocialMediaRAGChain(
                vectorstore=mock_vectorstore,
                llm=mock_llm,
                temperature=0.5,
                k=3,
                return_source_documents=False,
                prompt_template=custom_prompt,
                similarity_threshold=0.5
            )
            
            assert chain.vectorstore == mock_vectorstore
            assert chain.llm == mock_llm
            assert chain.k == 3
            assert chain.return_source_documents is False
            assert chain.similarity_threshold == 0.5
    
    def test_create_chain(self, mock_vectorstore, mock_llm):
        """Test the chain creation process."""
        # Mock necessary classes
        with patch('src.rag_chain.PromptTemplate') as MockPromptTemplate:
            with patch('src.rag_chain.RunnablePassthrough') as MockRunnablePassthrough:
                with patch('src.rag_chain.StrOutputParser') as MockStrOutputParser:
                    MockPromptTemplate.from_template.return_value = MagicMock()
                    
                    chain = SocialMediaRAGChain(
                        vectorstore=mock_vectorstore,
                        llm=mock_llm
                    )
                    
                    # Verify retriever was set up correctly from vectorstore
                    mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 4})
                    
                    # Verify chain was created 
                    assert chain.chain is not None
    
    def test_query_success(self, mock_vectorstore, mock_llm):
        """Test successful query execution with relevant documents."""
        # Mock the LCEL chain components
        with patch('src.rag_chain.PromptTemplate') as MockPromptTemplate:
            with patch('src.rag_chain.RunnablePassthrough') as MockRunnablePassthrough:
                with patch('src.rag_chain.StrOutputParser') as MockStrOutputParser:
                    MockPromptTemplate.from_template.return_value = MagicMock()
                    
                    chain = SocialMediaRAGChain(
                        vectorstore=mock_vectorstore,
                        llm=mock_llm
                    )
                    
                    # Configure chain to return a response
                    chain.chain = MagicMock()
                    chain.chain.invoke.return_value = "Here's how to reset your password..."
                    
                    # Configure mock_vectorstore to return relevant docs with scores
                    docs = [
                        Document(page_content="How to reset password", 
                                metadata={"source": "password_reset.txt"}),
                        Document(page_content="Account recovery steps", 
                                metadata={"source": "account_recovery.txt"})
                    ]
                    
                    mock_vectorstore.similarity_search_with_score.return_value = [
                        (docs[0], 0.3),  # Below threshold
                        (docs[1], 0.5)   # Below threshold
                    ]
                    
                    result = chain.query("How do I reset my password?")
                    
                    # Verify similarity search was called
                    mock_vectorstore.similarity_search_with_score.assert_called_once_with(
                        query="How do I reset my password?",
                        k=4
                    )
                    
                    # Verify chain was invoked
                    chain.chain.invoke.assert_called_once_with("How do I reset my password?")
                    
                    # Check result structure
                    assert "answer" in result
                    assert result["answer"] == "Here's how to reset your password..."
                    assert "sources" in result
                    assert len(result["sources"]) == 2  # Both sources are unique
                    assert "source_documents" in result
                    assert len(result["source_documents"]) == 2
    
    def test_query_no_relevant_docs(self, mock_vectorstore, mock_llm):
        """Test query execution when no documents meet the similarity threshold."""
        # Same setup as previous test
        with patch('src.rag_chain.PromptTemplate') as MockPromptTemplate:
            with patch('src.rag_chain.RunnablePassthrough') as MockRunnablePassthrough:
                with patch('src.rag_chain.StrOutputParser') as MockStrOutputParser:
                    MockPromptTemplate.from_template.return_value = MagicMock()
                    
                    chain = SocialMediaRAGChain(
                        vectorstore=mock_vectorstore,
                        llm=mock_llm,
                        similarity_threshold=0.1  # Very strict threshold
                    )
                    
                    # Configure mock to return docs with scores above threshold
                    mock_vectorstore.similarity_search_with_score.return_value = [
                        (Document(page_content="Test content", metadata={"source": "doc1.txt"}), 0.8),  # Above threshold
                        (Document(page_content="More test content", metadata={"source": "doc2.txt"}), 0.9)   # Above threshold
                    ]
                    
                    result = chain.query("How do I reset my password?")
                    
                    # Check for uncertainty message
                    assert "I don't have enough information" in result["answer"]
                    assert "source_documents" in result
                    assert len(result["source_documents"]) == 0
    
    def test_query_exception_handling(self, mock_vectorstore, mock_llm):
        """Test exception handling during query execution."""
        # Same setup as previous tests
        with patch('src.rag_chain.PromptTemplate') as MockPromptTemplate:
            with patch('src.rag_chain.RunnablePassthrough') as MockRunnablePassthrough:
                with patch('src.rag_chain.StrOutputParser') as MockStrOutputParser:
                    MockPromptTemplate.from_template.return_value = MagicMock()
                    
                    chain = SocialMediaRAGChain(
                        vectorstore=mock_vectorstore,
                        llm=mock_llm
                    )
                    
                    # Set up mock to raise exception
                    mock_vectorstore.similarity_search_with_score.side_effect = Exception("Test error")
                    
                    result = chain.query("How do I reset my password?")
                    
                    # Check error handling
                    assert "I'm sorry, I encountered an error" in result["answer"]
                    assert "error" in result
                    assert result["error"] == "Test error"
                    assert "source_documents" in result
                    assert len(result["source_documents"]) == 0
    
    def test_get_relevant_documents(self, mock_vectorstore, mock_llm):
        """Test retrieving relevant documents without generating an answer."""
        with patch('src.rag_chain.PromptTemplate') as MockPromptTemplate:
            with patch('src.rag_chain.RunnablePassthrough') as MockRunnablePassthrough:
                with patch('src.rag_chain.StrOutputParser') as MockStrOutputParser:
                    MockPromptTemplate.from_template.return_value = MagicMock()
                    
                    chain = SocialMediaRAGChain(
                        vectorstore=mock_vectorstore,
                        llm=mock_llm
                    )
                    
                    # Configure mock to return specific docs
                    mock_docs = [Document(page_content="Test content", metadata={"source": "test.txt"})]
                    mock_vectorstore.similarity_search.return_value = mock_docs
                    
                    docs = chain.get_relevant_documents("password reset")
                    
                    # Verify correct method was called with default k
                    mock_vectorstore.similarity_search.assert_called_once_with("password reset", k=4)
                    assert docs == mock_docs
                    
                    # Reset mock and verify with custom k
                    mock_vectorstore.similarity_search.reset_mock()
                    chain.get_relevant_documents("password reset", k=10)
                    mock_vectorstore.similarity_search.assert_called_once_with("password reset", k=10)
import os
import pytest
from unittest.mock import patch, MagicMock

# Import the required classes directly
from src.document_loader import SocialMediaDocumentLoader
from src.vector_store import SocialMediaVectorStore
from src.rag_chain import SocialMediaRAGChain
from langchain_core.documents import Document


class TestIntegration:
    def test_end_to_end_workflow(self, temp_dir, mock_text_file):
        """Test the complete workflow from document loading to query response."""
        # Mock necessary components
        with patch("src.rag_chain.PromptTemplate") as MockPromptTemplate:
            MockPromptTemplate.from_template.return_value = MagicMock()
            
            with patch("src.rag_chain.RunnablePassthrough") as MockRunnablePassthrough:
                passthrough_instance = MagicMock()
                MockRunnablePassthrough.return_value = passthrough_instance
                
                with patch("src.rag_chain.StrOutputParser") as MockStrOutputParser:
                    # Set up a mock chain that will be returned after chaining functions
                    mock_chain = MagicMock()
                    mock_chain.invoke.return_value = "Here's how to reset your password..."
                    
                    # Make sure our mocks chain properly
                    # Simulating the following LCEL pattern: 
                    # {"context": self.retriever | format_docs, "question": RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser()
                    MockPromptTemplate.from_template.return_value.__or__.return_value = MagicMock()
                    MockPromptTemplate.from_template.return_value.__or__.return_value.__or__.return_value = mock_chain
                    
                    # 1. Set up document loader and load documents
                    with patch.object(SocialMediaDocumentLoader, 'load_and_process') as mock_load_process:
                        documents = [Document(page_content="Test content", metadata={"source": "test.txt"})]
                        mock_load_process.return_value = documents
                        
                        # 2. Create vector store with the documents
                        with patch.object(SocialMediaVectorStore, 'create_vectorstore') as mock_create_vs:
                            # Set up vector store mock with a proper retriever
                            mock_vs = MagicMock()
                            
                            # Create a mock retriever that supports LCEL
                            mock_retriever = MagicMock()
                            mock_retriever.__or__.return_value = lambda docs: f"Formatted docs: {len(docs) if isinstance(docs, list) else 'unknown'} documents"
                            mock_vs.as_retriever.return_value = mock_retriever
                            
                            # Set up similarity search
                            mock_vs.similarity_search_with_score.return_value = [(documents[0], 0.3)]
                            mock_create_vs.return_value = mock_vs
                            
                            # Execute the workflow
                            loader = SocialMediaDocumentLoader(temp_dir)
                            documents = loader.load_and_process()
                            
                            assert len(documents) > 0, "Should have loaded at least one document"
                            
                            vector_store = SocialMediaVectorStore()
                            vs = vector_store.create_vectorstore(documents)
                            
                            # 3. Create RAG chain with the vector store
                            with patch.object(SocialMediaRAGChain, '_create_chain'):
                                rag_chain = SocialMediaRAGChain(vectorstore=vs)
                                rag_chain.chain = mock_chain  # Set the chain directly
                                
                                # 4. Execute a query and check the response
                                response = rag_chain.query("How do I reset my password?")
                                
                                # Verify the query went through
                                assert "answer" in response
                                assert "Here's how to reset your password..." in response["answer"]
                                assert "source_documents" in response
                                
                                # Verify all mock methods were called
                                mock_load_process.assert_called_once()
                                mock_create_vs.assert_called_once_with(documents)
import logging
from typing import List, Dict, Any, Optional
from vector_store import QdrantVectorStore
from document_processor import DocumentProcessor
from ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) pipeline that orchestrates document processing,
    vector storage, retrieval, and response generation.
    Follows Single Responsibility Principle - handles only RAG orchestration.
    """
    
    def __init__(self):
        self.vector_store = QdrantVectorStore()
        self.document_processor = DocumentProcessor()
        self.ollama_client = OllamaClient()
        
        # Verify services are available
        self._verify_services()
    
    def _verify_services(self) -> None:
        """Verify that all required services are available."""
        try:
            # Check Ollama availability
            if not self.ollama_client.is_available():
                logger.warning("Ollama service is not available. Some features may not work.")
            
            # Check if the configured model exists
            if not self.ollama_client.check_model_exists():
                logger.warning(f"Model '{self.ollama_client.model}' not found in Ollama")
                
        except Exception as e:
            logger.error(f"Error verifying services: {e}")
    
    def add_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a document to the RAG system.
        
        Args:
            file_path: Path to the document file
            metadata: Optional additional metadata
            
        Returns:
            Dictionary with processing results and document IDs
        """
        try:
            # Validate file
            if not self.document_processor.validate_file(file_path):
                return {
                    'success': False,
                    'error': 'File validation failed',
                    'document_ids': []
                }
            
            # Process document
            logger.info(f"Processing document: {file_path}")
            documents = self.document_processor.process_file(file_path, metadata)
            
            if not documents:
                return {
                    'success': False,
                    'error': 'No content extracted from document',
                    'document_ids': []
                }
            
            # Add to vector store
            logger.info(f"Adding {len(documents)} chunks to vector store")
            document_ids = self.vector_store.add_documents(documents)
            
            return {
                'success': True,
                'message': f'Successfully processed document into {len(documents)} chunks',
                'document_ids': document_ids,
                'chunks_count': len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'document_ids': []
            }
    
    def add_text_content(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add text content directly to the RAG system.
        
        Args:
            text: Text content to add
            metadata: Optional metadata
            
        Returns:
            Dictionary with processing results
        """
        try:
            if not text.strip():
                return {
                    'success': False,
                    'error': 'Empty text provided',
                    'document_ids': []
                }
            
            # Process text into chunks
            documents = self.document_processor.chunk_text(text, metadata)
            
            if not documents:
                return {
                    'success': False,
                    'error': 'No chunks created from text',
                    'document_ids': []
                }
            
            # Add to vector store
            document_ids = self.vector_store.add_documents(documents)
            
            return {
                'success': True,
                'message': f'Successfully processed text into {len(documents)} chunks',
                'document_ids': document_ids,
                'chunks_count': len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error adding text content: {e}")
            return {
                'success': False,
                'error': str(e),
                'document_ids': []
            }
    
    def query(
        self, 
        question: str, 
        num_results: int = 5,
        temperature: float = 0.7,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            num_results: Number of similar documents to retrieve
            temperature: LLM temperature for response generation
            system_instruction: Optional system instruction for the LLM
            
        Returns:
            Dictionary with answer, retrieved documents, and metadata
        """
        try:
            if not question.strip():
                return {
                    'success': False,
                    'error': 'Empty question provided',
                    'answer': '',
                    'retrieved_documents': []
                }
            
            # Retrieve similar documents
            logger.info(f"Searching for documents similar to: {question}")
            retrieved_docs = self.vector_store.search_similar_documents(
                query=question,
                limit=num_results
            )
            
            if not retrieved_docs:
                return {
                    'success': True,
                    'answer': "I couldn't find any relevant documents to answer your question. Please make sure you have uploaded some documents first.",
                    'retrieved_documents': [],
                    'message': 'No relevant documents found'
                }
            
            # Generate RAG prompt
            rag_prompt = self.ollama_client.create_rag_prompt(
                query=question,
                retrieved_documents=retrieved_docs,
                system_instruction=system_instruction
            )
            
            # Generate response
            logger.info("Generating response using Ollama")
            answer = self.ollama_client.generate_response(
                prompt=rag_prompt,
                temperature=temperature
            )
            
            return {
                'success': True,
                'answer': answer,
                'retrieved_documents': retrieved_docs,
                'num_retrieved': len(retrieved_docs),
                'question': question
            }
            
        except Exception as e:
            logger.error(f"Error processing query '{question}': {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': 'An error occurred while processing your question.',
                'retrieved_documents': []
            }
    
    def query_streaming(
        self, 
        question: str, 
        num_results: int = 5,
        temperature: float = 0.7,
        system_instruction: Optional[str] = None
    ):
        """
        Query the RAG system with streaming response.
        
        Args:
            question: User's question
            num_results: Number of similar documents to retrieve
            temperature: LLM temperature for response generation
            system_instruction: Optional system instruction
            
        Yields:
            Response chunks and metadata
        """
        try:
            if not question.strip():
                yield {
                    'type': 'error',
                    'content': 'Empty question provided'
                }
                return
            
            # Retrieve similar documents
            yield {
                'type': 'status',
                'content': 'Searching for relevant documents...'
            }
            
            retrieved_docs = self.vector_store.search_similar_documents(
                query=question,
                limit=num_results
            )
            
            if not retrieved_docs:
                yield {
                    'type': 'answer',
                    'content': "I couldn't find any relevant documents to answer your question."
                }
                return
            
            yield {
                'type': 'status',
                'content': f'Found {len(retrieved_docs)} relevant documents. Generating response...'
            }
            
            # Generate RAG prompt
            rag_prompt = self.ollama_client.create_rag_prompt(
                query=question,
                retrieved_documents=retrieved_docs,
                system_instruction=system_instruction
            )
            
            # Stream response
            yield {
                'type': 'answer_start',
                'retrieved_documents': retrieved_docs
            }
            
            for chunk in self.ollama_client.generate_streaming_response(
                prompt=rag_prompt,
                temperature=temperature
            ):
                yield {
                    'type': 'answer_chunk',
                    'content': chunk
                }
            
            yield {
                'type': 'answer_end',
                'question': question
            }
            
        except Exception as e:
            logger.error(f"Error in streaming query '{question}': {e}")
            yield {
                'type': 'error',
                'content': f'An error occurred: {str(e)}'
            }
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            return self.vector_store.delete_document(document_id)
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get status of all system components.
        
        Returns:
            Dictionary with system status information
        """
        try:
            status = {
                'ollama': {
                    'available': self.ollama_client.is_available(),
                    'model': self.ollama_client.model,
                    'model_exists': False
                },
                'vector_store': {
                    'collection_info': {}
                }
            }
            
            # Check Ollama model
            if status['ollama']['available']:
                status['ollama']['model_exists'] = self.ollama_client.check_model_exists()
                status['ollama']['available_models'] = self.ollama_client.list_models()
            
            # Get vector store info
            try:
                status['vector_store']['collection_info'] = self.vector_store.get_collection_info()
            except Exception as e:
                logger.error(f"Error getting collection info: {e}")
                status['vector_store']['error'] = str(e)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'error': str(e)
            }
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if model was pulled successfully
        """
        try:
            return self.ollama_client.pull_model(model_name)
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

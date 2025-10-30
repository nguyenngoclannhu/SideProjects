#!/usr/bin/env python3
"""
Test script for RAG Chat System
Tests all components and provides sample interactions
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from config import Config
        from vector_store import QdrantVectorStore
        from document_processor import DocumentProcessor
        from ollama_client import OllamaClient
        from rag_pipeline import RAGPipeline
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from config import Config
        
        # Test that config values are loaded
        assert hasattr(Config, 'QDRANT_HOST')
        assert hasattr(Config, 'OLLAMA_HOST')
        assert hasattr(Config, 'EMBEDDING_MODEL')
        
        print(f"✅ Configuration loaded:")
        print(f"   Qdrant: {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
        print(f"   Ollama: {Config.OLLAMA_HOST}")
        print(f"   Model: {Config.OLLAMA_MODEL}")
        print(f"   Embedding: {Config.EMBEDDING_MODEL}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_document_processor():
    """Test document processing functionality."""
    print("\nTesting document processor...")
    try:
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test text chunking
        sample_text = """
        This is a sample document for testing the RAG system.
        It contains multiple sentences and paragraphs to test text chunking.
        
        The document processor should be able to split this text into appropriate chunks
        while maintaining semantic coherence and respecting the configured chunk size.
        
        This helps ensure that the vector search can find relevant information
        when users ask questions about the uploaded documents.
        """
        
        chunks = processor.chunk_text(sample_text, {"test": True})
        
        assert len(chunks) > 0, "No chunks created"
        assert all('content' in chunk for chunk in chunks), "Missing content in chunks"
        assert all('metadata' in chunk for chunk in chunks), "Missing metadata in chunks"
        
        print(f"✅ Document processor working - created {len(chunks)} chunks")
        return True
        
    except Exception as e:
        print(f"❌ Document processor error: {e}")
        return False

def test_ollama_client():
    """Test Ollama client connectivity."""
    print("\nTesting Ollama client...")
    try:
        from ollama_client import OllamaClient
        
        client = OllamaClient()
        
        # Test availability
        if not client.is_available():
            print("⚠️  Ollama service not available - skipping client tests")
            return False
        
        # Test model check
        model_exists = client.check_model_exists()
        if not model_exists:
            print(f"⚠️  Model '{client.model}' not found - skipping generation tests")
            return False
        
        # Test simple generation
        response = client.generate_response("Hello, this is a test.", temperature=0.1)
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        
        print(f"✅ Ollama client working - generated response: '{response[:50]}...'")
        return True
        
    except Exception as e:
        print(f"❌ Ollama client error: {e}")
        return False

def test_vector_store():
    """Test vector store functionality."""
    print("\nTesting vector store...")
    try:
        from vector_store import QdrantVectorStore
        
        vector_store = QdrantVectorStore()
        
        # Test adding documents
        test_docs = [
            {
                'content': 'This is a test document about artificial intelligence and machine learning.',
                'metadata': {'source': 'test', 'type': 'ai'}
            },
            {
                'content': 'Python is a popular programming language for data science and web development.',
                'metadata': {'source': 'test', 'type': 'programming'}
            }
        ]
        
        doc_ids = vector_store.add_documents(test_docs)
        assert len(doc_ids) == 2, "Should return 2 document IDs"
        
        # Test search
        results = vector_store.search_similar_documents("artificial intelligence", limit=1)
        assert len(results) > 0, "Should find similar documents"
        assert 'content' in results[0], "Results should contain content"
        assert 'score' in results[0], "Results should contain similarity score"
        
        # Clean up test documents
        for doc_id in doc_ids:
            vector_store.delete_document(doc_id)
        
        print("✅ Vector store working - added, searched, and cleaned up test documents")
        return True
        
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return False

def test_rag_pipeline():
    """Test the complete RAG pipeline."""
    print("\nTesting RAG pipeline...")
    try:
        from rag_pipeline import RAGPipeline
        
        rag = RAGPipeline()
        
        # Test system status
        status = rag.get_system_status()
        assert isinstance(status, dict), "Status should be a dictionary"
        
        # Test adding text content
        test_content = """
        The RAG (Retrieval-Augmented Generation) system combines information retrieval 
        with language generation. It first searches for relevant documents in a vector 
        database, then uses those documents as context for generating responses.
        
        This approach allows the system to provide accurate, contextual answers based 
        on the uploaded document collection, rather than relying solely on the 
        language model's training data.
        """
        
        result = rag.add_text_content(test_content, {"source": "test_pipeline"})
        assert result['success'], f"Failed to add content: {result.get('error')}"
        
        # Test querying (only if Ollama is available)
        if status.get('ollama', {}).get('available') and status.get('ollama', {}).get('model_exists'):
            query_result = rag.query("What is RAG?", num_results=2)
            assert query_result['success'], f"Query failed: {query_result.get('error')}"
            assert 'answer' in query_result, "Query result should contain answer"
            assert len(query_result.get('retrieved_documents', [])) > 0, "Should retrieve documents"
            
            print(f"✅ RAG pipeline working - Query: 'What is RAG?'")
            print(f"   Answer: {query_result['answer'][:100]}...")
        else:
            print("✅ RAG pipeline basic functionality working (Ollama not available for full test)")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG pipeline error: {e}")
        return False

def test_file_processing():
    """Test file processing with a temporary file."""
    print("\nTesting file processing...")
    try:
        from document_processor import DocumentProcessor
        from rag_pipeline import RAGPipeline
        
        processor = DocumentProcessor()
        rag = RAGPipeline()
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            Sample Document for Testing
            
            This is a sample document created for testing the file processing capabilities
            of the RAG system. It contains multiple paragraphs and should be processed
            correctly by the document processor.
            
            The system should be able to:
            1. Read this file
            2. Extract the text content
            3. Split it into appropriate chunks
            4. Add it to the vector database
            5. Make it searchable for queries
            
            This ensures that users can upload documents and ask questions about them.
            """)
            temp_file_path = f.name
        
        try:
            # Test file validation
            is_valid = processor.validate_file(temp_file_path)
            assert is_valid, "File should be valid"
            
            # Test file processing
            result = rag.add_document(temp_file_path, {"test_file": True})
            assert result['success'], f"File processing failed: {result.get('error')}"
            
            print(f"✅ File processing working - processed file into {result['chunks_count']} chunks")
            return True
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
        
    except Exception as e:
        print(f"❌ File processing error: {e}")
        return False

def run_interactive_test():
    """Run an interactive test session."""
    print("\n" + "="*60)
    print(" Interactive Test Session")
    print("="*60)
    
    try:
        from rag_pipeline import RAGPipeline
        
        rag = RAGPipeline()
        
        # Add some sample content
        sample_content = """
        Machine Learning Basics
        
        Machine learning is a subset of artificial intelligence that enables computers
        to learn and make decisions from data without being explicitly programmed.
        
        There are three main types of machine learning:
        1. Supervised Learning: Uses labeled data to train models
        2. Unsupervised Learning: Finds patterns in unlabeled data
        3. Reinforcement Learning: Learns through interaction and feedback
        
        Common algorithms include:
        - Linear Regression for predicting continuous values
        - Decision Trees for classification and regression
        - Neural Networks for complex pattern recognition
        - K-Means for clustering data points
        
        Applications of machine learning include:
        - Image recognition and computer vision
        - Natural language processing
        - Recommendation systems
        - Fraud detection
        - Autonomous vehicles
        """
        
        print("Adding sample content about machine learning...")
        result = rag.add_text_content(sample_content, {"topic": "machine_learning", "source": "interactive_test"})
        
        if not result['success']:
            print(f"Failed to add content: {result['error']}")
            return False
        
        print(f"✅ Added content - created {result['chunks_count']} chunks")
        
        # Interactive Q&A
        print("\nYou can now ask questions about machine learning!")
        print("Type 'quit' to exit the interactive session.")
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                print("Searching for relevant information...")
                result = rag.query(question, num_results=3)
                
                if result['success']:
                    print(f"\n🤖 Answer: {result['answer']}")
                    print(f"\n📚 Retrieved {result.get('num_retrieved', 0)} relevant document chunks")
                else:
                    print(f"❌ Error: {result['error']}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error during query: {e}")
        
        print("\nInteractive session ended.")
        return True
        
    except Exception as e:
        print(f"❌ Interactive test error: {e}")
        return False

def main():
    """Run all tests."""
    print("RAG Chat System - Test Suite")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Document Processor", test_document_processor),
        ("Ollama Client", test_ollama_client),
        ("Vector Store", test_vector_store),
        ("RAG Pipeline", test_rag_pipeline),
        ("File Processing", test_file_processing),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print(" Test Results Summary")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        
        # Offer interactive test
        response = input("\nWould you like to run an interactive test? (y/N): ")
        if response.lower() == 'y':
            run_interactive_test()
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure Qdrant and Ollama services are running.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

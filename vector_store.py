from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
import uuid
import logging
import numpy as np
from config import Config

# Try to import sentence transformers, with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)

class SimpleEmbedding:
    """Simple fallback embedding using basic text features."""
    
    def __init__(self, vector_size=384):
        self.vector_size = vector_size
    
    def encode(self, texts):
        """Create simple embeddings based on text features."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Simple feature extraction
            text = text.lower()
            
            # Basic features: length, word count, character frequencies
            features = []
            
            # Text length (normalized)
            features.append(min(len(text) / 1000.0, 1.0))
            
            # Word count (normalized)
            word_count = len(text.split())
            features.append(min(word_count / 100.0, 1.0))
            
            # Character frequency features
            char_counts = {}
            for char in text:
                if char.isalpha():
                    char_counts[char] = char_counts.get(char, 0) + 1
            
            # Top 26 character frequencies (a-z)
            for i in range(26):
                char = chr(ord('a') + i)
                freq = char_counts.get(char, 0) / max(len(text), 1)
                features.append(min(freq * 10, 1.0))
            
            # Pad or truncate to desired vector size
            while len(features) < self.vector_size:
                features.append(0.0)
            features = features[:self.vector_size]
            
            embeddings.append(np.array(features, dtype=np.float32))
        
        return embeddings if len(embeddings) > 1 else embeddings[0]

class QdrantVectorStore:
    """
    Vector store implementation using Qdrant for document storage and retrieval.
    Follows Single Responsibility Principle - handles only vector operations.
    """
    
    def __init__(self):
        self.client = QdrantClient(
            host=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT
        )
        self.collection_name = Config.QDRANT_COLLECTION_NAME
        
        # Try to use SentenceTransformer, fallback to simple embedding
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info("Attempting to load SentenceTransformer model...")
                self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
                logger.info("✅ SentenceTransformer loaded successfully")
            else:
                raise ImportError("SentenceTransformers not available")
        except Exception as e:
            logger.warning(f"⚠️  Could not load SentenceTransformer: {e}")
            logger.info("Using simple fallback embedding model")
            self.embedding_model = SimpleEmbedding()
        
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self) -> None:
        """
        Ensures the collection exists in Qdrant.
        Creates it if it doesn't exist.
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                # Get embedding dimension from the model
                sample_embedding = self.embedding_model.encode(["test"])
                vector_size = len(sample_embedding[0])
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection '{self.collection_name}' with vector size {vector_size}")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' keys
            
        Returns:
            List of document IDs that were added
        """
        try:
            points = []
            document_ids = []
            
            for doc in documents:
                doc_id = str(uuid.uuid4())
                document_ids.append(doc_id)
                
                # Generate embedding for document content
                embedding = self.embedding_model.encode([doc['content']])[0]
                
                # Create point with embedding and metadata
                point = PointStruct(
                    id=doc_id,
                    vector=embedding.tolist(),
                    payload={
                        'content': doc['content'],
                        'metadata': doc.get('metadata', {})
                    }
                )
                points.append(point)
            
            # Upload points to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search_similar_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents based on query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of similar documents with content, metadata, and similarity scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    'id': result.id,
                    'content': result.payload['content'],
                    'metadata': result.payload['metadata'],
                    'score': result.score
                })
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            raise
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[document_id]
            )
            logger.info(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': info.config.params.vectors.size,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance,
                'points_count': info.points_count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

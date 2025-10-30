import requests
import json
import logging
from typing import Dict, Any, List, Optional
from config import Config

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for interacting with Ollama LLM service.
    Follows Single Responsibility Principle - handles only LLM interactions.
    """
    
    def __init__(self):
        self.base_url = Config.OLLAMA_HOST
        self.model = Config.OLLAMA_MODEL
        self.session = requests.Session()
        self.session.timeout = 60  # 60 seconds timeout
    
    def is_available(self) -> bool:
        """
        Check if Ollama service is available.
        
        Returns:
            True if Ollama is running and accessible
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama service not available: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List of available models with their details
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            return data.get('models', [])
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if model was pulled successfully
        """
        try:
            payload = {"name": model_name}
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                stream=True
            )
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get('status') == 'success':
                        logger.info(f"Successfully pulled model: {model_name}")
                        return True
                    elif 'error' in data:
                        logger.error(f"Error pulling model: {data['error']}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response from Ollama model.
        
        Args:
            prompt: User prompt/question
            context: Optional conversation context
            system_prompt: Optional system prompt for behavior control
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            # Prepare the payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if context:
                # Format context as conversation history
                formatted_context = ""
                for msg in context:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    formatted_context += f"{role}: {content}\n"
                
                payload["prompt"] = f"{formatted_context}\nuser: {prompt}"
            
            # Make request to Ollama
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('response', '').strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error generating response: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_streaming_response(
        self, 
        prompt: str, 
        context: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Generate streaming response from Ollama model.
        
        Args:
            prompt: User prompt/question
            context: Optional conversation context
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            
        Yields:
            Response chunks as they are generated
        """
        try:
            # Prepare the payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if context:
                # Format context as conversation history
                formatted_context = ""
                for msg in context:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    formatted_context += f"{role}: {content}\n"
                
                payload["prompt"] = f"{formatted_context}\nuser: {prompt}"
            
            # Make streaming request
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'response' in data:
                            yield data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            raise
    
    def create_rag_prompt(
        self, 
        query: str, 
        retrieved_documents: List[Dict[str, Any]],
        system_instruction: str = None
    ) -> str:
        """
        Create a RAG (Retrieval-Augmented Generation) prompt.
        
        Args:
            query: User's question
            retrieved_documents: Documents retrieved from vector store
            system_instruction: Optional system instruction
            
        Returns:
            Formatted RAG prompt
        """
        # Default system instruction for RAG
        if not system_instruction:
            system_instruction = (
                "You are a professional BIM architect assistant that answers questions based on the standards and guidelines on BIM modeling. "
                "Use only the information from the context to answer the question. "
                "If the context doesn't contain enough information to answer the question, "
                "say so clearly. Be concise and accurate."
                "Always return the answers in markdown format. The answers should include the references to the source documents."
                "Please adhere to the above guidelines when providing answers"
            )
        
        # Format retrieved documents
        context_text = ""
        for i, doc in enumerate(retrieved_documents, 1):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            
            context_text += f"Document {i} (from {filename}):\n{content}\n\n"
        
        print(">>> Context Text for RAG Prompt:\n", context_text)  # Debugging line
        # Create the full prompt
        rag_prompt = f"""# System: {system_instruction}

# Context:
{context_text}

# Question: {query}

Answer the question using only the above context.
For example: 
1. Question: What is the type name of wall made of Brick? 
   Answer: B
2. Question: What is the type name of wall made of Concrete with 200mm thickness?
   Answer: C200
Formatted your response in markdown including references to the source documents as following.
## Answer: The type name is 'XYZ' as per the guidelines.
- Explanation for the answer
## References:
- Document 1's content in markdown format
- Document 2's content in markdown format
"""
        
        return rag_prompt
    
    def check_model_exists(self, model_name: str = None) -> bool:
        """
        Check if a specific model exists in Ollama.
        
        Args:
            model_name: Name of model to check (defaults to configured model)
            
        Returns:
            True if model exists
        """
        if not model_name:
            model_name = self.model
            
        try:
            models = self.list_models()
            model_names = [model.get('name', '') for model in models]
            return model_name in model_names
            
        except Exception as e:
            logger.error(f"Error checking if model exists: {e}")
            return False

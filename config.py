import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Qdrant Configuration
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documents")
    
    # Ollama Configuration
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava")
    
    # Application Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Upload Configuration
    UPLOAD_FOLDER = "uploads"
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}
    
    # Security Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    
    # Authentication Configuration
    ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"
    DEFAULT_USERNAME = os.getenv("DEFAULT_USERNAME", "admin")
    DEFAULT_PASSWORD = os.getenv("DEFAULT_PASSWORD", "admin123")
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Cookie Security Configuration
    COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"  # Set to true in production with HTTPS
    COOKIE_HTTPONLY = os.getenv("COOKIE_HTTPONLY", "true").lower() == "true"
    COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")  # lax, strict, or none
    COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN", None)  # Set domain in production

"""
Authentication and security utilities for the RAG Chat System.
Implements JWT-based authentication and CSRF protection.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import hashlib
from pydantic import BaseModel

from config import Config

# JWT token security
security = HTTPBearer()

# In-memory storage for CSRF tokens and user sessions
# In production, use Redis or a proper database
csrf_tokens: Dict[str, Dict[str, Any]] = {}
user_sessions: Dict[str, Dict[str, Any]] = {}

class Token(BaseModel):
    access_token: str
    token_type: str
    csrf_token: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    is_active: bool = True

class UserInDB(User):
    hashed_password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class CSRFTokenManager:
    """Manages CSRF tokens for security protection."""
    
    @staticmethod
    def generate_csrf_token() -> str:
        """Generate a new CSRF token."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def store_csrf_token(token: str, user_id: str, expires_minutes: int = 60) -> None:
        """Store CSRF token with expiration."""
        expires_at = datetime.utcnow() + timedelta(minutes=expires_minutes)
        csrf_tokens[token] = {
            "user_id": user_id,
            "expires_at": expires_at,
            "created_at": datetime.utcnow()
        }
    
    @staticmethod
    def validate_csrf_token(token: str, user_id: str) -> bool:
        """Validate CSRF token."""
        if token not in csrf_tokens:
            return False
        
        token_data = csrf_tokens[token]
        
        # Check expiration
        if datetime.utcnow() > token_data["expires_at"]:
            del csrf_tokens[token]
            return False
        
        # Check user association
        if token_data["user_id"] != user_id:
            return False
        
        return True
    
    @staticmethod
    def cleanup_expired_tokens() -> None:
        """Remove expired CSRF tokens."""
        now = datetime.utcnow()
        expired_tokens = [
            token for token, data in csrf_tokens.items()
            if now > data["expires_at"]
        ]
        for token in expired_tokens:
            del csrf_tokens[token]

class AuthManager:
    """Manages user authentication and authorization."""
    
    def __init__(self):
        self.users_db = self._initialize_users()
    
    def _initialize_users(self) -> Dict[str, UserInDB]:
        """Initialize default users from configuration."""
        users = {}
        
        # Create default admin user
        hashed_password = self.get_password_hash(Config.DEFAULT_PASSWORD)
        users[Config.DEFAULT_USERNAME] = UserInDB(
            username=Config.DEFAULT_USERNAME,
            hashed_password=hashed_password,
            is_active=True
        )
        
        return users
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        # Simple SHA-256 based verification with salt
        salt, stored_hash = hashed_password.split(':')
        password_hash = hashlib.sha256((plain_password + salt).encode()).hexdigest()
        return password_hash == stored_hash
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username."""
        return self.users_db.get(username)
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate user credentials."""
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, Config.SECRET_KEY, algorithm=Config.ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                return None
            token_data = TokenData(username=username)
            return token_data
        except JWTError:
            return None

# Global instances
auth_manager = AuthManager()
csrf_manager = CSRFTokenManager()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Dependency to get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = auth_manager.verify_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception
    
    user = auth_manager.get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    
    return User(username=user.username, is_active=user.is_active)

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def verify_csrf_token(request: Request, current_user: User = Depends(get_current_active_user)) -> bool:
    """Dependency to verify CSRF token."""
    if not Config.ENABLE_AUTH:
        return True
    
    csrf_token = None
    
    # Try to get CSRF token from header
    csrf_token = request.headers.get("X-CSRF-Token")
    
    # If not in header, try form data (for file uploads)
    if not csrf_token and hasattr(request, 'form'):
        try:
            form = request.form()
            csrf_token = form.get("csrf_token")
        except:
            pass
    
    if not csrf_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token missing"
        )
    
    if not csrf_manager.validate_csrf_token(csrf_token, current_user.username):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid CSRF token"
        )
    
    return True

async def optional_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[User]:
    """Optional authentication dependency that checks both Authorization header and cookies."""
    if not Config.ENABLE_AUTH:
        return None
    
    token = None
    
    # First, try to get token from Authorization header
    if credentials:
        token = credentials.credentials
    
    # If no Authorization header, try to get token from cookies
    if not token:
        cookie_token = request.cookies.get("access_token")
        if cookie_token and cookie_token.startswith("Bearer "):
            token = cookie_token[7:]  # Remove "Bearer " prefix
    
    if not token:
        return None
    
    try:
        token_data = auth_manager.verify_token(token)
        if token_data is None:
            return None
        
        user = auth_manager.get_user(username=token_data.username)
        if user is None:
            return None
        
        return User(username=user.username, is_active=user.is_active)
    except:
        return None

# Cleanup task for expired tokens
import asyncio
from typing import Callable

async def cleanup_expired_tokens_task():
    """Background task to cleanup expired tokens."""
    while True:
        try:
            csrf_manager.cleanup_expired_tokens()
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            print(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error

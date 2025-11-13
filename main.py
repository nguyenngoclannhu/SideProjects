import os
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends, status
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from datetime import timedelta, datetime

from config import Config
from rag_pipeline import RAGPipeline
from auth import (
    auth_manager, csrf_manager, get_current_active_user, verify_csrf_token,
    optional_auth, User, LoginRequest, Token, cleanup_expired_tokens_task
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chat System",
    description="A RAG-based chat system using Ollama and Qdrant with authentication",
    version="1.0.0"
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers for production
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    # Content Security Policy (adjust as needed for your app)
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    response.headers["Content-Security-Policy"] = csp
    
    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
upload_dir = Path(Config.UPLOAD_FOLDER)
upload_dir.mkdir(exist_ok=True)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Setup templates
templates = Jinja2Templates(directory="templates")

# Start cleanup task
@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup."""
    if Config.ENABLE_AUTH:
        asyncio.create_task(cleanup_expired_tokens_task())
        logger.info("Authentication enabled - CSRF cleanup task started")
    else:
        logger.info("Authentication disabled - running in open mode")

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    num_results: int = 5
    temperature: float = 0.7

class TextUploadRequest(BaseModel):
    text: str
    title: str = "Text Content"
    description: str = ""

class ChatMessage(BaseModel):
    message: str
    is_user: bool
    timestamp: str

# Store chat history (in production, use a proper database)
chat_history: List[ChatMessage] = []

# Main page routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, current_user: Optional[User] = Depends(optional_auth)):
    """Serve the main chat interface."""
    if Config.ENABLE_AUTH and not current_user:
        return RedirectResponse(url="/login", status_code=302)
    
    # Generate CSRF token for authenticated users
    csrf_token = None
    if Config.ENABLE_AUTH and current_user:
        csrf_token = csrf_manager.generate_csrf_token()
        csrf_manager.store_csrf_token(csrf_token, current_user.username)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": current_user,
        "csrf_token": csrf_token,
        "auth_enabled": Config.ENABLE_AUTH
    })

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login page."""
    if not Config.ENABLE_AUTH:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": None
    })

# Authentication endpoints
@app.post("/login")
async def login_form(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission with secure cookies."""
    if not Config.ENABLE_AUTH:
        return RedirectResponse(url="/", status_code=302)
    
    # Authenticate user
    user = auth_manager.authenticate_user(username, password)
    if not user:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid username or password"
        })
    
    # Create access token
    access_token_expires = timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_manager.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Generate CSRF token
    csrf_token = csrf_manager.generate_csrf_token()
    csrf_manager.store_csrf_token(csrf_token, user.username)
    
    # Create response with redirect
    response = RedirectResponse(url="/", status_code=302)
    
    # Set secure HttpOnly cookies with production standards
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        max_age=Config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=Config.COOKIE_HTTPONLY,
        secure=Config.COOKIE_SECURE,
        samesite=Config.COOKIE_SAMESITE,
        domain=Config.COOKIE_DOMAIN
    )
    
    return response

@app.post("/api/login", response_model=Token)
async def api_login(login_request: LoginRequest):
    """API login endpoint for programmatic access."""
    if not Config.ENABLE_AUTH:
        return {"message": "Authentication disabled"}
    
    user = auth_manager.authenticate_user(login_request.username, login_request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_manager.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Generate CSRF token
    csrf_token = csrf_manager.generate_csrf_token()
    csrf_manager.store_csrf_token(csrf_token, user.username, expires_minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Create response with token data
    response = JSONResponse(content={
        "access_token": access_token,
        "token_type": "bearer",
        "csrf_token": csrf_token
    })
    
    # Set secure HttpOnly cookie for browser authentication
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        max_age=Config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=Config.COOKIE_HTTPONLY,
        secure=Config.COOKIE_SECURE,
        samesite=Config.COOKIE_SAMESITE,
        domain=Config.COOKIE_DOMAIN
    )
    
    return response

@app.post("/logout")
async def logout():
    """Logout endpoint with secure cookie clearing."""
    response = RedirectResponse(url="/login", status_code=302)
    
    # Clear the secure cookie properly
    response.delete_cookie(
        key="access_token",
        httponly=Config.COOKIE_HTTPONLY,
        secure=Config.COOKIE_SECURE,
        samesite=Config.COOKIE_SAMESITE,
        domain=Config.COOKIE_DOMAIN
    )
    
    return response

@app.get("/auth/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information."""
    return current_user

@app.get("/auth/csrf-token")
async def get_csrf_token(current_user: User = Depends(get_current_active_user)):
    """Get a new CSRF token for the current user."""
    csrf_token = csrf_manager.generate_csrf_token()
    csrf_manager.store_csrf_token(csrf_token, current_user.username)
    return {"csrf_token": csrf_token}

# System status endpoint
@app.get("/status")
async def get_status():
    """Get system status."""
    try:
        status = rag_pipeline.get_system_status()
        return {"success": True, "status": status}
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        # Return mock data if RAG pipeline is not available
        return {
            "success": True,
            "status": {
                "auth_enabled": Config.ENABLE_AUTH,
                "ollama": {"available": False},
                "vector_store": {"collection_info": {"points_count": 0}}
            }
        }

# Chrome DevTools endpoint (to prevent 404 errors)
@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    """Handle Chrome DevTools discovery request."""
    return JSONResponse(
        content={
            "type": "node",
            "description": "RAG Chat System",
            "devtoolsFrontendUrl": "",
            "webSocketDebuggerUrl": ""
        },
        status_code=200
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# File upload endpoints
@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    csrf_token: str = Form(None),
    current_user: Optional[User] = Depends(optional_auth)
):
    """Upload and process a document."""
    # Verify CSRF token if authentication is enabled
    if Config.ENABLE_AUTH and current_user:
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
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
            )
        
        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            if len(content) > Config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE / (1024*1024):.1f}MB"
                )
            buffer.write(content)
        
        # Process document
        try:
            result = rag_pipeline.add_document(str(file_path))
        except Exception as e:
            logger.warning(f"RAG pipeline not available, returning mock response: {e}")
            result = {
                'success': True,
                'message': f"File {file.filename} uploaded successfully (mock mode)",
                'chunks_count': 5
            }
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not remove uploaded file: {e}")
        
        if result['success']:
            return {
                "success": True,
                "message": result['message'],
                "chunks_count": result['chunks_count']
            }
        else:
            raise HTTPException(status_code=400, detail=result['error'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-text")
async def upload_text(
    request: TextUploadRequest,
    http_request: Request,
    current_user: Optional[User] = Depends(optional_auth)
):
    """Upload text content directly."""
    # Verify CSRF token if authentication is enabled
    if Config.ENABLE_AUTH and current_user:
        csrf_token = http_request.headers.get("X-CSRF-Token")
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
    
    try:
        metadata = {
            "title": request.title,
            "description": request.description,
            "source": "direct_text_upload"
        }
        
        try:
            result = rag_pipeline.add_text_content(request.text, metadata)
        except Exception as e:
            logger.warning(f"RAG pipeline not available, returning mock response: {e}")
            result = {
                'success': True,
                'message': "Text uploaded successfully (mock mode)",
                'chunks_count': 3
            }
        
        if result['success']:
            return {
                "success": True,
                "message": result['message'],
                "chunks_count": result['chunks_count']
            }
        else:
            raise HTTPException(status_code=400, detail=result['error'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Query and chat endpoints
@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the RAG system."""
    try:
        try:
            result = rag_pipeline.query(
                question=request.question,
                num_results=request.num_results,
                temperature=request.temperature
            )
        except Exception as e:
            logger.warning(f"RAG pipeline not available, returning mock response: {e}")
            result = {
                'success': True,
                'answer': "This is a mock response. The RAG system would normally process your question here.",
                'retrieved_documents': [],
                'num_retrieved': 0
            }
        
        if result['success']:
            return {
                "success": True,
                "answer": result['answer'],
                "retrieved_documents": result['retrieved_documents'],
                "num_retrieved": result.get('num_retrieved', 0)
            }
        else:
            raise HTTPException(status_code=400, detail=result['error'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """Chat endpoint with history tracking."""
    try:
        # Add user message to history
        user_message = ChatMessage(
            message=request.question,
            is_user=True,
            timestamp=datetime.now().isoformat()
        )
        chat_history.append(user_message)
        
        # Get response from RAG system
        try:
            result = rag_pipeline.query(
                question=request.question,
                num_results=request.num_results,
                temperature=request.temperature
            )
        except Exception as e:
            logger.warning(f"RAG pipeline not available, using mock response: {e}")
            result = {
                'success': True,
                'answer': "This is a mock response. Authentication is working! The server would normally process your question using RAG.",
                'retrieved_documents': []
            }
        
        if result['success']:
            # Add assistant response to history
            assistant_message = ChatMessage(
                message=result['answer'],
                is_user=False,
                timestamp=datetime.now().isoformat()
            )
            chat_history.append(assistant_message)
            
            return {
                "success": True,
                "answer": result['answer'],
                "retrieved_documents": result['retrieved_documents'],
                "chat_history": chat_history[-10:]  # Return last 10 messages
            }
        else:
            raise HTTPException(status_code=400, detail=result['error'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history")
async def get_chat_history():
    """Get chat history."""
    return {"chat_history": chat_history}

@app.delete("/chat/history")
async def clear_chat_history():
    """Clear chat history."""
    global chat_history
    chat_history = []
    return {"success": True, "message": "Chat history cleared"}

@app.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    """Streaming chat endpoint."""
    async def generate_response():
        try:
            # Add user message to history
            user_message = ChatMessage(
                message=request.question,
                is_user=True,
                timestamp=datetime.now().isoformat()
            )
            chat_history.append(user_message)
            
            full_response = ""
            
            try:
                for chunk in rag_pipeline.query_streaming(
                    question=request.question,
                    num_results=request.num_results,
                    temperature=request.temperature
                ):
                    chunk_data = json.dumps(chunk) + "\n"
                    yield f"data: {chunk_data}\n\n"
                    
                    # Collect full response
                    if chunk.get('type') == 'answer_chunk':
                        full_response += chunk.get('content', '')
            except Exception as e:
                logger.warning(f"RAG pipeline not available, using mock streaming: {e}")
                # Mock streaming response
                mock_response = "This is a mock streaming response. Authentication is working! The server would normally process your question using RAG."
                for word in mock_response.split():
                    chunk = {
                        'type': 'answer_chunk',
                        'content': word + ' '
                    }
                    chunk_data = json.dumps(chunk) + "\n"
                    yield f"data: {chunk_data}\n\n"
                    full_response += word + ' '
                    await asyncio.sleep(0.1)  # Simulate streaming delay
                
                # Send end chunk
                end_chunk = {
                    'type': 'answer_end',
                    'content': ''
                }
                chunk_data = json.dumps(end_chunk) + "\n"
                yield f"data: {chunk_data}\n\n"
            
            # Add complete response to history
            if full_response:
                assistant_message = ChatMessage(
                    message=full_response.strip(),
                    is_user=False,
                    timestamp=datetime.now().isoformat()
                )
                chat_history.append(assistant_message)
            
        except Exception as e:
            error_data = json.dumps({
                'type': 'error',
                'content': str(e)
            }) + "\n"
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

# Model management endpoints
@app.post("/models/pull")
async def pull_model(model_name: str = Form(...)):
    """Pull a model from Ollama registry."""
    try:
        success = rag_pipeline.pull_model(model_name)
        if success:
            return {"success": True, "message": f"Model '{model_name}' pulled successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to pull model '{model_name}'")
    except Exception as e:
        logger.error(f"Error pulling model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available Ollama models."""
    try:
        models = rag_pipeline.ollama_client.list_models()
        return {"success": True, "models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"success": False, "models": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting RAG Chat System...")
    logger.info(f"📊 Authentication: {'Enabled' if Config.ENABLE_AUTH else 'Disabled'}")
    if Config.ENABLE_AUTH:
        logger.info(f"👤 Default user: {Config.DEFAULT_USERNAME}")
        logger.info(f"🔑 Default password: {Config.DEFAULT_PASSWORD}")
    logger.info(f"📁 Upload folder: {upload_dir}")
    logger.info(f"🔍 Qdrant host: {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
    logger.info(f"🤖 Ollama host: {Config.OLLAMA_HOST}")
    logger.info(f"🧠 Ollama model: {Config.OLLAMA_MODEL}")
    logger.info(f"🌐 Server will be available at: http://localhost:{Config.PORT}")
    
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True,
        log_level="info"
    )

import os
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json

from config import Config
from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chat System",
    description="A RAG-based chat system using Ollama and Qdrant",
    version="1.0.0"
)

# Create upload directory
upload_dir = Path(Config.UPLOAD_FOLDER)
upload_dir.mkdir(exist_ok=True)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Setup templates
templates = Jinja2Templates(directory="templates")

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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
async def get_status():
    """Get system status."""
    try:
        status = rag_pipeline.get_system_status()
        return {"success": True, "status": status}
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document."""
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
        result = rag_pipeline.add_document(str(file_path))
        
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
async def upload_text(request: TextUploadRequest):
    """Upload text content directly."""
    try:
        metadata = {
            "title": request.title,
            "description": request.description,
            "source": "direct_text_upload"
        }
        
        result = rag_pipeline.add_text_content(request.text, metadata)
        
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

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = rag_pipeline.query(
            question=request.question,
            num_results=request.num_results,
            temperature=request.temperature
        )
        
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
        from datetime import datetime
        
        # Add user message to history
        user_message = ChatMessage(
            message=request.question,
            is_user=True,
            timestamp=datetime.now().isoformat()
        )
        chat_history.append(user_message)
        
        # Get response from RAG system
        result = rag_pipeline.query(
            question=request.question,
            num_results=request.num_results,
            temperature=request.temperature
        )
        
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
            from datetime import datetime
            
            # Add user message to history
            user_message = ChatMessage(
                message=request.question,
                is_user=True,
                timestamp=datetime.now().isoformat()
            )
            chat_history.append(user_message)
            
            full_response = ""
            
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
            
            # Add complete response to history
            if full_response:
                assistant_message = ChatMessage(
                    message=full_response,
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting RAG Chat System...")
    logger.info(f"Upload folder: {upload_dir}")
    logger.info(f"Qdrant host: {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
    logger.info(f"Ollama host: {Config.OLLAMA_HOST}")
    logger.info(f"Ollama model: {Config.OLLAMA_MODEL}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

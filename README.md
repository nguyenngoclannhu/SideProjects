# RAG Chat System with Ollama and Qdrant

A complete Retrieval-Augmented Generation (RAG) system built with Python, featuring document upload, vector storage with Qdrant, and chat interface powered by Ollama LLMs.

## Features

- 📄 **Document Processing**: Support for PDF, DOCX, and TXT files
- 🔍 **Vector Search**: Powered by Qdrant vector database
- 🤖 **LLM Integration**: Uses Ollama for local language model inference
- 💬 **Chat Interface**: Modern web UI with real-time chat
- 📝 **Text Upload**: Direct text content addition
- 🔄 **Streaming Responses**: Real-time response generation
- 📊 **System Status**: Monitor service health and document count

## Architecture

The system follows SOLID principles with clear separation of concerns:

- **Document Processor**: Handles file parsing and text chunking
- **Vector Store**: Manages Qdrant operations and embeddings
- **Ollama Client**: Interfaces with Ollama LLM service
- **RAG Pipeline**: Orchestrates the complete RAG workflow
- **FastAPI Backend**: Provides REST API and web interface
- **Modern Frontend**: Responsive chat interface with drag-and-drop upload

## Prerequisites

Before running the system, ensure you have the following services installed and running:

### 1. Qdrant Vector Database

**Option A: Using Docker (Recommended)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Local Installation**
```bash
# Install Qdrant locally (see official docs)
# https://qdrant.tech/documentation/quick-start/
```

### 2. Ollama

**Install Ollama:**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows - Download from https://ollama.ai/download
```

**Start Ollama service:**
```bash
ollama serve
```

**Pull a model (e.g., Llama 2):**
```bash
ollama pull llama2
```

## Installation

1. **Clone or create the project directory:**
```bash
mkdir rag-chat-system
cd rag-chat-system
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**
```bash
# Copy and modify the .env file as needed
cp .env .env.local
```

## Configuration

The system uses environment variables for configuration. Key settings in `.env`:

```env
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2

# Application Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Usage

### 1. Start the Application

```bash
python main.py
```

The application will start on `http://localhost:8000`

### 2. Upload Documents

- **Web Interface**: Drag and drop files or click to browse
- **API Endpoint**: POST to `/upload` with file data
- **Text Content**: Use the "Add Text" button for direct text input

### 3. Chat with Your Documents

- Type questions in the chat interface
- The system will retrieve relevant document chunks
- Ollama generates responses based on the retrieved context

### 4. Monitor System Status

The sidebar shows real-time status of:
- Ollama service availability
- Qdrant connection status
- Number of documents in the vector store

## API Endpoints

### Document Management
- `POST /upload` - Upload document files
- `POST /upload-text` - Add text content directly

### Chat & Query
- `POST /chat` - Send chat message with history
- `POST /query` - Query documents without history
- `POST /chat/stream` - Streaming chat responses

### System Management
- `GET /status` - Get system status
- `GET /models` - List available Ollama models
- `POST /models/pull` - Pull new Ollama model

### Chat History
- `GET /chat/history` - Get chat history
- `DELETE /chat/history` - Clear chat history

## File Structure

```
rag-chat-system/
├── main.py                 # FastAPI application
├── config.py              # Configuration management
├── vector_store.py        # Qdrant vector operations
├── document_processor.py  # Document parsing and chunking
├── ollama_client.py       # Ollama LLM client
├── rag_pipeline.py        # RAG orchestration
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
├── templates/
│   └── index.html         # Web interface
└── uploads/               # Temporary file storage
```

## Supported File Types

- **PDF**: Text extraction from PDF documents
- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files

**File Limits:**
- Maximum file size: 10MB
- Multiple file upload supported

## Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   - Ensure Qdrant is running on port 6333
   - Check firewall settings
   - Verify QDRANT_HOST in .env

2. **Ollama Not Available**
   - Start Ollama service: `ollama serve`
   - Pull required model: `ollama pull llama2`
   - Check OLLAMA_HOST in .env

3. **Model Not Found**
   - List available models: `ollama list`
   - Pull missing model: `ollama pull <model-name>`
   - Update OLLAMA_MODEL in .env

4. **Document Processing Errors**
   - Check file format and size
   - Ensure file is not corrupted
   - Verify file permissions

### Logs

The application provides detailed logging. Check console output for:
- Service connection status
- Document processing progress
- Error messages and stack traces

## Development

### Adding New Document Types

1. Extend `DocumentProcessor._extract_from_*` methods
2. Update `Config.ALLOWED_EXTENSIONS`
3. Add file type validation

### Customizing Embeddings

1. Change `EMBEDDING_MODEL` in .env
2. Update vector dimensions in Qdrant collection
3. Restart the application

### UI Customization

- Modify `templates/index.html` for interface changes
- Update CSS styles for visual customization
- Add new JavaScript functions for features

## Performance Optimization

### For Large Document Collections

1. **Increase Chunk Size**: Adjust `CHUNK_SIZE` for longer documents
2. **Optimize Retrieval**: Tune `num_results` parameter
3. **Use GPU**: Configure Ollama for GPU acceleration
4. **Scale Qdrant**: Use Qdrant cluster for production

### Memory Management

- Monitor RAM usage with large embeddings
- Consider using quantized models
- Implement document cleanup strategies

## Security Considerations

- File upload validation and sanitization
- Rate limiting for API endpoints
- Secure file storage and cleanup
- Environment variable protection

## Contributing

1. Follow SOLID principles for new features
2. Add comprehensive error handling
3. Include unit tests for new components
4. Update documentation for changes

## License

This project is open source. See LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Verify service dependencies
4. Create detailed issue reports

---

**Note**: This system is designed for local development and testing. For production deployment, consider additional security measures, scalability improvements, and monitoring solutions.

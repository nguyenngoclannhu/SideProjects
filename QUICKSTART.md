# Quick Start Guide - RAG Chat System

Get your RAG Chat System up and running in minutes!

## 🚀 Quick Setup (5 minutes)

### 1. Prerequisites Check
Make sure you have:
- Python 3.8+ installed
- Docker installed (for Qdrant)
- Internet connection

### 2. Install Dependencies
```bash
# Use the dependency installer to avoid version conflicts
python install_deps.py
```

This script will:
- ✅ Install dependencies in the correct order
- ✅ Handle version conflicts automatically
- ✅ Use fallback implementations when needed

**Alternative: Standard pip install**
```bash
pip install -r requirements.txt
```

### 3. Manual Setup (if needed)

If the automated installation doesn't work, follow these steps:

**Install Dependencies Manually:**
```bash
pip install fastapi uvicorn python-multipart jinja2
pip install qdrant-client sentence-transformers
pip install PyPDF2 python-docx ollama
pip install python-dotenv pydantic numpy requests
```

**Start Qdrant:**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Install & Start Ollama:**
```bash
# macOS
brew install ollama
ollama serve

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Pull a model
ollama pull llama2
```

### 4. Start the Application
```bash
python main.py
```

### 5. Open Your Browser
Navigate to: **http://localhost:8000**

## 🎯 First Steps

### Upload Your First Document
1. **Drag & Drop**: Drop a PDF, DOCX, or TXT file into the upload area
2. **Or Click**: Click the upload area to browse files
3. **Or Add Text**: Click "📝 Add Text" to paste content directly

### Ask Your First Question
1. Type a question in the chat box
2. Press Enter or click Send
3. Watch the AI search your documents and respond!

## 📋 Test Everything Works

Run the test suite:
```bash
python test_system.py
```

This will test all components and offer an interactive demo.

## 🔧 Common Issues & Solutions

### "Qdrant connection failed"
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### "Ollama not available"
```bash
# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull llama2
```

### "Import errors"
```bash
# Install dependencies
pip install -r requirements.txt
```

### "Permission denied"
```bash
# Make scripts executable
chmod +x setup.py test_system.py
```

## 🎮 Try These Examples

### Example 1: Upload a Research Paper
1. Upload a PDF research paper
2. Ask: "What is the main conclusion of this paper?"
3. Ask: "What methodology was used?"

### Example 2: Company Documentation
1. Upload your company's documentation
2. Ask: "What is our refund policy?"
3. Ask: "How do I reset my password?"

### Example 3: Technical Manual
1. Upload a technical manual
2. Ask: "How do I install this software?"
3. Ask: "What are the system requirements?"

## 📊 System Status

Check the sidebar for real-time status:
- 🟢 **Green dot**: Service is running
- 🔴 **Red dot**: Service needs attention
- **Document count**: Number of chunks in your database

## 🔄 Advanced Usage

### Custom Configuration
Edit `.env` file to customize:
- Model selection (`OLLAMA_MODEL=llama2`)
- Chunk size (`CHUNK_SIZE=1000`)
- Embedding model (`EMBEDDING_MODEL=all-MiniLM-L6-v2`)

### API Usage
The system provides REST APIs:
- `POST /upload` - Upload files
- `POST /chat` - Chat with documents
- `GET /status` - System status

### Streaming Responses
For real-time responses, use:
- `POST /chat/stream` - Streaming chat

## 🎯 Tips for Better Results

### Document Upload Tips
- **Use clear, well-formatted documents**
- **Avoid scanned PDFs** (text extraction may fail)
- **Break large documents** into sections if needed
- **Use descriptive filenames**

### Question Tips
- **Be specific**: "What is the pricing for Pro plan?" vs "What is pricing?"
- **Use document language**: Match the terminology in your documents
- **Ask follow-up questions**: Build on previous answers
- **Reference context**: "Based on the manual you just mentioned..."

### Performance Tips
- **Smaller chunks** = more precise but slower
- **Larger chunks** = faster but less precise
- **More retrieved documents** = better context but slower
- **Lower temperature** = more focused answers

## 🛠️ Troubleshooting

### Check Logs
The application shows detailed logs in the terminal. Look for:
- Connection errors
- Processing status
- Error messages

### Restart Services
If something isn't working:
```bash
# Stop everything
docker stop qdrant-rag
pkill -f ollama

# Restart
docker run -p 6333:6333 qdrant/qdrant
ollama serve
python main.py
```

### Clear Data
To start fresh:
```bash
# Remove Qdrant container and data
docker rm -f qdrant-rag
docker run -p 6333:6333 qdrant/qdrant
```

## 🎉 You're Ready!

Your RAG Chat System is now ready to use! 

- **Upload documents** and start asking questions
- **Experiment** with different types of content
- **Share** with your team for collaborative knowledge management
- **Customize** the system for your specific needs

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the code to understand how RAG works
- Customize the UI in `templates/index.html`
- Add new document types in `document_processor.py`
- Scale up with production deployment

---

**Need Help?** Check the troubleshooting section in README.md or run `python test_system.py` to diagnose issues.

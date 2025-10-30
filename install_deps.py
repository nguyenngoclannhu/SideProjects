#!/usr/bin/env python3
"""
Dependency installation script for RAG Chat System
Handles version conflicts and ensures compatible installations
"""

import subprocess
import sys
import os

def run_pip_command(command):
    """Run a pip command and return success status."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def install_dependencies():
    """Install dependencies in the correct order to avoid conflicts."""
    print("Installing RAG Chat System Dependencies")
    print("=" * 50)
    
    # Upgrade pip first
    print("\n1. Upgrading pip...")
    if not run_pip_command(f"{sys.executable} -m pip install --upgrade pip"):
        return False
    
    # Install core dependencies first
    print("\n2. Installing core dependencies...")
    core_deps = [
        "python-dotenv==1.0.0",
        "pydantic==2.5.0",
        "numpy==1.24.3",
        "requests==2.31.0",
    ]
    
    for dep in core_deps:
        if not run_pip_command(f"{sys.executable} -m pip install {dep}"):
            return False
    
    # Install ML dependencies
    print("\n3. Installing ML dependencies...")
    ml_deps = [
        "torch>=1.9.0",
        "transformers==4.35.2",
        "sentence-transformers==2.2.2",
    ]
    
    for dep in ml_deps:
        if not run_pip_command(f"{sys.executable} -m pip install {dep}"):
            return False
    
    # Install web framework
    print("\n4. Installing web framework...")
    web_deps = [
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "python-multipart==0.0.6",
        "jinja2==3.1.2",
    ]
    
    for dep in web_deps:
        if not run_pip_command(f"{sys.executable} -m pip install {dep}"):
            return False
    
    # Install document processing
    print("\n5. Installing document processing...")
    doc_deps = [
        "PyPDF2==3.0.1",
        "python-docx==1.1.0",
    ]
    
    for dep in doc_deps:
        if not run_pip_command(f"{sys.executable} -m pip install {dep}"):
            return False
    
    # Install vector database client
    print("\n6. Installing vector database client...")
    if not run_pip_command(f"{sys.executable} -m pip install qdrant-client==1.7.0"):
        return False
    
    # Install LangChain (simplified version)
    print("\n7. Installing text processing...")
    langchain_deps = [
        "langchain-text-splitters==0.0.1",
    ]
    
    for dep in langchain_deps:
        if not run_pip_command(f"{sys.executable} -m pip install {dep}"):
            # If this fails, we'll use a simple text splitter
            print("⚠️  LangChain text splitters not available, will use simple splitter")
    
    # Install Ollama client
    print("\n8. Installing Ollama client...")
    if not run_pip_command(f"{sys.executable} -m pip install ollama==0.1.7"):
        return False
    
    print("\n✅ All dependencies installed successfully!")
    return True

def main():
    """Main installation function."""
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the errors above.")
        sys.exit(1)
    
    print("\n🎉 Installation completed successfully!")
    print("\nNext steps:")
    print("1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    print("2. Start Ollama: ollama serve")
    print("3. Pull a model: ollama pull llama2")
    print("4. Run the application: python main.py")

if __name__ == "__main__":
    main()

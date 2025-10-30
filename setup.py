#!/usr/bin/env python3
"""
Setup script for RAG Chat System
Helps with installation, dependency checking, and initial setup
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def print_step(step, text):
    """Print a formatted step."""
    print(f"\n[{step}] {text}")

def run_command(command, description="", check=True):
    """Run a shell command and handle errors."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print_step("1", "Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_virtual_environment():
    """Check if running in a virtual environment."""
    print_step("2", "Checking virtual environment...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        return True
    else:
        print("⚠️  Not running in virtual environment")
        print("Recommendation: Create and activate a virtual environment:")
        print("  python -m venv venv")
        print("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        
        response = input("Continue anyway? (y/N): ")
        return response.lower() == 'y'

def install_dependencies():
    """Install Python dependencies."""
    print_step("3", "Installing Python dependencies...")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    success = run_command(f"{sys.executable} -m pip install --upgrade pip")
    if not success:
        print("❌ Failed to upgrade pip")
        return False
    
    success = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    if success:
        print("✅ Dependencies installed successfully")
        return True
    else:
        print("❌ Failed to install dependencies")
        return False

def check_docker():
    """Check if Docker is available."""
    print_step("4", "Checking Docker availability...")
    
    success = run_command("docker --version", check=False)
    if success:
        print("✅ Docker is available")
        return True
    else:
        print("⚠️  Docker not found. You'll need to install Qdrant manually.")
        return False

def start_qdrant():
    """Start Qdrant using Docker."""
    print_step("5", "Starting Qdrant vector database...")
    
    # Check if Qdrant is already running
    try:
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is already running")
            return True
    except:
        pass
    
    print("Starting Qdrant container...")
    success = run_command("docker run -d -p 6333:6333 --name qdrant-rag qdrant/qdrant", check=False)
    
    if success:
        print("Waiting for Qdrant to start...")
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:6333/collections", timeout=2)
                if response.status_code == 200:
                    print("✅ Qdrant started successfully")
                    return True
            except:
                time.sleep(1)
        
        print("❌ Qdrant failed to start within 30 seconds")
        return False
    else:
        print("❌ Failed to start Qdrant container")
        print("Try manually: docker run -p 6333:6333 qdrant/qdrant")
        return False

def check_ollama():
    """Check if Ollama is available."""
    print_step("6", "Checking Ollama availability...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            
            # Check if llama2 model is available
            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]
            
            if any('llama2' in name for name in model_names):
                print("✅ Llama2 model is available")
                return True
            else:
                print("⚠️  Llama2 model not found")
                print("Run: ollama pull llama2")
                return False
        else:
            print("❌ Ollama is not responding correctly")
            return False
    except requests.exceptions.RequestException:
        print("❌ Ollama is not running")
        print("Install and start Ollama:")
        print("  macOS: brew install ollama && ollama serve")
        print("  Linux: curl -fsSL https://ollama.ai/install.sh | sh && ollama serve")
        print("  Windows: Download from https://ollama.ai/download")
        return False

def create_directories():
    """Create necessary directories."""
    print_step("7", "Creating directories...")
    
    directories = ["uploads", "templates"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")
    
    return True

def test_system():
    """Test the complete system."""
    print_step("8", "Testing system components...")
    
    try:
        # Test imports
        print("Testing imports...")
        from config import Config
        from vector_store import QdrantVectorStore
        from document_processor import DocumentProcessor
        from ollama_client import OllamaClient
        from rag_pipeline import RAGPipeline
        print("✅ All imports successful")
        
        # Test Qdrant connection
        print("Testing Qdrant connection...")
        vector_store = QdrantVectorStore()
        print("✅ Qdrant connection successful")
        
        # Test Ollama connection
        print("Testing Ollama connection...")
        ollama_client = OllamaClient()
        if ollama_client.is_available():
            print("✅ Ollama connection successful")
        else:
            print("❌ Ollama connection failed")
            return False
        
        # Test RAG pipeline
        print("Testing RAG pipeline...")
        rag = RAGPipeline()
        status = rag.get_system_status()
        print("✅ RAG pipeline initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main setup function."""
    print_header("RAG Chat System Setup")
    print("This script will help you set up the RAG Chat System")
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Check and start services
    docker_available = check_docker()
    if docker_available:
        qdrant_ok = start_qdrant()
    else:
        print("Please install and start Qdrant manually")
        qdrant_ok = False
    
    ollama_ok = check_ollama()
    
    # Create directories
    create_directories()
    
    # Test system if services are available
    if qdrant_ok and ollama_ok:
        test_success = test_system()
    else:
        test_success = False
    
    # Final summary
    print_header("Setup Summary")
    
    if test_success:
        print("🎉 Setup completed successfully!")
        print("\nTo start the application:")
        print("  python main.py")
        print("\nThen open: http://localhost:8000")
    else:
        print("⚠️  Setup completed with warnings")
        print("\nPlease ensure the following services are running:")
        if not qdrant_ok:
            print("  - Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        if not ollama_ok:
            print("  - Ollama: ollama serve")
            print("  - Model: ollama pull llama2")
        
        print("\nThen run: python main.py")
    
    print("\nFor detailed instructions, see README.md")

if __name__ == "__main__":
    main()

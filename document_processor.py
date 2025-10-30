import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from docx import Document
from config import Config

# Try to import LangChain text splitter, fallback to simple splitter
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        # Simple fallback text splitter
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=200, length_function=len, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", " ", ""]
            
            def split_text(self, text: str) -> List[str]:
                """Simple text splitting implementation."""
                if not text:
                    return []
                
                chunks = []
                current_chunk = ""
                
                # Split by separators in order of preference
                for separator in self.separators:
                    if separator in text:
                        parts = text.split(separator)
                        for part in parts:
                            if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                                if current_chunk:
                                    current_chunk += separator + part
                                else:
                                    current_chunk = part
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk)
                                current_chunk = part
                        break
                
                # If no separator worked, split by character count
                if not chunks and len(text) > self.chunk_size:
                    for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                        chunk = text[i:i + self.chunk_size]
                        if chunk.strip():
                            chunks.append(chunk)
                elif current_chunk:
                    chunks.append(current_chunk)
                
                return [chunk.strip() for chunk in chunks if chunk.strip()]

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Document processing class that handles file reading and text chunking.
    Follows Single Responsibility Principle - handles only document processing.
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text content from various file formats.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension == '.docx':
                return self._extract_from_docx(file_path)
            elif file_extension == '.txt':
                return self._extract_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
            raise
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading DOCX file {file_path}: {e}")
            raise
    
    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read().strip()
            except Exception as e:
                logger.error(f"Error reading TXT file {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {e}")
            raise
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks for vector storage.
        
        Args:
            text: Text content to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of document chunks with content and metadata
        """
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        try:
            chunks = self.text_splitter.split_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update({
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk)
                    })
                    
                    documents.append({
                        'content': chunk.strip(),
                        'metadata': chunk_metadata
                    })
            
            logger.info(f"Created {len(documents)} chunks from text")
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise
    
    def process_file(self, file_path: str, additional_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Complete file processing pipeline: extract text and create chunks.
        
        Args:
            file_path: Path to the file to process
            additional_metadata: Optional additional metadata to attach
            
        Returns:
            List of processed document chunks ready for vector storage
        """
        try:
            # Extract text from file
            text = self.extract_text_from_file(file_path)
            
            if not text.strip():
                logger.warning(f"No text extracted from file: {file_path}")
                return []
            
            # Prepare metadata
            file_path_obj = Path(file_path)
            metadata = {
                'filename': file_path_obj.name,
                'file_path': str(file_path_obj),
                'file_extension': file_path_obj.suffix.lower(),
                'file_size': file_path_obj.stat().st_size,
                'processed_at': str(file_path_obj.stat().st_mtime)
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Create chunks
            documents = self.chunk_text(text, metadata)
            
            logger.info(f"Successfully processed file {file_path} into {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if file can be processed.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid for processing
        """
        try:
            file_path_obj = Path(file_path)
            
            # Check if file exists
            if not file_path_obj.exists():
                logger.error(f"File does not exist: {file_path}")
                return False
            
            # Check file extension
            if file_path_obj.suffix.lower() not in Config.ALLOWED_EXTENSIONS:
                logger.error(f"Unsupported file extension: {file_path_obj.suffix}")
                return False
            
            # Check file size
            file_size = file_path_obj.stat().st_size
            if file_size > Config.MAX_FILE_SIZE:
                logger.error(f"File too large: {file_size} bytes (max: {Config.MAX_FILE_SIZE})")
                return False
            
            # Check if file is readable
            if not os.access(file_path, os.R_OK):
                logger.error(f"File is not readable: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False

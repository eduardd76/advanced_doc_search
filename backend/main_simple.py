"""
Simple Document Search API - Lightweight FastAPI Application
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import our simple modules
from retrieval.simple_engine import SimpleRetrievalEngine
from processing.document_processor import AdvancedDocumentProcessor
from processing.chunker import AdvancedChunker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
simple_engine = None
document_processor = None
chunker = None

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    min_score: float = 0.0

class IndexRequest(BaseModel):
    name: Optional[str] = "default"
    directory_paths: List[str]
    file_extensions: List[str] = ["*.pdf", "*.docx", "*.txt", "*.md"] 
    force_rebuild: bool = False
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Also accept the old format for backward compatibility
    directory_path: Optional[str] = None
    file_patterns: Optional[List[str]] = None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_results: int
    query_time_ms: float

class IndexResponse(BaseModel):
    message: str
    total_documents: int
    total_chunks: int
    processing_time_ms: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global simple_engine, document_processor, chunker
    
    logger.info("Initializing Simple Document Search System")
    
    # Initialize components
    try:
        simple_engine = SimpleRetrievalEngine(min_doc_freq=1, max_doc_freq=10)
        document_processor = AdvancedDocumentProcessor()
        # Initialize chunker with smaller default settings for simple version
        chunker = AdvancedChunker(
            base_chunk_size=256,
            min_chunk_size=10,
            max_chunk_size=2048,
            overlap_percentage=10
        )
        
        logger.info("✅ Simple system initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize simple system: {e}")
        simple_engine = SimpleRetrievalEngine(min_doc_freq=1, max_doc_freq=10)  # Fallback
        document_processor = AdvancedDocumentProcessor()
        chunker = AdvancedChunker()
    
    yield
    
    logger.info("Shutting down Simple Document Search System")

# Create FastAPI app
app = FastAPI(
    title="Simple Document Search API",
    description="Lightweight document search and retrieval system using only BM25",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3003", "http://127.0.0.1:3003", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Simple Document Search API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine_ready": simple_engine is not None and simple_engine.bm25 is not None
    }

@app.post("/index", response_model=IndexResponse)
async def create_index(request: IndexRequest, background_tasks: BackgroundTasks):
    """Create search index from documents"""
    import time
    start_time = time.time()
    
    try:
        # Handle both new and old format
        if request.directory_paths:
            directory_paths = request.directory_paths
            file_patterns = request.file_extensions
        elif request.directory_path:
            directory_paths = [request.directory_path]
            file_patterns = request.file_patterns or ["*.pdf", "*.docx", "*.txt", "*.md"]
        else:
            raise HTTPException(status_code=400, detail="Either directory_paths or directory_path must be provided")
        
        # Convert file extensions to glob patterns if needed
        if file_patterns:
            converted_patterns = []
            for pattern in file_patterns:
                if pattern.startswith('.') and not pattern.startswith('*.'):
                    # Convert .pdf to *.pdf
                    converted_patterns.append('*' + pattern)
                elif pattern.startswith('*.'):
                    # Already a glob pattern
                    converted_patterns.append(pattern)
                else:
                    # Unknown format, add * prefix just in case
                    converted_patterns.append('*.' + pattern.lstrip('.'))
            file_patterns = converted_patterns
        
        logger.info(f"Starting indexing of directories: {directory_paths}")
        
        # Check if directories exist
        for dir_path in directory_paths:
            if not Path(dir_path).exists():
                raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")
        
        # Always clear the index for new indexing to avoid confusion
        simple_engine.clear_index()
        logger.info("Index cleared for new indexing operation")
        
        # Process documents
        documents = []
        total_chunks = 0
        total_files_found = 0
        
        for directory_path in directory_paths:
            for pattern in file_patterns:
                all_files = list(Path(directory_path).glob(pattern))
                # Filter out temporary Office files (starting with ~$)
                files = [f for f in all_files if not f.name.startswith('~$')]
                
                total_files_found += len(files)
                if len(all_files) != len(files):
                    logger.info(f"Filtered out {len(all_files) - len(files)} temporary Office files")
                logger.info(f"Found {len(files)} files matching pattern: {pattern} in {directory_path}")
                
                for file_path in files:
                    try:
                        logger.info(f"Processing file: {file_path}")
                        
                        # Process document
                        processed_doc = document_processor.process_document(str(file_path))
                        logger.info(f"Document processed, content length: {len(processed_doc.content) if processed_doc.content else 0}")
                        
                        doc_content = processed_doc.content
                        if not doc_content:
                            logger.warning(f"Empty content for file: {file_path}")
                            continue
                        
                        # Create a chunker with request-specific parameters
                        request_chunker = AdvancedChunker(
                            base_chunk_size=request.chunk_size,
                            min_chunk_size=max(5, min(20, request.chunk_size // 20)),  # At least 5 words, reasonable minimum for small docs
                            max_chunk_size=request.chunk_size * 2,
                            overlap_percentage=int((request.chunk_overlap / request.chunk_size) * 100)
                        )
                        
                        # Create chunks
                        doc_chunks = request_chunker.chunk_document(
                            doc_content,
                            doc_id=str(file_path.stem),
                            metadata={
                                "source_file": str(file_path),
                                "file_name": file_path.name,
                                "file_type": file_path.suffix,
                                "directory": str(Path(directory_path).name)
                            }
                        )
                        
                        # Add chunks to index
                        logger.info(f"Created {len(doc_chunks)} chunks for {file_path}")
                        for i, doc_chunk in enumerate(doc_chunks):
                            chunk_id = f"{file_path.stem}_chunk_{i}"
                            chunk_metadata = {
                                "source_file": str(file_path),
                                "file_name": file_path.name,
                                "chunk_index": i,
                                "total_chunks": len(doc_chunks),
                                "file_type": file_path.suffix,
                                "directory": str(Path(directory_path).name),
                                "start_position": doc_chunk.start_position,
                                "end_position": doc_chunk.end_position
                            }
                            
                            simple_engine.add_document(chunk_id, doc_chunk.content, chunk_metadata)
                            total_chunks += 1
                        
                        documents.append(str(file_path))
                        logger.info(f"Successfully processed file: {file_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process file {file_path}: {type(e).__name__}: {e}")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        continue
        
        # Check if any files were found
        if total_files_found == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"No files found matching patterns {file_patterns} in directories {directory_paths}. "
                       f"Please check that the directories exist and contain files with the specified extensions."
            )
        
        # Build index
        if simple_engine.total_documents > 0:
            simple_engine.build_index()
            logger.info("✅ Index built successfully")
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"No documents were processed successfully. Found {total_files_found} files but none could be processed. "
                       f"Please check file permissions and formats."
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return IndexResponse(
            message="Index created successfully",
            total_documents=len(documents),
            total_chunks=total_chunks,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents"""
    import time
    start_time = time.time()
    
    try:
        if simple_engine.bm25 is None:
            raise HTTPException(status_code=400, detail="No index available. Create an index first.")
        
        # Perform search
        results = simple_engine.search(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        # Convert results to response format
        formatted_results = []
        for result in results:
            formatted_results.append({
                "doc_id": result.doc_id,
                "score": result.score,
                "content": result.content,
                "metadata": result.metadata,
                "highlighted_content": result.highlighted_content
            })
        
        query_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=formatted_results,
            total_results=len(results),
            query_time_ms=query_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """Get index statistics"""
    try:
        if simple_engine is None:
            return {"error": "Engine not initialized"}
        
        stats = simple_engine.get_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.delete("/index")
async def clear_index():
    """Clear the current index"""
    try:
        simple_engine.clear_index()
        return {"message": "Index cleared successfully"}
        
    except Exception as e:
        logger.error(f"Failed to clear index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {str(e)}")

@app.get("/api/indexes/list")
async def list_indexes():
    """List available indexes - simplified version"""
    try:
        # In the simple version, we only have one active index
        has_index = simple_engine is not None and simple_engine.bm25 is not None
        if has_index:
            stats = simple_engine.get_statistics()
            return {
                "indexes": [{
                    "name": "current_index", 
                    "documents": stats["total_documents"],
                    "status": "active"
                }]
            }
        else:
            return {"indexes": []}
        
    except Exception as e:
        logger.error(f"Failed to list indexes: {e}")
        return {"indexes": []}

@app.get("/api/indexes/status")  
async def get_indexes_status():
    """Get indexing status - compatible format for frontend"""
    try:
        if simple_engine is None or simple_engine.bm25 is None:
            return {
                "is_indexing": False,
                "current_progress": 0,
                "total_documents": 0,
                "processed_documents": 0,
                "failed_documents": 0,
                "status_message": "No index available",
                "current_index": None
            }
        
        stats = simple_engine.get_statistics()
        return {
            "is_indexing": False,  # Simple version is always synchronous
            "current_progress": stats["total_documents"],
            "total_documents": stats["total_documents"], 
            "processed_documents": stats["total_documents"],
            "failed_documents": 0,
            "status_message": f"Index completed with {stats['total_documents']} documents and {stats['vocabulary_size']} vocabulary terms",
            "current_index": "current_index"
        }
        
    except Exception as e:
        logger.error(f"Failed to get indexing status: {e}")
        return {
            "is_indexing": False,
            "current_progress": 0,
            "total_documents": 0,
            "processed_documents": 0,
            "failed_documents": 1,
            "status_message": f"Error getting status: {str(e)}",
            "current_index": None
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
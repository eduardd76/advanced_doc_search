"""
Advanced Document Search API - Main FastAPI Application
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
from openai import OpenAI

# Import our modules
from config import get_config
from retrieval.hybrid_engine import HybridRetrievalEngine
from processing.document_processor import AdvancedDocumentProcessor
from processing.chunker import AdvancedChunker
from evaluation.evaluator import RetrievalEvaluator
from synthesis.synthesizer import DocumentSynthesizer, SynthesisResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
config = get_config()
hybrid_engine = None
document_processor = None
chunker = None
evaluator = None

# Indexing state
indexing_status = {
    "is_indexing": False,
    "current_progress": 0,
    "total_documents": 0,
    "status_message": "Ready to create index",
    "current_index": None,
    "processed_documents": 0,
    "failed_documents": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global hybrid_engine, document_processor, chunker, evaluator
    
    # Startup
    logger.info("Starting Advanced Document Search API...")
    
    # Initialize components
    hybrid_engine = HybridRetrievalEngine(
        bm25_config={
            'k1': config.bm25.k1,
            'b': config.bm25.b,
            'min_doc_freq': config.bm25.min_doc_freq,
            'max_doc_freq': config.bm25.max_doc_freq
        },
        dense_config={
            'model_name': config.dense_retrieval.model_name,
            'embedding_dim': config.dense_retrieval.embedding_dim,
            'batch_size': config.dense_retrieval.batch_size,
            'device': config.dense_retrieval.device
        },
        reranker_config={
            'model_name': config.cross_encoder.model_name,
            'max_candidates': config.cross_encoder.max_candidates,
            'batch_size': config.cross_encoder.batch_size,
            'device': config.cross_encoder.device
        },
        hybrid_config={
            'use_rrv_fusion': config.hybrid.use_rrv_fusion,
            'rrv_k': config.hybrid.rrv_k,
            'bm25_weight': config.hybrid.bm25_weight,
            'dense_weight': config.hybrid.dense_weight,
            'final_rerank_size': config.hybrid.final_rerank_size,
            'min_score_threshold': config.hybrid.min_score_threshold
        }
    )
    
    document_processor = AdvancedDocumentProcessor(
        enable_ocr=config.get('document_processing.ocr_enabled', True),
        ocr_languages=config.get('document_processing.ocr_languages', ['eng']),
        remove_boilerplate=config.get('text_processing.remove_boilerplate', True),
        detect_language=config.get('text_processing.language_detection', True)
    )
    
    chunker = AdvancedChunker(
        strategy=config.chunking.strategy,
        base_chunk_size=config.chunking.base_chunk_size,
        overlap_percentage=config.chunking.overlap_percentage,
        min_chunk_size=config.chunking.min_chunk_size,
        max_chunk_size=config.chunking.max_chunk_size,
        dynamic_overlap=config.chunking.dynamic_overlap,
        preserve_headers=config.chunking.preserve_headers,
        sentence_boundary=config.chunking.sentence_boundary,
        dedup_threshold=config.chunking.dedup_threshold
    )
    
    evaluator = RetrievalEvaluator()
    
    logger.info("Advanced Document Search API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Advanced Document Search API...")

# Create FastAPI app
app = FastAPI(
    title="Advanced Document Search API",
    description="State-of-the-art hybrid retrieval system with BM25, dense embeddings, and cross-encoder reranking",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class IndexRequest(BaseModel):
    name: str
    directory_paths: List[str]
    file_extensions: Optional[List[str]] = None
    force_rebuild: Optional[bool] = False
    chunk_size: Optional[int] = 1024

class SearchRequest(BaseModel):
    query: str
    index_name: str
    top_k: Optional[int] = 10
    use_reranking: Optional[bool] = True
    bm25_candidates: Optional[int] = 100
    dense_candidates: Optional[int] = 100

class EvaluationRequest(BaseModel):
    index_name: str
    benchmark_file: Optional[str] = None
    create_sample_benchmark: Optional[bool] = True
    num_sample_queries: Optional[int] = 20

class ChatRequest(BaseModel):
    message: str
    index_name: str
    api_key: Optional[str] = None
    model: Optional[str] = "gpt-3.5-turbo"
    use_synthesis: Optional[bool] = True

# API Endpoints

@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "status": "online",
        "service": "Advanced Document Search API",
        "version": "2.0.0",
        "features": [
            "Hybrid BM25 + Dense Retrieval",
            "Cross-encoder Reranking",
            "Structure-aware Chunking",
            "Multi-format Document Processing",
            "Comprehensive Evaluation Framework"
        ]
    }

@app.get("/api/status")
async def get_status():
    """Get system status and statistics"""
    global hybrid_engine, indexing_status
    
    system_stats = {}
    if hybrid_engine:
        system_stats = hybrid_engine.get_statistics()
    
    return {
        "system_status": "operational",
        "indexing_status": indexing_status,
        "system_statistics": system_stats,
        "configuration": {
            "server_port": config.server.port,
            "storage_dir": config.storage.base_dir,
            "supported_formats": document_processor.get_supported_formats() if document_processor else []
        }
    }

@app.post("/api/indexes/create")
async def create_index(request: IndexRequest, background_tasks: BackgroundTasks):
    """Create a new hybrid index"""
    global indexing_status, hybrid_engine, document_processor, chunker
    
    if indexing_status["is_indexing"]:
        raise HTTPException(status_code=409, detail="Another indexing operation is in progress")
    
    if not request.directory_paths:
        raise HTTPException(status_code=400, detail="At least one directory path is required")
    
    # Validate directories
    valid_directories = []
    logger.info(f"Received directory paths: {request.directory_paths}")
    
    for directory in request.directory_paths:
        # Strip whitespace and normalize path
        directory = directory.strip()
        if not directory:
            continue
        
        # Handle different path formats
        directory = directory.replace('\\', '/')
        dir_path = Path(directory).resolve()
        logger.info(f"Checking directory: '{directory}' -> '{dir_path}'")
        
        if dir_path.exists() and dir_path.is_dir():
            valid_directories.append(str(dir_path))
            logger.info(f"Valid directory: {dir_path}")
        else:
            logger.warning(f"Invalid directory: '{directory}' -> '{dir_path}' (exists: {dir_path.exists()}, is_dir: {dir_path.is_dir() if dir_path.exists() else 'N/A'})")
            # Check if the path exists but is not a directory
            if dir_path.exists():
                logger.warning(f"Path exists but is not a directory: {dir_path}")
            else:
                # Try alternative path formats
                alt_path = Path(directory)
                logger.warning(f"Alternative path check: {alt_path} (exists: {alt_path.exists()})")
    
    if not valid_directories:
        raise HTTPException(status_code=404, detail="No valid directories found")
    
    # Set file extensions
    file_extensions = request.file_extensions or config.get('document_processing.supported_formats', [
        '.pdf', '.docx', '.txt', '.md', '.epub', '.rtf', '.html'
    ])
    
    # Start background indexing
    background_tasks.add_task(
        build_index_background,
        request.name,
        valid_directories,
        file_extensions,
        request.force_rebuild,
        request.chunk_size
    )
    
    return {
        "message": f"Indexing started for {len(valid_directories)} directories",
        "index_name": request.name,
        "directories": valid_directories,
        "file_extensions": file_extensions
    }

@app.get("/api/indexes/status")
async def get_indexing_status():
    """Get current indexing status"""
    return indexing_status

@app.get("/api/indexes/list")
async def list_indexes():
    """Get list of available indexes (only those created through the Index tab)"""
    indexes_dir = Path(config.storage.indexes_dir)
    index_names = set()
    
    if indexes_dir.exists():
        # Look for indexes that have stats files (created through Index tab)
        for stats_dir in indexes_dir.iterdir():
            if stats_dir.is_dir():
                stats_file = stats_dir / "index_stats.json"
                if stats_file.exists():
                    # Verify the actual index files exist
                    index_name = stats_dir.name
                    bm25_file = indexes_dir / f"{index_name}_bm25.pkl"
                    dense_file = indexes_dir / f"{index_name}_dense.faiss"
                    
                    if bm25_file.exists() and dense_file.exists():
                        index_names.add(index_name)
    
    return {"indexes": sorted(list(index_names))}

@app.post("/api/indexes/load")
async def load_index(request: dict):
    """Load a specific index"""
    global hybrid_engine
    
    index_name = request.get("index_name")
    if not index_name:
        raise HTTPException(status_code=400, detail="Index name is required")
    
    index_path = Path(config.storage.indexes_dir) / index_name
    
    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    
    try:
        # Load the index
        hybrid_engine.load_index(str(index_path))
        logger.info(f"Loaded index: {index_name}")
        
        return {
            "message": f"Index '{index_name}' loaded successfully",
            "index_name": index_name
        }
        
    except Exception as e:
        logger.error(f"Error loading index {index_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load index: {str(e)}")

@app.delete("/api/indexes/{index_name}")
async def delete_index(index_name: str):
    """Delete an index and all its associated files"""
    import shutil
    
    if not index_name.strip():
        raise HTTPException(status_code=400, detail="Index name is required")
    
    indexes_dir = Path(config.storage.indexes_dir)
    
    try:
        # Files to delete
        files_to_delete = [
            indexes_dir / f"{index_name}_bm25.pkl",
            indexes_dir / f"{index_name}_dense.faiss", 
            indexes_dir / f"{index_name}_dense.pkl",
            indexes_dir / f"{index_name}_hybrid.pkl"
        ]
        
        # Directory to delete
        stats_dir = indexes_dir / index_name
        
        # Delete all index files
        deleted_files = []
        for file_path in files_to_delete:
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(file_path.name)
        
        # Delete stats directory
        if stats_dir.exists():
            shutil.rmtree(stats_dir)
            deleted_files.append(f"{index_name}/ directory")
        
        if not deleted_files:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
        
        logger.info(f"Deleted index '{index_name}': {', '.join(deleted_files)}")
        
        return {
            "message": f"Index '{index_name}' deleted successfully",
            "deleted_files": deleted_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting index {index_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete index: {str(e)}")

@app.post("/api/search")
async def search_documents(request: SearchRequest):
    """Perform hybrid search"""
    global hybrid_engine
    
    if not hybrid_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Perform hybrid search
        results = hybrid_engine.search(
            query=request.query,
            top_k=request.top_k,
            bm25_candidates=request.bm25_candidates,
            dense_candidates=request.dense_candidates,
            use_reranking=request.use_reranking,
            show_progress=False
        )
        
        # Format results with context expansion for incomplete chunks
        formatted_results = []
        seen_docs = set()  # Track which documents we've seen
        
        for result in results:
            # Check if content ends abruptly (common patterns)
            content = result.content
            needs_continuation = (
                content.rstrip().endswith(':') or 
                content.rstrip().endswith('steps:') or
                content.rstrip().endswith('following:') or
                content.rstrip().endswith('includes:') or
                content.rstrip().endswith('such as:')
            )
            
            # If content needs continuation, try to find the next chunk
            if needs_continuation and result.metadata:
                # Extract base document ID (remove chunk suffix)
                base_doc_id = result.doc_id.split('#')[0] if '#' in result.doc_id else result.doc_id
                
                # Look for continuation in other results
                continuation_found = False
                for other_result in results:
                    if other_result.doc_id != result.doc_id:
                        other_base_id = other_result.doc_id.split('#')[0] if '#' in other_result.doc_id else other_result.doc_id
                        
                        # Check if it's from the same document
                        if other_base_id == base_doc_id:
                            # Check if it's likely the next chunk
                            if '#' in result.doc_id and '#' in other_result.doc_id:
                                current_chunk = int(result.doc_id.split('#')[1])
                                other_chunk = int(other_result.doc_id.split('#')[1])
                                
                                # If it's the next sequential chunk
                                if other_chunk == current_chunk + 1:
                                    content += "\n\n[Continuation from next chunk:]\n" + other_result.content
                                    continuation_found = True
                                    seen_docs.add(other_result.doc_id)
                                    break
            
            # Skip if we've already included this as a continuation
            if result.doc_id in seen_docs:
                continue
                
            formatted_result = {
                "doc_id": result.doc_id,
                "content": content,
                "score": result.final_score,
                "metadata": result.metadata,
                "component_scores": result.component_scores,
                "rank_info": result.rank_info
            }
            formatted_results.append(formatted_result)
            seen_docs.add(result.doc_id)
        
        return {
            "query": request.query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "search_method": "hybrid",
            "parameters": {
                "top_k": request.top_k,
                "use_reranking": request.use_reranking,
                "bm25_candidates": request.bm25_candidates,
                "dense_candidates": request.dense_candidates
            }
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
async def evaluate_system(request: EvaluationRequest):
    """Evaluate the retrieval system"""
    global hybrid_engine, evaluator
    
    if not hybrid_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    try:
        # Load or create benchmark queries
        if request.create_sample_benchmark:
            queries = evaluator.create_sample_benchmark(request.num_sample_queries)
        elif request.benchmark_file:
            queries = evaluator.load_benchmark_queries(request.benchmark_file)
        else:
            raise HTTPException(status_code=400, detail="Either create_sample_benchmark or benchmark_file must be provided")
        
        if not queries:
            raise HTTPException(status_code=400, detail="No benchmark queries available")
        
        # Evaluate hybrid system
        evaluation_result = evaluator.evaluate_retrieval_system(
            hybrid_engine,
            method_name="hybrid_bm25_dense_rerank"
        )
        
        # Format results
        return {
            "evaluation_method": "hybrid_bm25_dense_rerank",
            "benchmark_queries": len(queries),
            "metrics": {
                "recall_at_5": evaluation_result['aggregate_metrics'].recall_at_5,
                "recall_at_10": evaluation_result['aggregate_metrics'].recall_at_10,
                "precision_at_5": evaluation_result['aggregate_metrics'].precision_at_5,
                "precision_at_10": evaluation_result['aggregate_metrics'].precision_at_10,
                "mrr": evaluation_result['aggregate_metrics'].mrr,
                "ndcg_at_5": evaluation_result['aggregate_metrics'].ndcg_at_5,
                "ndcg_at_10": evaluation_result['aggregate_metrics'].ndcg_at_10,
                "map_score": evaluation_result['aggregate_metrics'].map_score,
                "hit_rate": evaluation_result['aggregate_metrics'].hit_rate,
                "avg_retrieval_time": evaluation_result['aggregate_metrics'].avg_retrieval_time
            },
            "total_queries_evaluated": evaluation_result['total_queries']
        }
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_documents(request: ChatRequest):
    """Chat with documents using search context"""
    global hybrid_engine
    
    if not hybrid_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    try:
        # Search for relevant documents
        search_results = hybrid_engine.search(
            query=request.message,
            top_k=5,
            use_reranking=True
        )
        
        # Build context
        context_parts = []
        sources = []
        
        for i, result in enumerate(search_results):
            context_parts.append(f"Document {i+1}: {result.content[:2000]}")
            sources.append({
                "doc_id": result.doc_id,
                "score": result.final_score,
                "metadata": result.metadata
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate response - require API key for functionality
        if not request.api_key or not request.api_key.strip():
            raise HTTPException(
                status_code=400, 
                detail="OpenAI API key is required for chat functionality. Please provide your API key in the chat settings."
            )
        
        # Initialize OpenAI client with user's API key
        try:
            client = OpenAI(api_key=request.api_key)
            
            # Create a prompt with context
            system_prompt = """You are an AI assistant that answers questions based on provided document context. 
            Use the context to provide accurate, detailed answers. If the context doesn't contain enough information 
            to answer the question, say so clearly. Always cite which documents you're referencing."""
            
            user_prompt = f"""Question: {request.message}

Document Context:
{context}

Please provide a comprehensive answer based on this context."""

            # Make OpenAI API call
            completion = client.chat.completions.create(
                model=request.model if request.model else "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            response = completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Fall back to context-based response if OpenAI fails
            response = f"I found {len(search_results)} relevant documents. Here's what they contain:\n\n{context[:3000]}\n\n(Note: AI synthesis unavailable - using basic context mode)"
        
        return {
            "query": request.message,
            "response": response,
            "context_used": len(search_results),
            "sources": sources,
            "synthesis_mode": request.use_synthesis and bool(request.api_key)
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions

def build_index_background(
    index_name: str, 
    directories: List[str], 
    file_extensions: List[str],
    force_rebuild: bool = False,
    chunk_size: int = 1024
):
    """Build index in background"""
    global indexing_status, hybrid_engine, document_processor, chunker
    
    try:
        # Reinitialize chunker with custom chunk size if provided
        if chunk_size != config.chunking.base_chunk_size:
            chunker = AdvancedChunker(
                strategy=config.chunking.strategy,
                base_chunk_size=chunk_size,
                overlap_percentage=config.chunking.overlap_percentage,
                min_chunk_size=config.chunking.min_chunk_size,
                max_chunk_size=max(chunk_size * 2, config.chunking.max_chunk_size),
                dynamic_overlap=config.chunking.dynamic_overlap,
                preserve_headers=config.chunking.preserve_headers,
                sentence_boundary=config.chunking.sentence_boundary,
                dedup_threshold=config.chunking.dedup_threshold
            )
            logger.info(f"Using custom chunk size: {chunk_size}")
        
        indexing_status.update({
            "is_indexing": True,
            "current_index": index_name,
            "status_message": "Starting indexing process...",
            "current_progress": 0,
            "total_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0
        })
        
        logger.info(f"Starting background indexing for '{index_name}' with directories: {directories}")
        
        # Check if index already exists
        index_path = Path(config.storage.indexes_dir) / index_name
        if index_path.exists() and not force_rebuild:
            indexing_status["status_message"] = "Index already exists. Use force_rebuild=true to overwrite."
            return
        
        # Process all documents
        all_documents = []
        total_files = 0
        
        # First pass: count files
        logger.info(f"Scanning directories for documents: {directories}")
        for directory in directories:
            logger.info(f"Processing directory: {directory}")
            docs = document_processor.process_directory(
                directory, 
                recursive=True, 
                file_extensions=file_extensions
            )
            all_documents.extend(docs)
            total_files += len(docs)
            logger.info(f"Found {len(docs)} documents in {directory}")
        
        indexing_status.update({
            "total_documents": total_files,
            "status_message": f"Found {total_files} documents. Processing and chunking..."
        })
        
        if total_files == 0:
            indexing_status.update({
                "status_message": "No documents found in specified directories. Check paths and file extensions.",
                "is_indexing": False
            })
            logger.warning(f"No documents found for indexing in directories: {directories}")
            return
        
        # Clear existing index
        hybrid_engine.clear_index()
        
        # Process and chunk documents
        total_chunks = 0
        processed_docs = 0
        failed_docs = 0
        
        for doc in all_documents:
            try:
                processed_docs += 1
                indexing_status.update({
                    "current_progress": processed_docs,
                    "processed_documents": processed_docs,
                    "failed_documents": failed_docs,
                    "status_message": f"Processing: {Path(doc.file_path).name} ({processed_docs}/{total_files})"
                })
                
                if not doc.content.strip():
                    failed_docs += 1
                    indexing_status["failed_documents"] = failed_docs
                    continue
                
                # Chunk document
                chunks = chunker.chunk_document(
                    text=doc.content,
                    doc_id=doc.doc_id,
                    metadata=doc.metadata
                )
                
                # Add chunks to hybrid engine
                for chunk in chunks:
                    hybrid_engine.add_document(
                        doc_id=chunk.chunk_id,
                        content=chunk.content,
                        metadata={
                            **chunk.metadata,
                            'original_doc_id': doc.doc_id,
                            'chunk_info': {
                                'word_count': chunk.word_count,
                                'section_header': chunk.section_header,
                                'page_number': chunk.page_number
                            }
                        }
                    )
                
                total_chunks += len(chunks)
                
            except Exception as e:
                logger.error(f"Error processing document {doc.doc_id}: {e}")
                failed_docs += 1
                indexing_status.update({
                    "failed_documents": failed_docs,
                    "status_message": f"Error processing {Path(doc.file_path).name}: {str(e)}"
                })
                continue
        
        indexing_status.update({
            "status_message": "Building search indexes...",
            "processed_documents": processed_docs,
            "failed_documents": failed_docs
        })
        
        # Build the hybrid index
        hybrid_engine.build_index(show_progress=False)
        
        # Save index
        hybrid_engine.save_index(str(index_path))
        
        # Save index statistics
        import json
        from datetime import datetime
        stats = {
            "name": index_name,
            "created_at": datetime.now().isoformat(),
            "document_count": processed_docs,
            "chunk_count": total_chunks,
            "failed_documents": failed_docs,
            "directories": directories,
            "file_extensions": file_extensions,
            "chunk_size": chunk_size
        }
        
        # Ensure the index directory exists
        index_path.mkdir(parents=True, exist_ok=True)
        
        stats_file = index_path / "index_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Update final status
        indexing_status.update({
            "status_message": f"Index '{index_name}' created successfully! "
                            f"Processed {processed_docs} documents into {total_chunks} chunks. "
                            f"Failed: {failed_docs}",
            "current_progress": total_files
        })
        
        logger.info(f"Indexing completed: {index_name} - {processed_docs} docs, {total_chunks} chunks, {failed_docs} failed")
        
    except Exception as e:
        logger.error(f"Indexing error for '{index_name}': {e}", exc_info=True)
        indexing_status.update({
            "status_message": f"Indexing failed: {str(e)}",
            "is_indexing": False
        })
    
    finally:
        indexing_status["is_indexing"] = False

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_level="info"
    )
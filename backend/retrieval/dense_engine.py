"""
Dense embedding retrieval engine using sentence transformers and FAISS
"""

import numpy as np
import faiss
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)

@dataclass
class DenseDocument:
    """Document representation for dense retrieval"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

@dataclass
class DenseSearchResult:
    """Dense search result"""
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any]

class DenseRetrievalEngine:
    """
    Dense retrieval engine using sentence transformers and FAISS:
    - Semantic similarity search
    - GPU/CPU optimization
    - Batch processing
    - Memory-efficient indexing
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: str = "cpu",
        faiss_index_type: str = "IndexFlatIP",
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.faiss_index_type = faiss_index_type
        self.cache_dir = cache_dir
        
        # Initialize components
        self.model = None
        self.faiss_index = None
        self.documents = []
        self.doc_id_to_index = {}
        
        # Load model
        self._load_model()
        
        # Initialize FAISS index
        self._initialize_faiss_index()
    
    def _load_model(self):
        """Load sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            
            # Verify embedding dimension
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            actual_dim = test_embedding.shape[1]
            
            if actual_dim != self.embedding_dim:
                logger.warning(f"Model embedding dim ({actual_dim}) != expected ({self.embedding_dim})")
                self.embedding_dim = actual_dim
                # Reinitialize FAISS index with correct dimension
                self._initialize_faiss_index()
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index"""
        try:
            if self.faiss_index_type == "IndexFlatIP":
                # Inner Product index (for normalized embeddings)
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            elif self.faiss_index_type == "IndexFlatL2":
                # L2 distance index
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            elif self.faiss_index_type == "IndexHNSWFlat":
                # HNSW index for faster approximate search
                self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                self.faiss_index.hnsw.efSearch = 128
            elif self.faiss_index_type == "IndexIVFFlat":
                # IVF index for large datasets
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            else:
                raise ValueError(f"Unsupported FAISS index type: {self.faiss_index_type}")
            
            logger.info(f"FAISS index initialized: {self.faiss_index_type}")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            raise
    
    def _encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode texts into embeddings"""
        if not texts:
            return np.array([])
        
        try:
            # Encode in batches to manage memory
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), self.batch_size), 
                         desc="Encoding texts", 
                         disable=not show_progress):
                batch_texts = texts[i:i + self.batch_size]
                
                # Encode batch
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings
                )
                
                all_embeddings.append(batch_embeddings)
                
                # Clear GPU memory if using CUDA
                if self.device.startswith('cuda'):
                    torch.cuda.empty_cache()
            
            # Combine all embeddings
            embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
            
            logger.debug(f"Encoded {len(texts)} texts to embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add a document to the index"""
        if metadata is None:
            metadata = {}
        
        if not content.strip():
            logger.warning(f"Empty content for document {doc_id}")
            return
        
        # Create document without embedding (will be computed during build)
        document = DenseDocument(
            doc_id=doc_id,
            content=content,
            metadata=metadata,
            embedding=None
        )
        
        # Add to collection
        doc_index = len(self.documents)
        self.documents.append(document)
        self.doc_id_to_index[doc_id] = doc_index
        
        logger.debug(f"Added document {doc_id} for dense retrieval")
    
    def build_index(self, show_progress: bool = True):
        """Build the dense retrieval index"""
        if not self.documents:
            raise ValueError("No documents to index")
        
        logger.info(f"Building dense index for {len(self.documents)} documents")
        
        # Extract content for encoding
        contents = [doc.content for doc in self.documents]
        
        # Encode all documents
        embeddings = self._encode_texts(contents, show_progress=show_progress)
        
        if embeddings.size == 0:
            raise ValueError("No embeddings generated")
        
        # Store embeddings in documents
        for i, doc in enumerate(self.documents):
            doc.embedding = embeddings[i]
        
        # Add embeddings to FAISS index
        if self.faiss_index_type == "IndexIVFFlat":
            # Train IVF index
            logger.info("Training IVF index...")
            self.faiss_index.train(embeddings)
        
        self.faiss_index.add(embeddings)
        
        # Clear memory
        del embeddings
        gc.collect()
        
        logger.info(f"Dense index built with {self.faiss_index.ntotal} vectors")
    
    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        min_score: float = 0.0
    ) -> List[DenseSearchResult]:
        """Search documents using dense embeddings"""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            raise ValueError("Index not built. Call build_index() first.")
        
        if not query.strip():
            logger.warning("Empty query")
            return []
        
        try:
            # Encode query
            query_embedding = self._encode_texts([query], show_progress=False)
            
            if query_embedding.size == 0:
                logger.warning("Query encoding failed")
                return []
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Convert to results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # Invalid index
                    break
                
                score = float(score)
                if score < min_score:
                    continue
                
                doc = self.documents[idx]
                
                result = DenseSearchResult(
                    doc_id=doc.doc_id,
                    score=score,
                    content=doc.content,
                    metadata=doc.metadata
                )
                results.append(result)
            
            logger.debug(f"Dense search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[DenseDocument]:
        """Get document by ID"""
        if doc_id in self.doc_id_to_index:
            return self.documents[self.doc_id_to_index[doc_id]]
        return None
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text"""
        return self._encode_texts([text], show_progress=False)[0]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts"""
        embeddings = self._encode_texts([text1, text2], show_progress=False)
        
        if embeddings.shape[0] != 2:
            return 0.0
        
        # Compute cosine similarity
        if self.normalize_embeddings:
            # If embeddings are normalized, dot product = cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1])
        else:
            # Compute cosine similarity manually
            norm1 = np.linalg.norm(embeddings[0])
            norm2 = np.linalg.norm(embeddings[1])
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = np.dot(embeddings[0], embeddings[1]) / (norm1 * norm2)
        
        return float(similarity)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_documents": len(self.documents),
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "embedding_dimension": self.embedding_dim,
            "index_built": self.faiss_index is not None and self.faiss_index.ntotal > 0,
            "model_name": self.model_name,
            "device": self.device,
            "faiss_index_type": self.faiss_index_type,
            "normalize_embeddings": self.normalize_embeddings
        }
    
    def save_index(self, index_path: str):
        """Save dense index to disk"""
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_path = index_path.with_suffix('.faiss')
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(faiss_path))
        
        # Save documents and metadata (without embeddings to save space)
        docs_data = []
        for doc in self.documents:
            doc_data = {
                'doc_id': doc.doc_id,
                'content': doc.content,
                'metadata': doc.metadata
            }
            docs_data.append(doc_data)
        
        metadata = {
            'documents': docs_data,
            'doc_id_to_index': self.doc_id_to_index,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'batch_size': self.batch_size,
            'normalize_embeddings': self.normalize_embeddings,
            'device': self.device,
            'faiss_index_type': self.faiss_index_type
        }
        
        # Save metadata
        metadata_path = index_path.with_suffix('.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Dense index saved to {index_path}")
    
    def load_index(self, index_path: str):
        """Load dense index from disk"""
        index_path = Path(index_path)
        
        # Load metadata
        metadata_path = index_path.with_suffix('.pkl')
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Restore documents
        self.documents = []
        for doc_data in metadata['documents']:
            doc = DenseDocument(
                doc_id=doc_data['doc_id'],
                content=doc_data['content'],
                metadata=doc_data['metadata'],
                embedding=None  # Embeddings not stored
            )
            self.documents.append(doc)
        
        self.doc_id_to_index = metadata['doc_id_to_index']
        
        # Update configuration
        self.model_name = metadata['model_name']
        self.embedding_dim = metadata['embedding_dim']
        self.batch_size = metadata['batch_size']
        self.normalize_embeddings = metadata['normalize_embeddings']
        self.faiss_index_type = metadata['faiss_index_type']
        
        # Load FAISS index
        faiss_path = index_path.with_suffix('.faiss')
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        else:
            logger.warning("FAISS index file not found. Index needs to be rebuilt.")
            self._initialize_faiss_index()
        
        # Load model if needed
        if self.model is None:
            self._load_model()
        
        logger.info(f"Dense index loaded from {index_path}")
    
    def clear_index(self):
        """Clear the current index"""
        self.faiss_index = None
        self.documents.clear()
        self.doc_id_to_index.clear()
        self._initialize_faiss_index()
        
        # Clear GPU memory
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        logger.info("Dense index cleared")
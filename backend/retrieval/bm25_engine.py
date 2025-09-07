"""
Advanced BM25 retrieval engine with optimizations
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import math
import re
from rank_bm25 import BM25Okapi
# Commented out to avoid scipy memory issues
# from sklearn.feature_extraction.text import HashingVectorizer
# from scipy.sparse import csr_matrix, save_npz, load_npz
# import joblib

logger = logging.getLogger(__name__)

@dataclass
class BM25Document:
    """Document representation for BM25 index"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    tokens: List[str]
    
@dataclass
class BM25SearchResult:
    """BM25 search result"""
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    highlighted_content: str = ""

class AdvancedBM25Engine:
    """
    Advanced BM25 retrieval engine with optimizations:
    - Optimized parameters (k1, b)
    - Vocabulary pruning
    - Memory-efficient storage
    - Incremental indexing support
    """
    
    def __init__(
        self,
        k1: float = 1.6,
        b: float = 0.75,
        epsilon: float = 0.25,
        min_doc_freq: int = 3,
        max_doc_freq: float = 0.8,
        use_stemming: bool = True,
        use_stopwords: bool = True,
        hash_features: int = 262144
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.min_doc_freq = min_doc_freq
        self.max_doc_freq = max_doc_freq
        self.use_stemming = use_stemming
        self.use_stopwords = use_stopwords
        self.hash_features = hash_features
        
        # Initialize components
        self.bm25 = None
        self.documents = []
        self.doc_id_to_index = {}
        self.vocabulary = set()
        self.document_frequencies = defaultdict(int)
        self.total_documents = 0
        
        # Text preprocessing
        self.stop_words = self._load_stopwords() if use_stopwords else set()
        self.stemmer = self._load_stemmer() if use_stemming else None
        
        # Vectorizer for feature hashing - commented out to avoid scipy issues
        # self.vectorizer = HashingVectorizer(
        #     n_features=hash_features,
        #     lowercase=True,
        #     stop_words='english' if use_stopwords else None,
        #     ngram_range=(1, 2),  # Unigrams and bigrams
        #     dtype=np.float16  # Memory optimization
        # )
        self.vectorizer = None
        
    def _load_stopwords(self) -> set:
        """Load stopwords"""
        try:
            import nltk
            from nltk.corpus import stopwords
            nltk.download('stopwords', quiet=True)
            return set(stopwords.words('english'))
        except ImportError:
            # Fallback stopwords
            return {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those'
            }
    
    def _load_stemmer(self):
        """Load stemmer"""
        try:
            import nltk
            from nltk.stem import PorterStemmer
            nltk.download('punkt', quiet=True)
            return PorterStemmer()
        except ImportError:
            logger.warning("NLTK not available for stemming")
            return None
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into tokens"""
        # Basic cleaning
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Stem tokens
        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Filter by length
        tokens = [token for token in tokens if 2 <= len(token) <= 20]
        
        return tokens
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add a document to the index"""
        if metadata is None:
            metadata = {}
        
        # Preprocess content
        tokens = self._preprocess_text(content)
        
        if not tokens:
            logger.warning(f"No valid tokens found for document {doc_id}")
            return
        
        # Create document
        document = BM25Document(
            doc_id=doc_id,
            content=content,
            metadata=metadata,
            tokens=tokens
        )
        
        # Add to collection
        doc_index = len(self.documents)
        self.documents.append(document)
        self.doc_id_to_index[doc_id] = doc_index
        
        # Update vocabulary and document frequencies
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.vocabulary.add(token)
            self.document_frequencies[token] += 1
        
        self.total_documents += 1
        
        logger.debug(f"Added document {doc_id} with {len(tokens)} tokens")
    
    def build_index(self):
        """Build the BM25 index"""
        if not self.documents:
            raise ValueError("No documents to index")
        
        logger.info(f"Building BM25 index for {len(self.documents)} documents")
        
        # Apply vocabulary pruning
        self._prune_vocabulary()
        
        # Prepare tokenized documents for BM25
        tokenized_docs = []
        for doc in self.documents:
            # Filter tokens based on pruned vocabulary
            filtered_tokens = [token for token in doc.tokens if token in self.vocabulary]
            tokenized_docs.append(filtered_tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b, epsilon=self.epsilon)
        
        logger.info(f"BM25 index built with vocabulary size: {len(self.vocabulary)}")
    
    def _prune_vocabulary(self):
        """Prune vocabulary based on document frequency thresholds"""
        original_size = len(self.vocabulary)
        
        # Calculate thresholds
        min_docs = max(self.min_doc_freq, 1)
        max_docs = int(self.max_doc_freq * self.total_documents) if self.max_doc_freq < 1.0 else self.max_doc_freq
        
        # Filter vocabulary
        pruned_vocab = set()
        for token in self.vocabulary:
            freq = self.document_frequencies[token]
            if min_docs <= freq <= max_docs:
                pruned_vocab.add(token)
        
        self.vocabulary = pruned_vocab
        pruned_size = len(self.vocabulary)
        
        logger.info(f"Vocabulary pruned from {original_size} to {pruned_size} tokens "
                   f"(min_freq={min_docs}, max_freq={max_docs})")
    
    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        min_score: float = 0.0
    ) -> List[BM25SearchResult]:
        """Search documents using BM25"""
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Preprocess query
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            logger.warning("No valid tokens in query")
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                break
            
            doc = self.documents[idx]
            
            # Generate highlighted content
            highlighted_content = self._highlight_terms(doc.content, query_tokens)
            
            result = BM25SearchResult(
                doc_id=doc.doc_id,
                score=score,
                content=doc.content,
                metadata=doc.metadata,
                highlighted_content=highlighted_content
            )
            results.append(result)
        
        logger.debug(f"BM25 search returned {len(results)} results for query: {query}")
        return results
    
    def _highlight_terms(self, content: str, query_tokens: List[str]) -> str:
        """Highlight query terms in content"""
        if not query_tokens:
            return content
        
        # Create pattern for highlighting
        # Case-insensitive matching
        patterns = []
        for token in set(query_tokens):
            if len(token) > 2:  # Only highlight longer terms
                pattern = re.escape(token)
                patterns.append(f"({pattern})")
        
        if not patterns:
            return content
        
        combined_pattern = '|'.join(patterns)
        
        try:
            # Highlight matches with <mark> tags
            highlighted = re.sub(
                combined_pattern, 
                r'<mark>\1</mark>', 
                content, 
                flags=re.IGNORECASE
            )
            return highlighted
        except Exception as e:
            logger.warning(f"Error highlighting terms: {e}")
            return content
    
    def get_document_by_id(self, doc_id: str) -> Optional[BM25Document]:
        """Get document by ID"""
        if doc_id in self.doc_id_to_index:
            return self.documents[self.doc_id_to_index[doc_id]]
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_documents": len(self.documents),
            "vocabulary_size": len(self.vocabulary),
            "average_doc_length": np.mean([len(doc.tokens) for doc in self.documents]) if self.documents else 0,
            "index_built": self.bm25 is not None,
            "parameters": {
                "k1": self.k1,
                "b": self.b,
                "epsilon": self.epsilon,
                "min_doc_freq": self.min_doc_freq,
                "max_doc_freq": self.max_doc_freq
            }
        }
    
    def save_index(self, index_path: str):
        """Save BM25 index to disk"""
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'bm25': self.bm25,
            'documents': self.documents,
            'doc_id_to_index': self.doc_id_to_index,
            'vocabulary': self.vocabulary,
            'document_frequencies': dict(self.document_frequencies),
            'total_documents': self.total_documents,
            'parameters': {
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon,
                'min_doc_freq': self.min_doc_freq,
                'max_doc_freq': self.max_doc_freq
            }
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"BM25 index saved to {index_path}")
    
    def load_index(self, index_path: str):
        """Load BM25 index from disk"""
        index_path = Path(index_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.bm25 = index_data['bm25']
        self.documents = index_data['documents']
        self.doc_id_to_index = index_data['doc_id_to_index']
        self.vocabulary = index_data['vocabulary']
        self.document_frequencies = defaultdict(int, index_data['document_frequencies'])
        self.total_documents = index_data['total_documents']
        
        # Update parameters
        params = index_data.get('parameters', {})
        self.k1 = params.get('k1', self.k1)
        self.b = params.get('b', self.b)
        self.epsilon = params.get('epsilon', self.epsilon)
        
        logger.info(f"BM25 index loaded from {index_path}")
    
    def clear_index(self):
        """Clear the current index"""
        self.bm25 = None
        self.documents.clear()
        self.doc_id_to_index.clear()
        self.vocabulary.clear()
        self.document_frequencies.clear()
        self.total_documents = 0
        
        logger.info("BM25 index cleared")
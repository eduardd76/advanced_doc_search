"""
Simple retrieval engine using only BM25 to avoid memory issues
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import math
import re
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

@dataclass
class SimpleDocument:
    """Document representation for simple index"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    tokens: List[str]
    
@dataclass
class SimpleSearchResult:
    """Simple search result"""
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    highlighted_content: str = ""

class SimpleRetrievalEngine:
    """
    Simple retrieval engine using only BM25 - no heavy ML libraries
    """
    
    def __init__(
        self,
        k1: float = 1.6,
        b: float = 0.75,
        epsilon: float = 0.25,
        min_doc_freq: int = 3,
        max_doc_freq: float = 0.8,
        use_stopwords: bool = True
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.min_doc_freq = min_doc_freq
        self.max_doc_freq = max_doc_freq
        self.use_stopwords = use_stopwords
        
        # Initialize components
        self.bm25 = None
        self.documents = []
        self.doc_id_to_index = {}
        self.vocabulary = set()
        self.document_frequencies = defaultdict(int)
        self.total_documents = 0
        
        # Text preprocessing
        self.stop_words = self._load_stopwords() if use_stopwords else set()
        
    def _load_stopwords(self) -> set:
        """Load basic stopwords"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those'
        }
    
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
        document = SimpleDocument(
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
        
        logger.info(f"Building simple BM25 index for {len(self.documents)} documents")
        
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
        
        logger.info(f"Simple BM25 index built with vocabulary size: {len(self.vocabulary)}")
    
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
    ) -> List[SimpleSearchResult]:
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
            
            result = SimpleSearchResult(
                doc_id=doc.doc_id,
                score=score,
                content=doc.content,
                metadata=doc.metadata,
                highlighted_content=highlighted_content
            )
            results.append(result)
        
        logger.debug(f"Simple BM25 search returned {len(results)} results for query: {query}")
        return results
    
    def _highlight_terms(self, content: str, query_tokens: List[str]) -> str:
        """Highlight query terms in content"""
        if not query_tokens:
            return content
        
        # Create pattern for highlighting
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
    
    def get_document_by_id(self, doc_id: str) -> Optional[SimpleDocument]:
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
    
    def clear_index(self):
        """Clear the current index"""
        self.bm25 = None
        self.documents.clear()
        self.doc_id_to_index.clear()
        self.vocabulary.clear()
        self.document_frequencies.clear()
        self.total_documents = 0
        
        logger.info("Simple BM25 index cleared")
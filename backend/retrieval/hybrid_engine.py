"""
Hybrid retrieval engine combining BM25, dense retrieval, and reranking with RRF fusion
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import math

from .bm25_engine import AdvancedBM25Engine, BM25SearchResult
from .dense_engine import DenseRetrievalEngine, DenseSearchResult
from .reranker import CrossEncoderReranker, RerankResult

logger = logging.getLogger(__name__)

@dataclass
class HybridSearchResult:
    """Hybrid search result combining multiple retrieval methods"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    final_score: float
    component_scores: Dict[str, float]
    rank_info: Dict[str, int]

class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) implementation
    Combines rankings from multiple retrieval systems
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF
        
        Args:
            k: RRF parameter (typically 60)
        """
        self.k = k
    
    def fuse_rankings(
        self,
        rankings_dict: Dict[str, List[Tuple[str, float]]],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple rankings using RRF
        
        Args:
            rankings_dict: Dict of {method_name: [(doc_id, score), ...]}
            weights: Optional weights for each method
            
        Returns:
            List of (doc_id, fused_score) sorted by fused score
        """
        if not rankings_dict:
            return []
        
        if weights is None:
            weights = {method: 1.0 for method in rankings_dict.keys()}
        
        # Collect all unique document IDs
        all_doc_ids = set()
        for rankings in rankings_dict.values():
            all_doc_ids.update(doc_id for doc_id, _ in rankings)
        
        # Calculate RRF scores for each document
        fused_scores = defaultdict(float)
        
        for method, rankings in rankings_dict.items():
            weight = weights.get(method, 1.0)
            
            # Create rank mapping
            rank_map = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(rankings)}
            
            # Calculate RRF contribution for each document
            for doc_id in all_doc_ids:
                if doc_id in rank_map:
                    rank = rank_map[doc_id]
                    rrf_score = weight / (self.k + rank)
                else:
                    # Document not found in this ranking - assign lowest score
                    rrf_score = weight / (self.k + len(rankings) + 1)
                
                fused_scores[doc_id] += rrf_score
        
        # Sort by fused score (descending)
        fused_rankings = sorted(
            fused_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return fused_rankings

class HybridRetrievalEngine:
    """
    State-of-the-art hybrid retrieval engine combining:
    1. BM25 sparse retrieval
    2. Dense embedding retrieval
    3. Reciprocal Rank Fusion
    4. Cross-encoder reranking
    """
    
    def __init__(
        self,
        bm25_config: Dict[str, Any] = None,
        dense_config: Dict[str, Any] = None,
        reranker_config: Dict[str, Any] = None,
        hybrid_config: Dict[str, Any] = None
    ):
        # Default configurations
        if bm25_config is None:
            bm25_config = {}
        if dense_config is None:
            dense_config = {}
        if reranker_config is None:
            reranker_config = {}
        if hybrid_config is None:
            hybrid_config = {}
        
        # Initialize retrieval engines
        self.bm25_engine = AdvancedBM25Engine(**bm25_config)
        self.dense_engine = DenseRetrievalEngine(**dense_config)
        self.reranker = CrossEncoderReranker(**reranker_config)
        
        # RRF fusion
        self.rrf = ReciprocalRankFusion(k=hybrid_config.get('rrv_k', 60))
        
        # Hybrid configuration
        self.use_rrv_fusion = hybrid_config.get('use_rrv_fusion', True)
        self.bm25_weight = hybrid_config.get('bm25_weight', 0.5)
        self.dense_weight = hybrid_config.get('dense_weight', 0.5)
        self.final_rerank_size = hybrid_config.get('final_rerank_size', 50)
        self.min_score_threshold = hybrid_config.get('min_score_threshold', 0.1)
        self.use_cross_encoder = hybrid_config.get('use_cross_encoder', True)
        
        # Document storage
        self.documents = {}  # doc_id -> document info
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add document to both sparse and dense indexes"""
        if metadata is None:
            metadata = {}
        
        # Store document info
        self.documents[doc_id] = {
            'content': content,
            'metadata': metadata
        }
        
        # Add to both engines
        self.bm25_engine.add_document(doc_id, content, metadata)
        self.dense_engine.add_document(doc_id, content, metadata)
        
        logger.debug(f"Added document {doc_id} to hybrid index")
    
    def build_index(self, show_progress: bool = True):
        """Build both sparse and dense indexes"""
        logger.info("Building hybrid index (BM25 + Dense)")
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25_engine.build_index()
        
        # Build dense index
        logger.info("Building dense index...")
        self.dense_engine.build_index(show_progress=show_progress)
        
        logger.info("Hybrid index building completed")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        bm25_candidates: int = 100,
        dense_candidates: int = 100,
        use_reranking: Optional[bool] = None,
        show_progress: bool = False
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search with RRF fusion and optional reranking
        
        Args:
            query: Search query
            top_k: Number of final results
            bm25_candidates: Number of BM25 candidates to retrieve
            dense_candidates: Number of dense candidates to retrieve
            use_reranking: Whether to use cross-encoder reranking
            show_progress: Show progress bars
            
        Returns:
            List of hybrid search results
        """
        if use_reranking is None:
            use_reranking = self.use_cross_encoder
        
        logger.debug(f"Hybrid search for query: {query}")
        
        # Step 1: Retrieve candidates from both engines
        bm25_results = self.bm25_engine.search(
            query, 
            top_k=bm25_candidates,
            min_score=0.0
        )
        
        dense_results = self.dense_engine.search(
            query,
            top_k=dense_candidates,
            min_score=0.0
        )
        
        logger.debug(f"Retrieved {len(bm25_results)} BM25 and {len(dense_results)} dense candidates")
        
        # Step 2: Apply RRF fusion if enabled
        if self.use_rrv_fusion:
            fused_results = self._apply_rrv_fusion(query, bm25_results, dense_results)
        else:
            fused_results = self._apply_weighted_fusion(bm25_results, dense_results)
        
        # Step 3: Apply cross-encoder reranking if enabled
        if use_reranking and len(fused_results) > 1:
            final_results = self._apply_reranking(query, fused_results, show_progress)
        else:
            final_results = fused_results
        
        # Step 4: Convert to HybridSearchResult format
        hybrid_results = self._convert_to_hybrid_results(final_results)
        
        # Step 5: Apply final filtering and limiting
        filtered_results = [
            result for result in hybrid_results
            if result.final_score >= self.min_score_threshold
        ]
        
        return filtered_results[:top_k]
    
    def _apply_rrv_fusion(
        self,
        query: str,
        bm25_results: List[BM25SearchResult],
        dense_results: List[DenseSearchResult]
    ) -> List[Dict[str, Any]]:
        """Apply Reciprocal Rank Fusion"""
        # Prepare rankings
        bm25_rankings = [(result.doc_id, result.score) for result in bm25_results]
        dense_rankings = [(result.doc_id, result.score) for result in dense_results]
        
        rankings_dict = {
            'bm25': bm25_rankings,
            'dense': dense_rankings
        }
        
        weights = {
            'bm25': self.bm25_weight,
            'dense': self.dense_weight
        }
        
        # Apply RRF fusion
        fused_rankings = self.rrf.fuse_rankings(rankings_dict, weights)
        
        # Create result objects
        fused_results = []
        for doc_id, fused_score in fused_rankings:
            if doc_id in self.documents:
                doc_info = self.documents[doc_id]
                
                # Find original scores
                bm25_score = next((r.score for r in bm25_results if r.doc_id == doc_id), 0.0)
                dense_score = next((r.score for r in dense_results if r.doc_id == doc_id), 0.0)
                
                result = {
                    'doc_id': doc_id,
                    'content': doc_info['content'],
                    'metadata': doc_info['metadata'],
                    'score': fused_score,
                    'bm25_score': bm25_score,
                    'dense_score': dense_score,
                    'fusion_method': 'rrv'
                }
                fused_results.append(result)
        
        logger.debug(f"RRF fusion produced {len(fused_results)} results")
        return fused_results
    
    def _apply_weighted_fusion(
        self,
        bm25_results: List[BM25SearchResult],
        dense_results: List[DenseSearchResult]
    ) -> List[Dict[str, Any]]:
        """Apply simple weighted score fusion"""
        # Normalize scores
        def normalize_scores(scores):
            if len(set(scores)) <= 1:
                return scores
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            return [(score - min_score) / score_range for score in scores]
        
        # Get all scores for normalization
        bm25_scores = [r.score for r in bm25_results] if bm25_results else [0.0]
        dense_scores = [r.score for r in dense_results] if dense_results else [0.0]
        
        bm25_scores_norm = normalize_scores(bm25_scores)
        dense_scores_norm = normalize_scores(dense_scores)
        
        # Create score maps
        bm25_score_map = {r.doc_id: (r.score, norm_score) for r, norm_score in zip(bm25_results, bm25_scores_norm)}
        dense_score_map = {r.doc_id: (r.score, norm_score) for r, norm_score in zip(dense_results, dense_scores_norm)}
        
        # Collect all unique documents
        all_doc_ids = set()
        all_doc_ids.update(bm25_score_map.keys())
        all_doc_ids.update(dense_score_map.keys())
        
        # Calculate weighted scores
        weighted_results = []
        for doc_id in all_doc_ids:
            if doc_id not in self.documents:
                continue
            
            bm25_orig, bm25_norm = bm25_score_map.get(doc_id, (0.0, 0.0))
            dense_orig, dense_norm = dense_score_map.get(doc_id, (0.0, 0.0))
            
            weighted_score = (
                self.bm25_weight * bm25_norm +
                self.dense_weight * dense_norm
            )
            
            doc_info = self.documents[doc_id]
            result = {
                'doc_id': doc_id,
                'content': doc_info['content'],
                'metadata': doc_info['metadata'],
                'score': weighted_score,
                'bm25_score': bm25_orig,
                'dense_score': dense_orig,
                'fusion_method': 'weighted'
            }
            weighted_results.append(result)
        
        # Sort by weighted score
        weighted_results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.debug(f"Weighted fusion produced {len(weighted_results)} results")
        return weighted_results
    
    def _apply_reranking(
        self,
        query: str,
        fused_results: List[Dict[str, Any]],
        show_progress: bool = False
    ) -> List[Dict[str, Any]]:
        """Apply cross-encoder reranking"""
        # Limit candidates for reranking
        candidates_for_rerank = fused_results[:self.final_rerank_size]
        
        if len(candidates_for_rerank) <= 1:
            return fused_results
        
        # Rerank using cross-encoder
        rerank_results = self.reranker.rerank(
            query=query,
            candidates=candidates_for_rerank,
            score_field='score',
            content_field='content',
            top_k=None,
            show_progress=show_progress
        )
        
        # Convert back to dict format
        reranked_dicts = []
        for result in rerank_results:
            original_result = next(
                (r for r in candidates_for_rerank if r['doc_id'] == result.doc_id),
                None
            )
            
            if original_result:
                reranked_dict = {
                    **original_result,
                    'score': result.rerank_score,
                    'original_fusion_score': original_result['score'],
                    'cross_encoder_score': result.rerank_score,
                    'rank_change': result.rank_change
                }
                reranked_dicts.append(reranked_dict)
        
        # Add any remaining results that weren't reranked
        remaining_results = fused_results[self.final_rerank_size:]
        reranked_dicts.extend(remaining_results)
        
        logger.debug(f"Cross-encoder reranking processed {len(rerank_results)} candidates")
        return reranked_dicts
    
    def _convert_to_hybrid_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[HybridSearchResult]:
        """Convert internal results to HybridSearchResult format"""
        hybrid_results = []
        
        for result in results:
            # Extract component scores
            component_scores = {
                'bm25': result.get('bm25_score', 0.0),
                'dense': result.get('dense_score', 0.0),
                'fusion': result.get('original_fusion_score', result.get('score', 0.0)),
                'cross_encoder': result.get('cross_encoder_score', 0.0)
            }
            
            # Extract rank information
            rank_info = {
                'rank_change': result.get('rank_change', 0),
                'fusion_method': result.get('fusion_method', 'unknown')
            }
            
            hybrid_result = HybridSearchResult(
                doc_id=result['doc_id'],
                content=result['content'],
                metadata=result['metadata'],
                final_score=result['score'],
                component_scores=component_scores,
                rank_info=rank_info
            )
            hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components"""
        return {
            "total_documents": len(self.documents),
            "bm25_stats": self.bm25_engine.get_statistics(),
            "dense_stats": self.dense_engine.get_statistics(),
            "reranker_stats": self.reranker.get_statistics(),
            "hybrid_config": {
                "use_rrv_fusion": self.use_rrv_fusion,
                "bm25_weight": self.bm25_weight,
                "dense_weight": self.dense_weight,
                "final_rerank_size": self.final_rerank_size,
                "min_score_threshold": self.min_score_threshold,
                "use_cross_encoder": self.use_cross_encoder,
                "rrv_k": self.rrf.k
            }
        }
    
    def save_index(self, base_path: str):
        """Save all component indexes"""
        from pathlib import Path
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save individual components
        self.bm25_engine.save_index(str(base_path) + "_bm25.pkl")
        self.dense_engine.save_index(str(base_path) + "_dense")
        
        # Save document storage and hybrid config
        import pickle
        hybrid_data = {
            'documents': self.documents,
            'hybrid_config': {
                'use_rrv_fusion': self.use_rrv_fusion,
                'bm25_weight': self.bm25_weight,
                'dense_weight': self.dense_weight,
                'final_rerank_size': self.final_rerank_size,
                'min_score_threshold': self.min_score_threshold,
                'use_cross_encoder': self.use_cross_encoder,
                'rrv_k': self.rrf.k
            }
        }
        
        with open(str(base_path) + "_hybrid.pkl", 'wb') as f:
            pickle.dump(hybrid_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Hybrid index saved to {base_path}")
    
    def load_index(self, base_path: str):
        """Load all component indexes"""
        from pathlib import Path
        import pickle
        
        base_path = Path(base_path)
        
        # Load individual components
        self.bm25_engine.load_index(str(base_path) + "_bm25.pkl")
        self.dense_engine.load_index(str(base_path) + "_dense")
        
        # Load hybrid data
        hybrid_file = str(base_path) + "_hybrid.pkl"
        if Path(hybrid_file).exists():
            with open(hybrid_file, 'rb') as f:
                hybrid_data = pickle.load(f)
            
            self.documents = hybrid_data['documents']
            
            # Update hybrid config
            config = hybrid_data.get('hybrid_config', {})
            self.use_rrv_fusion = config.get('use_rrv_fusion', self.use_rrv_fusion)
            self.bm25_weight = config.get('bm25_weight', self.bm25_weight)
            self.dense_weight = config.get('dense_weight', self.dense_weight)
            self.final_rerank_size = config.get('final_rerank_size', self.final_rerank_size)
            self.min_score_threshold = config.get('min_score_threshold', self.min_score_threshold)
            self.use_cross_encoder = config.get('use_cross_encoder', self.use_cross_encoder)
            self.rrf.k = config.get('rrv_k', self.rrf.k)
        
        logger.info(f"Hybrid index loaded from {base_path}")
    
    def clear_index(self):
        """Clear all indexes"""
        self.bm25_engine.clear_index()
        self.dense_engine.clear_index()
        self.documents.clear()
        
        logger.info("Hybrid index cleared")
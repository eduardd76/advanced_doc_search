"""
Cross-encoder reranking system for improved relevance
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class RerankResult:
    """Reranking result"""
    doc_id: str
    original_score: float
    rerank_score: float
    content: str
    metadata: Dict[str, Any]
    rank_change: int = 0  # Change in ranking position

class CrossEncoderReranker:
    """
    Cross-encoder reranking system:
    - Reranks top candidates from sparse/dense retrieval
    - Uses query-document pairs for better relevance
    - Supports batch processing
    - Provides ranking change analysis
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_candidates: int = 100,
        batch_size: int = 16,
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.max_candidates = max_candidates
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = cache_dir
        
        # Initialize model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load cross-encoder model"""
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512  # Max token length for input pairs
            )
            
            logger.info("Cross-encoder model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading cross-encoder model: {e}")
            raise
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        score_field: str = "score",
        content_field: str = "content",
        top_k: Optional[int] = None,
        show_progress: bool = False
    ) -> List[RerankResult]:
        """
        Rerank candidates using cross-encoder
        
        Args:
            query: Search query
            candidates: List of candidate documents with scores
            score_field: Field name containing original scores
            content_field: Field name containing document content
            top_k: Number of top results to return
            show_progress: Show progress bar
            
        Returns:
            List of reranked results
        """
        if not candidates:
            return []
        
        if not query.strip():
            logger.warning("Empty query for reranking")
            return []
        
        # Limit candidates
        candidates = candidates[:self.max_candidates]
        
        if len(candidates) == 1:
            # No need to rerank single candidate
            candidate = candidates[0]
            return [RerankResult(
                doc_id=candidate.get('doc_id', str(0)),
                original_score=candidate.get(score_field, 0.0),
                rerank_score=candidate.get(score_field, 0.0),
                content=candidate.get(content_field, ''),
                metadata=candidate.get('metadata', {}),
                rank_change=0
            )]
        
        try:
            # Prepare query-document pairs
            pairs = []
            for candidate in candidates:
                content = candidate.get(content_field, '')
                if content:
                    # Truncate content if too long
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    pairs.append([query, content])
                else:
                    # Empty content gets lowest score
                    pairs.append([query, ""])
            
            # Predict relevance scores in batches
            all_scores = []
            
            for i in tqdm(range(0, len(pairs), self.batch_size), 
                         desc="Reranking", 
                         disable=not show_progress):
                batch_pairs = pairs[i:i + self.batch_size]
                
                # Get cross-encoder scores
                batch_scores = self.model.predict(batch_pairs, show_progress_bar=False)
                all_scores.extend(batch_scores)
                
                # Clear GPU memory if using CUDA
                if self.device.startswith('cuda'):
                    torch.cuda.empty_cache()
            
            # Create rerank results
            rerank_results = []
            for i, (candidate, rerank_score) in enumerate(zip(candidates, all_scores)):
                result = RerankResult(
                    doc_id=candidate.get('doc_id', str(i)),
                    original_score=candidate.get(score_field, 0.0),
                    rerank_score=float(rerank_score),
                    content=candidate.get(content_field, ''),
                    metadata=candidate.get('metadata', {}),
                    rank_change=0  # Will be calculated after sorting
                )
                rerank_results.append(result)
            
            # Sort by rerank score (descending)
            rerank_results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Calculate rank changes
            for new_rank, result in enumerate(rerank_results):
                # Find original rank
                original_rank = next(
                    i for i, candidate in enumerate(candidates) 
                    if candidate.get('doc_id', str(i)) == result.doc_id
                )
                result.rank_change = original_rank - new_rank
            
            # Return top_k results
            if top_k is not None:
                rerank_results = rerank_results[:top_k]
            
            logger.debug(f"Reranked {len(candidates)} candidates, returning {len(rerank_results)}")
            return rerank_results
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Return original results as fallback
            fallback_results = []
            for i, candidate in enumerate(candidates):
                result = RerankResult(
                    doc_id=candidate.get('doc_id', str(i)),
                    original_score=candidate.get(score_field, 0.0),
                    rerank_score=candidate.get(score_field, 0.0),
                    content=candidate.get(content_field, ''),
                    metadata=candidate.get('metadata', {}),
                    rank_change=0
                )
                fallback_results.append(result)
            
            return fallback_results[:top_k] if top_k else fallback_results
    
    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair"""
        if not query.strip() or not document.strip():
            return 0.0
        
        try:
            # Truncate document if too long
            if len(document) > 1000:
                document = document[:1000] + "..."
            
            score = self.model.predict([[query, document]], show_progress_bar=False)[0]
            return float(score)
            
        except Exception as e:
            logger.error(f"Error scoring pair: {e}")
            return 0.0
    
    def get_rank_improvements(self, results: List[RerankResult]) -> Dict[str, Any]:
        """Analyze ranking improvements after reranking"""
        if not results:
            return {}
        
        rank_changes = [result.rank_change for result in results]
        improved_count = sum(1 for change in rank_changes if change > 0)
        degraded_count = sum(1 for change in rank_changes if change < 0)
        unchanged_count = sum(1 for change in rank_changes if change == 0)
        
        avg_rank_change = np.mean(rank_changes)
        max_improvement = max(rank_changes) if rank_changes else 0
        max_degradation = min(rank_changes) if rank_changes else 0
        
        return {
            "total_results": len(results),
            "improved": improved_count,
            "degraded": degraded_count,
            "unchanged": unchanged_count,
            "improvement_rate": improved_count / len(results) if results else 0.0,
            "avg_rank_change": float(avg_rank_change),
            "max_improvement": max_improvement,
            "max_degradation": max_degradation,
            "score_range": {
                "min": min(r.rerank_score for r in results),
                "max": max(r.rerank_score for r in results),
                "mean": np.mean([r.rerank_score for r in results])
            }
        }
    
    def batch_score_pairs(
        self, 
        query_doc_pairs: List[Tuple[str, str]], 
        show_progress: bool = False
    ) -> List[float]:
        """Score multiple query-document pairs in batches"""
        if not query_doc_pairs:
            return []
        
        try:
            # Prepare pairs and truncate documents
            processed_pairs = []
            for query, document in query_doc_pairs:
                if len(document) > 1000:
                    document = document[:1000] + "..."
                processed_pairs.append([query, document])
            
            # Score in batches
            all_scores = []
            
            for i in tqdm(range(0, len(processed_pairs), self.batch_size),
                         desc="Scoring pairs",
                         disable=not show_progress):
                batch_pairs = processed_pairs[i:i + self.batch_size]
                batch_scores = self.model.predict(batch_pairs, show_progress_bar=False)
                all_scores.extend(batch_scores)
                
                # Clear GPU memory if using CUDA
                if self.device.startswith('cuda'):
                    torch.cuda.empty_cache()
            
            return [float(score) for score in all_scores]
            
        except Exception as e:
            logger.error(f"Error in batch scoring: {e}")
            return [0.0] * len(query_doc_pairs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reranker statistics"""
        return {
            "model_name": self.model_name,
            "max_candidates": self.max_candidates,
            "batch_size": self.batch_size,
            "device": self.device,
            "model_loaded": self.model is not None
        }

class HybridReranker:
    """
    Hybrid reranker that combines multiple ranking signals
    """
    
    def __init__(
        self,
        cross_encoder: CrossEncoderReranker,
        sparse_weight: float = 0.3,
        dense_weight: float = 0.3,
        cross_encoder_weight: float = 0.4
    ):
        self.cross_encoder = cross_encoder
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.cross_encoder_weight = cross_encoder_weight
        
        # Normalize weights
        total_weight = sparse_weight + dense_weight + cross_encoder_weight
        self.sparse_weight /= total_weight
        self.dense_weight /= total_weight
        self.cross_encoder_weight /= total_weight
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        sparse_scores: Optional[List[float]] = None,
        dense_scores: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        show_progress: bool = False
    ) -> List[RerankResult]:
        """
        Rerank using hybrid approach combining multiple signals
        """
        if not candidates:
            return []
        
        # Get cross-encoder scores
        ce_results = self.cross_encoder.rerank(
            query, candidates, top_k=None, show_progress=show_progress
        )
        
        # Extract cross-encoder scores
        cross_encoder_scores = [result.rerank_score for result in ce_results]
        
        # Normalize all scores to [0, 1]
        def normalize_scores(scores):
            if not scores or len(set(scores)) <= 1:
                return scores
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            return [(score - min_score) / score_range for score in scores]
        
        # Prepare score arrays
        if sparse_scores is None:
            sparse_scores = [c.get('bm25_score', c.get('score', 0.0)) for c in candidates]
        
        if dense_scores is None:
            dense_scores = [c.get('dense_score', c.get('score', 0.0)) for c in candidates]
        
        # Normalize scores
        sparse_scores_norm = normalize_scores(sparse_scores[:len(candidates)])
        dense_scores_norm = normalize_scores(dense_scores[:len(candidates)])
        cross_encoder_scores_norm = normalize_scores(cross_encoder_scores)
        
        # Combine scores
        hybrid_results = []
        for i, (result, sparse_norm, dense_norm, ce_norm) in enumerate(zip(
            ce_results, sparse_scores_norm, dense_scores_norm, cross_encoder_scores_norm
        )):
            hybrid_score = (
                self.sparse_weight * sparse_norm +
                self.dense_weight * dense_norm +
                self.cross_encoder_weight * ce_norm
            )
            
            hybrid_result = RerankResult(
                doc_id=result.doc_id,
                original_score=result.original_score,
                rerank_score=hybrid_score,
                content=result.content,
                metadata={
                    **result.metadata,
                    "hybrid_components": {
                        "sparse_score": sparse_scores[i] if i < len(sparse_scores) else 0.0,
                        "dense_score": dense_scores[i] if i < len(dense_scores) else 0.0,
                        "cross_encoder_score": result.rerank_score,
                        "hybrid_score": hybrid_score
                    }
                },
                rank_change=0
            )
            hybrid_results.append(hybrid_result)
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Calculate rank changes
        for new_rank, result in enumerate(hybrid_results):
            original_rank = next(
                i for i, candidate in enumerate(candidates)
                if candidate.get('doc_id', str(i)) == result.doc_id
            )
            result.rank_change = original_rank - new_rank
        
        # Return top_k results
        if top_k is not None:
            hybrid_results = hybrid_results[:top_k]
        
        return hybrid_results
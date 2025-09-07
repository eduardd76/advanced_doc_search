"""
Tests for Hybrid Retrieval Engine
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
import sys
import os

# Add backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from retrieval.hybrid_engine import HybridRetrievalEngine

class TestHybridRetrievalEngine:
    def setup_method(self):
        """Setup test fixtures"""
        # Use lightweight configs for testing
        self.bm25_config = {
            'k1': 1.2, 'b': 0.75, 'min_doc_freq': 1, 'max_doc_freq': 0.9
        }
        self.dense_config = {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'embedding_dim': 384, 'batch_size': 2, 'device': 'cpu'
        }
        self.reranker_config = {
            'model_name': 'cross-encoder/ms-marco-MiniLM-L-2-v2',
            'max_candidates': 10, 'batch_size': 2, 'device': 'cpu'
        }
        self.hybrid_config = {
            'use_rrv_fusion': True, 'rrv_k': 60, 'bm25_weight': 0.5,
            'dense_weight': 0.5, 'final_rerank_size': 5,
            'min_score_threshold': 0.0
        }
        
        self.engine = HybridRetrievalEngine(
            self.bm25_config, self.dense_config, 
            self.reranker_config, self.hybrid_config
        )
        
        # Sample documents for testing
        self.test_docs = [
            {
                'doc_id': 'doc1',
                'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
                'metadata': {'category': 'AI', 'length': 'short'}
            },
            {
                'doc_id': 'doc2', 
                'content': 'Natural language processing enables computers to understand and generate human language.',
                'metadata': {'category': 'NLP', 'length': 'medium'}
            },
            {
                'doc_id': 'doc3',
                'content': 'Deep learning neural networks use multiple layers to learn complex patterns in data.',
                'metadata': {'category': 'Deep Learning', 'length': 'medium'}
            },
            {
                'doc_id': 'doc4',
                'content': 'Computer vision algorithms analyze visual content like images and videos.',
                'metadata': {'category': 'Computer Vision', 'length': 'short'}
            },
            {
                'doc_id': 'doc5',
                'content': 'Reinforcement learning trains agents to make decisions through interaction with environments.',
                'metadata': {'category': 'Reinforcement Learning', 'length': 'medium'}
            }
        ]

    def test_engine_initialization(self):
        """Test hybrid engine initialization"""
        assert self.engine.bm25_engine is not None
        assert self.engine.dense_engine is not None
        assert self.engine.reranker is not None
        assert self.engine.hybrid_config['use_rrv_fusion'] == True

    def test_add_documents(self):
        """Test adding documents to the hybrid index"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        
        assert len(self.engine.bm25_engine.documents) == 5
        assert len(self.engine.dense_engine.documents) == 5

    def test_build_index(self):
        """Test building the hybrid index"""
        # Add documents
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        
        # Build index
        self.engine.build_index()
        
        # Check that both engines are built
        assert self.engine.bm25_engine.bm25 is not None
        assert self.engine.dense_engine.faiss_index is not None
        assert self.engine.dense_engine.faiss_index.ntotal == 5

    @pytest.mark.slow
    def test_hybrid_search_with_reranking(self):
        """Test hybrid search with reranking enabled"""
        # Setup index
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        # Test search with reranking
        results = self.engine.search(
            query="machine learning algorithms",
            top_k=3,
            use_reranking=True
        )
        
        assert len(results) <= 3
        assert all(hasattr(result, 'doc_id') for result in results)
        assert all(hasattr(result, 'final_score') for result in results)
        assert all(hasattr(result, 'component_scores') for result in results)
        
        # Check that results have reranking scores
        for result in results:
            assert 'rerank_score' in result.component_scores

    def test_hybrid_search_without_reranking(self):
        """Test hybrid search without reranking"""
        # Setup index
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        # Test search without reranking
        results = self.engine.search(
            query="machine learning algorithms",
            top_k=3,
            use_reranking=False
        )
        
        assert len(results) <= 3
        assert all(hasattr(result, 'doc_id') for result in results)
        assert all(hasattr(result, 'final_score') for result in results)
        
        # Check that results don't have reranking scores
        for result in results:
            assert 'rerank_score' not in result.component_scores

    def test_rrv_fusion(self):
        """Test RRF (Reciprocal Rank Fusion) functionality"""
        # Setup index
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        # Get BM25 and dense results separately
        bm25_results = self.engine.bm25_engine.search("machine learning", top_k=5)
        dense_results = self.engine.dense_engine.search("machine learning", top_k=5)
        
        # Apply RRF fusion
        fused_results = self.engine._apply_rrv_fusion(
            "machine learning", bm25_results, dense_results
        )
        
        assert len(fused_results) <= 5
        assert all(hasattr(result, 'final_score') for result in fused_results)
        assert all(hasattr(result, 'component_scores') for result in fused_results)
        
        # Check that component scores include both BM25 and dense
        for result in fused_results:
            assert 'bm25_score' in result.component_scores
            assert 'dense_score' in result.component_scores

    def test_different_candidate_counts(self):
        """Test search with different candidate counts"""
        # Setup index
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        # Test with different candidate counts
        results1 = self.engine.search("machine learning", bm25_candidates=3, dense_candidates=3, top_k=2)
        results2 = self.engine.search("machine learning", bm25_candidates=5, dense_candidates=5, top_k=2)
        
        assert len(results1) <= 2
        assert len(results2) <= 2

    def test_save_load_index(self):
        """Test saving and loading hybrid index"""
        # Setup and build index
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        # Test search before save
        results_before = self.engine.search("machine learning", top_k=3)
        
        # Save index
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test_hybrid_index"
            self.engine.save_index(str(index_path))
            
            # Create new engine and load
            new_engine = HybridRetrievalEngine(
                self.bm25_config, self.dense_config,
                self.reranker_config, self.hybrid_config
            )
            new_engine.load_index(str(index_path))
            
            # Test search after load
            results_after = new_engine.search("machine learning", top_k=3)
            
            assert len(results_before) == len(results_after)
            # Document IDs should be the same (order might differ due to slight score differences)
            before_ids = set(r.doc_id for r in results_before)
            after_ids = set(r.doc_id for r in results_after)
            assert before_ids == after_ids

    def test_clear_index(self):
        """Test clearing the hybrid index"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        assert len(self.engine.bm25_engine.documents) == 5
        assert len(self.engine.dense_engine.documents) == 5
        
        self.engine.clear_index()
        
        assert len(self.engine.bm25_engine.documents) == 0
        assert len(self.engine.dense_engine.documents) == 0

    def test_get_statistics(self):
        """Test getting hybrid engine statistics"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        stats = self.engine.get_statistics()
        
        assert 'bm25_stats' in stats
        assert 'dense_stats' in stats
        assert 'hybrid_config' in stats
        assert stats['bm25_stats']['total_documents'] == 5
        assert stats['dense_stats']['total_documents'] == 5

    def test_empty_query_handling(self):
        """Test handling of empty queries"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        results = self.engine.search("", top_k=5)
        assert len(results) == 0

    def test_component_score_tracking(self):
        """Test that component scores are properly tracked"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        results = self.engine.search("machine learning algorithms", top_k=3, use_reranking=True)
        
        for result in results:
            assert 'bm25_score' in result.component_scores
            assert 'dense_score' in result.component_scores
            assert 'rrv_score' in result.component_scores
            assert 'rerank_score' in result.component_scores
            
            # All scores should be numbers
            for score_name, score_value in result.component_scores.items():
                assert isinstance(score_value, (int, float, np.number))

    def test_ranking_information(self):
        """Test that ranking information is preserved"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        results = self.engine.search("machine learning", top_k=3)
        
        for result in results:
            assert hasattr(result, 'rank_info')
            assert isinstance(result.rank_info, dict)

    def test_metadata_preservation(self):
        """Test that metadata is preserved through the pipeline"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        results = self.engine.search("machine learning", top_k=3)
        
        for result in results:
            assert hasattr(result, 'metadata')
            assert 'category' in result.metadata
            assert 'length' in result.metadata

    def test_score_threshold_filtering(self):
        """Test score threshold filtering"""
        # Create engine with higher threshold
        hybrid_config_threshold = self.hybrid_config.copy()
        hybrid_config_threshold['min_score_threshold'] = 0.5
        
        engine_threshold = HybridRetrievalEngine(
            self.bm25_config, self.dense_config,
            self.reranker_config, hybrid_config_threshold
        )
        
        for doc in self.test_docs:
            engine_threshold.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        engine_threshold.build_index()
        
        # Search with a query that might have low scores
        results = engine_threshold.search("quantum physics", top_k=5)
        
        # All results should meet the threshold (or no results if all scores are too low)
        for result in results:
            assert result.final_score >= 0.5

if __name__ == '__main__':
    # Run with specific markers to control test execution
    pytest.main([__file__, '-v', '-m', 'not slow'])
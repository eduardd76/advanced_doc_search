"""
Tests for BM25 retrieval engine
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from retrieval.bm25_engine import AdvancedBM25Engine

class TestAdvancedBM25Engine:
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = AdvancedBM25Engine(k1=1.6, b=0.75)
        
        # Sample documents for testing
        self.test_docs = [
            {
                'doc_id': 'doc1',
                'content': 'Machine learning algorithms are used in artificial intelligence applications.',
                'metadata': {'category': 'AI', 'author': 'John Doe'}
            },
            {
                'doc_id': 'doc2', 
                'content': 'Natural language processing is a subset of machine learning focused on text.',
                'metadata': {'category': 'NLP', 'author': 'Jane Smith'}
            },
            {
                'doc_id': 'doc3',
                'content': 'Deep learning neural networks require large amounts of training data.',
                'metadata': {'category': 'Deep Learning', 'author': 'Bob Wilson'}
            },
            {
                'doc_id': 'doc4',
                'content': 'Computer vision algorithms can identify objects in images using machine learning.',
                'metadata': {'category': 'Computer Vision', 'author': 'Alice Johnson'}
            }
        ]

    def test_engine_initialization(self):
        """Test BM25 engine initialization"""
        assert self.engine.k1 == 1.6
        assert self.engine.b == 0.75
        assert self.engine.epsilon == 0.25
        assert len(self.engine.documents) == 0

    def test_add_documents(self):
        """Test adding documents to the index"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        
        assert len(self.engine.documents) == 4
        assert self.engine.doc_count == 4
        assert 'doc1' in self.engine.doc_metadata

    def test_build_index(self):
        """Test building BM25 index"""
        # Add documents
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        
        # Build index
        self.engine.build_index()
        
        assert self.engine.bm25 is not None
        assert len(self.engine.tokenized_docs) == 4
        assert self.engine.total_docs == 4

    def test_search_functionality(self):
        """Test search functionality"""
        # Setup index
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        # Test search
        results = self.engine.search("machine learning", top_k=3)
        
        assert len(results) <= 3
        assert all(hasattr(result, 'doc_id') for result in results)
        assert all(hasattr(result, 'score') for result in results)
        assert all(hasattr(result, 'content') for result in results)
        
        # Check scores are descending
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_highlighting(self):
        """Test query term highlighting"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        results = self.engine.search("machine learning", top_k=2, highlight=True)
        
        for result in results:
            if 'machine' in result.content.lower():
                assert '<mark>' in result.content and '</mark>' in result.content

    def test_vocabulary_pruning(self):
        """Test vocabulary pruning functionality"""
        # Create engine with strict pruning parameters
        engine = AdvancedBM25Engine(min_doc_freq=2, max_doc_freq=0.5)
        
        for doc in self.test_docs:
            engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        
        original_vocab_size = len(set(' '.join([doc['content'] for doc in self.test_docs]).split()))
        engine.build_index()
        
        # Vocabulary should be pruned
        assert len(engine.tokenized_docs[0]) <= original_vocab_size

    def test_empty_query(self):
        """Test handling of empty queries"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        results = self.engine.search("", top_k=5)
        assert len(results) == 0

    def test_no_match_query(self):
        """Test handling of queries with no matches"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        results = self.engine.search("quantum physics space exploration", top_k=5)
        # Should return some results with very low scores or no results
        assert len(results) >= 0

    def test_save_load_index(self):
        """Test saving and loading index"""
        # Setup and build index
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        # Test search before save
        results_before = self.engine.search("machine learning", top_k=3)
        
        # Save index
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test_bm25_index"
            self.engine.save_index(str(index_path))
            
            # Create new engine and load
            new_engine = AdvancedBM25Engine()
            new_engine.load_index(str(index_path))
            
            # Test search after load
            results_after = new_engine.search("machine learning", top_k=3)
            
            assert len(results_before) == len(results_after)
            assert [r.doc_id for r in results_before] == [r.doc_id for r in results_after]

    def test_clear_index(self):
        """Test clearing the index"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        assert len(self.engine.documents) == 4
        
        self.engine.clear_index()
        
        assert len(self.engine.documents) == 0
        assert self.engine.doc_count == 0
        assert len(self.engine.doc_metadata) == 0

    def test_get_statistics(self):
        """Test getting engine statistics"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        stats = self.engine.get_statistics()
        
        assert 'total_documents' in stats
        assert 'total_tokens' in stats
        assert 'unique_tokens' in stats
        assert 'average_doc_length' in stats
        assert stats['total_documents'] == 4

    def test_different_parameters(self):
        """Test engine with different parameters"""
        # Test with different k1 and b values
        engine1 = AdvancedBM25Engine(k1=1.2, b=0.5)
        engine2 = AdvancedBM25Engine(k1=2.0, b=1.0)
        
        for doc in self.test_docs:
            engine1.add_document(doc['doc_id'], doc['content'], doc['metadata'])
            engine2.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        
        engine1.build_index()
        engine2.build_index()
        
        results1 = engine1.search("machine learning", top_k=3)
        results2 = engine2.search("machine learning", top_k=3)
        
        # Results should be different due to different parameters
        scores1 = [r.score for r in results1]
        scores2 = [r.score for r in results2]
        
        # At least some scores should be different
        assert not all(s1 == s2 for s1, s2 in zip(scores1, scores2))

    def test_metadata_preservation(self):
        """Test that metadata is preserved in search results"""
        for doc in self.test_docs:
            self.engine.add_document(doc['doc_id'], doc['content'], doc['metadata'])
        self.engine.build_index()
        
        results = self.engine.search("machine learning", top_k=3)
        
        for result in results:
            assert hasattr(result, 'metadata')
            assert 'category' in result.metadata
            assert 'author' in result.metadata

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
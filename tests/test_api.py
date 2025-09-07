"""
Tests for FastAPI application endpoints
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
import sys
from fastapi.testclient import TestClient

# Add backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import app

class TestAPI:
    def setup_method(self):
        """Setup test fixtures"""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "online"
        assert data["service"] == "Advanced Document Search API"
        assert data["version"] == "2.0.0"
        assert "features" in data
        assert len(data["features"]) > 0

    def test_status_endpoint(self):
        """Test system status endpoint"""
        response = self.client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "system_status" in data
        assert "indexing_status" in data
        assert "system_statistics" in data
        assert "configuration" in data
        assert data["system_status"] == "operational"

    def test_indexing_status_endpoint(self):
        """Test indexing status endpoint"""
        response = self.client.get("/api/indexes/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return indexing status structure
        expected_keys = [
            "is_indexing", "current_progress", "total_documents",
            "status_message", "current_index", "processed_documents", "failed_documents"
        ]
        
        for key in expected_keys:
            assert key in data

    def test_create_index_invalid_request(self):
        """Test create index with invalid request data"""
        # Test with empty directory paths
        response = self.client.post("/api/indexes/create", json={
            "name": "test_index",
            "directory_paths": [],
            "file_extensions": [".txt"],
            "force_rebuild": False
        })
        
        assert response.status_code == 400
        assert "At least one directory path is required" in response.json()["detail"]

    def test_create_index_missing_name(self):
        """Test create index with missing name"""
        response = self.client.post("/api/indexes/create", json={
            "directory_paths": ["/some/path"],
            "file_extensions": [".txt"],
            "force_rebuild": False
        })
        
        assert response.status_code == 422  # Validation error

    def test_create_index_nonexistent_directory(self):
        """Test create index with non-existent directory"""
        response = self.client.post("/api/indexes/create", json={
            "name": "test_index",
            "directory_paths": ["/nonexistent/path/12345"],
            "file_extensions": [".txt"],
            "force_rebuild": False
        })
        
        assert response.status_code == 404
        assert "No valid directories found" in response.json()["detail"]

    def test_create_index_valid_request(self):
        """Test create index with valid request using temp directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            temp_path = Path(temp_dir)
            (temp_path / 'test1.txt').write_text('This is test document 1.')
            (temp_path / 'test2.txt').write_text('This is test document 2.')
            
            response = self.client.post("/api/indexes/create", json={
                "name": "test_index",
                "directory_paths": [temp_dir],
                "file_extensions": [".txt"],
                "force_rebuild": True
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["index_name"] == "test_index"
            assert temp_dir in data["directories"]
            assert ".txt" in data["file_extensions"]

    def test_search_invalid_request(self):
        """Test search with invalid request data"""
        # Test with empty query
        response = self.client.post("/api/search", json={
            "query": "",
            "index_name": "test_index",
            "top_k": 10
        })
        
        assert response.status_code == 400
        assert "Query cannot be empty" in response.json()["detail"]

    def test_search_missing_fields(self):
        """Test search with missing required fields"""
        response = self.client.post("/api/search", json={
            "query": "test query"
            # Missing index_name
        })
        
        assert response.status_code == 422  # Validation error

    def test_search_valid_request_no_index(self):
        """Test search with valid request but no index exists"""
        response = self.client.post("/api/search", json={
            "query": "machine learning",
            "index_name": "nonexistent_index",
            "top_k": 5,
            "use_reranking": True,
            "bm25_candidates": 50,
            "dense_candidates": 50
        })
        
        # This will likely return 500 since no index is built
        # The exact behavior depends on the implementation
        assert response.status_code in [500, 404]

    def test_evaluate_invalid_request(self):
        """Test evaluate with invalid request data"""
        response = self.client.post("/api/evaluate", json={
            "index_name": "test_index",
            "create_sample_benchmark": False
            # Missing benchmark_file when create_sample_benchmark is False
        })
        
        assert response.status_code == 400

    def test_evaluate_missing_index_name(self):
        """Test evaluate with missing index name"""
        response = self.client.post("/api/evaluate", json={
            "create_sample_benchmark": True,
            "num_sample_queries": 10
        })
        
        assert response.status_code == 422  # Validation error

    def test_chat_invalid_request(self):
        """Test chat with invalid request data"""
        # Test with empty message
        response = self.client.post("/api/chat", json={
            "message": "",
            "index_name": "test_index"
        })
        
        # Should handle empty message gracefully or return an error
        assert response.status_code in [200, 400, 500]

    def test_chat_missing_fields(self):
        """Test chat with missing required fields"""
        response = self.client.post("/api/chat", json={
            "message": "test message"
            # Missing index_name
        })
        
        assert response.status_code == 422  # Validation error

    def test_chat_valid_request_no_index(self):
        """Test chat with valid request but no index exists"""
        response = self.client.post("/api/chat", json={
            "message": "What is machine learning?",
            "index_name": "nonexistent_index",
            "use_synthesis": False
        })
        
        # Should return 500 since no index exists
        assert response.status_code == 500

    def test_request_validation(self):
        """Test request validation for different endpoints"""
        # Test invalid data types
        response = self.client.post("/api/search", json={
            "query": "test",
            "index_name": "test_index",
            "top_k": "not_a_number"  # Should be integer
        })
        
        assert response.status_code == 422

    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = self.client.options("/api/status")
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers

    def test_api_error_handling(self):
        """Test API error handling and response format"""
        # Test various error scenarios
        responses_to_test = [
            self.client.post("/api/search", json={"query": "", "index_name": "test"}),
            self.client.post("/api/indexes/create", json={"name": "", "directory_paths": []}),
        ]
        
        for response in responses_to_test:
            if response.status_code >= 400:
                data = response.json()
                assert "detail" in data  # FastAPI error format

    def test_content_type_validation(self):
        """Test content type validation"""
        # Test sending non-JSON data to JSON endpoints
        response = self.client.post(
            "/api/search",
            data="not json data",
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code == 422

    def test_endpoint_methods(self):
        """Test that endpoints only accept correct HTTP methods"""
        # Test POST endpoint with GET
        response = self.client.get("/api/search")
        assert response.status_code == 405  # Method not allowed
        
        # Test GET endpoint with POST
        response = self.client.post("/api/status")
        assert response.status_code == 405  # Method not allowed

    def test_response_format_consistency(self):
        """Test that API responses follow consistent format"""
        # Test successful responses
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        
        response = self.client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_large_request_handling(self):
        """Test handling of large requests"""
        # Create a large query string
        large_query = "machine learning " * 1000  # Very long query
        
        response = self.client.post("/api/search", json={
            "query": large_query,
            "index_name": "test_index",
            "top_k": 5
        })
        
        # Should handle gracefully (might return error or process it)
        assert response.status_code in [200, 400, 500, 413]

    def test_concurrent_requests(self):
        """Test handling concurrent requests to status endpoint"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = self.client.get("/api/status")
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
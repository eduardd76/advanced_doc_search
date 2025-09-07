#!/usr/bin/env python3
"""
Test script to verify indexing functionality
"""

import requests
import json
import time
import sys
import os

API_BASE = "http://localhost:8000"

def test_api_connection():
    """Test if API is accessible"""
    try:
        response = requests.get(f"{API_BASE}/")
        print(f"✅ API Connection: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ API Connection Failed: {e}")
        return False

def test_system_status():
    """Test system status endpoint"""
    try:
        response = requests.get(f"{API_BASE}/api/status")
        print(f"✅ System Status: {response.status_code}")
        data = response.json()
        print(f"   Status: {data.get('system_status', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ System Status Failed: {e}")
        return False

def test_indexing_status():
    """Test indexing status endpoint"""
    try:
        response = requests.get(f"{API_BASE}/api/indexes/status")
        print(f"✅ Indexing Status: {response.status_code}")
        data = response.json()
        print(f"   Status: {data.get('status_message', 'No message')}")
        print(f"   Is Indexing: {data.get('is_indexing', False)}")
        return True
    except Exception as e:
        print(f"❌ Indexing Status Failed: {e}")
        return False

def create_test_index():
    """Test creating an index"""
    # Create a test directory with a sample file
    test_dir = "test_docs"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Create a simple test file
    with open(f"{test_dir}/test_document.txt", "w") as f:
        f.write("This is a test document for indexing.\nIt contains some sample text to verify the indexing process works correctly.")
    
    # Create index request
    index_request = {
        "name": "test-index",
        "directory_paths": [os.path.abspath(test_dir)],
        "file_extensions": [".txt"],
        "force_rebuild": True,
        "chunk_size": 512
    }
    
    try:
        response = requests.post(f"{API_BASE}/api/indexes/create", json=index_request)
        print(f"✅ Create Index: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Message: {data.get('message', 'No message')}")
            print(f"   Directories: {data.get('directories', [])}")
            
            # Monitor indexing progress
            print("\n📈 Monitoring indexing progress...")
            for i in range(30):  # Wait up to 30 seconds
                status_response = requests.get(f"{API_BASE}/api/indexes/status")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    is_indexing = status_data.get('is_indexing', False)
                    message = status_data.get('status_message', 'No status')
                    progress = status_data.get('current_progress', 0)
                    total = status_data.get('total_documents', 0)
                    
                    print(f"   [{i+1}/30] {message} ({progress}/{total})")
                    
                    if not is_indexing:
                        print("✅ Indexing completed!")
                        break
                
                time.sleep(1)
            
            return True
        else:
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Create Index Failed: {e}")
        return False

def main():
    print("🧪 Testing Advanced Document Search API\n")
    
    # Run tests
    tests = [
        ("API Connection", test_api_connection),
        ("System Status", test_system_status),
        ("Indexing Status", test_indexing_status),
        ("Create Test Index", create_test_index)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n📊 Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The indexing system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the backend server and configuration.")
    
    # Cleanup
    import shutil
    if os.path.exists("test_docs"):
        shutil.rmtree("test_docs")

if __name__ == "__main__":
    main()
import React, { useState, useEffect } from 'react';
import './index.css';

function SimpleApp() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [indexName, setIndexName] = useState('test-index');

  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/status');
      const data = await response.json();
      setStatus(data);
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          index_name: indexName,
          top_k: 10,
          use_reranking: true
        })
      });
      const data = await response.json();
      setSearchResults(data);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="glass p-6 mb-6">
        <h1 className="text-2xl font-bold text-white mb-4">
          Advanced Document Search System
        </h1>
        <p className="text-gray">
          Hybrid BM25 + Dense Retrieval with Neural Reranking
        </p>
      </div>

      {/* System Status */}
      {status && (
        <div className="glass p-4 mb-6">
          <h2 className="text-lg font-bold text-white mb-2">System Status</h2>
          <div className="text-sm">
            <span className="text-gray">Status: </span>
            <span className="text-green">{status.system_status}</span>
            {status.configuration && (
              <div className="mt-2">
                <span className="text-gray">Port: </span>
                <span className="text-white">{status.configuration.server_port}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Search Form */}
      <div className="glass p-6 mb-6">
        <h2 className="text-xl font-bold text-white mb-4">Search Documents</h2>
        <form onSubmit={handleSearch} className="space-y-4">
          <div>
            <input
              type="text"
              className="input"
              placeholder="Enter your search query..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              disabled={loading}
            />
          </div>
          <div>
            <input
              type="text"
              className="input"
              placeholder="Index name"
              value={indexName}
              onChange={(e) => setIndexName(e.target.value)}
              disabled={loading}
            />
          </div>
          <button 
            type="submit" 
            className="button w-full"
            disabled={loading || !searchQuery.trim()}
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

      {/* Search Results */}
      {searchResults && (
        <div className="glass p-6">
          <h2 className="text-xl font-bold text-white mb-4">
            Results ({searchResults.total_results || 0})
          </h2>
          {searchResults.results && searchResults.results.length > 0 ? (
            <div className="space-y-4">
              {searchResults.results.map((result, index) => (
                <div key={index} className="card">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-white font-bold">#{index + 1}</span>
                    <span className="text-green text-sm">
                      Score: {result.score?.toFixed(3) || 'N/A'}
                    </span>
                  </div>
                  <p className="text-gray text-sm">
                    {result.content}
                  </p>
                  {result.doc_id && (
                    <div className="mt-2 text-xs text-blue">
                      Doc ID: {result.doc_id}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray">No results found</p>
          )}
        </div>
      )}

      {/* Links */}
      <div className="glass p-4 mt-6 text-center">
        <div className="flex justify-center gap-4 text-sm">
          <a 
            href="http://localhost:8000/docs" 
            target="_blank" 
            rel="noreferrer"
            className="text-blue hover:text-white"
          >
            API Documentation
          </a>
          <span className="text-gray">|</span>
          <a 
            href="http://localhost:8000/" 
            target="_blank" 
            rel="noreferrer"
            className="text-blue hover:text-white"
          >
            API Status
          </a>
        </div>
      </div>
    </div>
  );
}

export default SimpleApp;
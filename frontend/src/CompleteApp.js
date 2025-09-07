import React, { useState, useEffect } from 'react';
import './index.css';

function CompleteApp() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('index');
  
  // Indexing state
  const [indexName, setIndexName] = useState('my-documents');
  const [directories, setDirectories] = useState(['']);
  const [fileExtensions, setFileExtensions] = useState(['.pdf', '.docx', '.txt', '.md']);
  const [indexingStatus, setIndexingStatus] = useState(null);
  const [forceRebuild, setForceRebuild] = useState(false);
  
  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchIndexName, setSearchIndexName] = useState('my-documents');
  const [searchResults, setSearchResults] = useState(null);
  const [useReranking, setUseReranking] = useState(true);
  const [maxResults, setMaxResults] = useState(10);

  useEffect(() => {
    fetchStatus();
    // Poll indexing status every 2 seconds
    const interval = setInterval(fetchIndexingStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:8003/health');
      const data = await response.json();
      setStatus(data);
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  };

  const fetchIndexingStatus = async () => {
    try {
      const response = await fetch('http://localhost:8003/api/indexes/status');
      const data = await response.json();
      setIndexingStatus(data);
    } catch (error) {
      console.error('Failed to fetch indexing status:', error);
    }
  };

  const handleCreateIndex = async (e) => {
    e.preventDefault();
    if (!indexName.trim() || directories.every(d => !d.trim())) {
      alert('Please provide an index name and at least one directory');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/indexes/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: indexName,
          directory_paths: directories.filter(d => d.trim()),
          file_extensions: fileExtensions.length > 0 ? fileExtensions : null,
          force_rebuild: forceRebuild
        })
      });
      
      if (!response.ok) {
        const error = await response.json();
        alert(`Error: ${error.detail || 'Failed to create index'}`);
      } else {
        const data = await response.json();
        alert(`Index creation started for: ${data.index_name}`);
        fetchIndexingStatus();
      }
    } catch (error) {
      console.error('Index creation failed:', error);
      alert('Failed to create index: ' + error.message);
    } finally {
      setLoading(false);
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
          index_name: searchIndexName,
          top_k: maxResults,
          use_reranking: useReranking,
          bm25_candidates: 100,
          dense_candidates: 100
        })
      });
      
      if (!response.ok) {
        const error = await response.json();
        alert(`Search error: ${error.detail || 'Search failed'}`);
        setSearchResults(null);
      } else {
        const data = await response.json();
        setSearchResults(data);
      }
    } catch (error) {
      console.error('Search failed:', error);
      alert('Search failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const addDirectory = () => {
    setDirectories([...directories, '']);
  };

  const updateDirectory = (index, value) => {
    const newDirs = [...directories];
    newDirs[index] = value;
    setDirectories(newDirs);
  };

  const removeDirectory = (index) => {
    setDirectories(directories.filter((_, i) => i !== index));
  };

  const toggleExtension = (ext) => {
    if (fileExtensions.includes(ext)) {
      setFileExtensions(fileExtensions.filter(e => e !== ext));
    } else {
      setFileExtensions([...fileExtensions, ext]);
    }
  };

  const getProgressPercent = () => {
    if (!indexingStatus || !indexingStatus.total_documents) return 0;
    return Math.round((indexingStatus.current_progress / indexingStatus.total_documents) * 100);
  };

  return (
    <div className="container">
      {/* Header */}
      <div className="glass p-6 mb-6">
        <h1 className="text-2xl font-bold text-white mb-4">
          Advanced Document Search System
        </h1>
        <p className="text-gray mb-4">
          State-of-the-art Hybrid BM25 + Dense Retrieval with Neural Reranking
        </p>
        
        {/* Tabs */}
        <div className="flex gap-4">
          <button
            className={`button ${activeTab === 'index' ? '' : 'bg-opacity-50'}`}
            onClick={() => setActiveTab('index')}
          >
            Create Index
          </button>
          <button
            className={`button ${activeTab === 'search' ? '' : 'bg-opacity-50'}`}
            onClick={() => setActiveTab('search')}
          >
            Search Documents
          </button>
          <button
            className={`button ${activeTab === 'status' ? '' : 'bg-opacity-50'}`}
            onClick={() => setActiveTab('status')}
          >
            System Status
          </button>
        </div>
      </div>

      {/* Index Tab */}
      {activeTab === 'index' && (
        <div className="glass p-6 mb-6">
          <h2 className="text-xl font-bold text-white mb-4">Create Document Index</h2>
          
          {/* Indexing Status */}
          {indexingStatus && indexingStatus.is_indexing && (
            <div className="card bg-blue-500 bg-opacity-20 mb-4">
              <h3 className="font-bold text-white mb-2">Indexing in Progress...</h3>
              <div className="progress-bar mb-2">
                <div 
                  className="progress-fill"
                  style={{ width: `${getProgressPercent()}%` }}
                />
              </div>
              <div className="text-sm text-gray">
                Progress: {indexingStatus.current_progress}/{indexingStatus.total_documents} documents
                ({getProgressPercent()}%)
              </div>
              <div className="text-sm text-white mt-1">
                {indexingStatus.status_message}
              </div>
            </div>
          )}

          <form onSubmit={handleCreateIndex} className="space-y-4">
            {/* Index Name */}
            <div>
              <label className="block text-white font-semibold mb-2">
                Index Name *
              </label>
              <input
                type="text"
                className="input"
                placeholder="e.g., my-documents"
                value={indexName}
                onChange={(e) => setIndexName(e.target.value)}
                disabled={loading}
              />
            </div>

            {/* Directory Paths */}
            <div>
              <label className="block text-white font-semibold mb-2">
                Directory Paths * (Use full paths like F:\Documents)
              </label>
              {directories.map((dir, index) => (
                <div key={index} className="flex gap-2 mb-2">
                  <input
                    type="text"
                    className="input flex-1"
                    placeholder="F:\Documents\MyFiles"
                    value={dir}
                    onChange={(e) => updateDirectory(index, e.target.value)}
                    disabled={loading}
                  />
                  {directories.length > 1 && (
                    <button
                      type="button"
                      className="button bg-red-500 bg-opacity-50"
                      onClick={() => removeDirectory(index)}
                      disabled={loading}
                    >
                      Remove
                    </button>
                  )}
                </div>
              ))}
              <button
                type="button"
                className="button bg-green-500 bg-opacity-50"
                onClick={addDirectory}
                disabled={loading}
              >
                + Add Directory
              </button>
            </div>

            {/* File Extensions */}
            <div>
              <label className="block text-white font-semibold mb-2">
                File Types to Index
              </label>
              <div className="flex flex-wrap gap-2">
                {['.pdf', '.docx', '.txt', '.md', '.html', '.rtf', '.epub'].map(ext => (
                  <label key={ext} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={fileExtensions.includes(ext)}
                      onChange={() => toggleExtension(ext)}
                      disabled={loading}
                    />
                    <span className="text-white">{ext}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Force Rebuild */}
            <div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={forceRebuild}
                  onChange={(e) => setForceRebuild(e.target.checked)}
                  disabled={loading}
                />
                <span className="text-white">Force rebuild (overwrite existing index)</span>
              </label>
            </div>

            <button
              type="submit"
              className="button w-full"
              disabled={loading || (indexingStatus && indexingStatus.is_indexing)}
            >
              {loading ? 'Creating Index...' : 
               (indexingStatus && indexingStatus.is_indexing) ? 'Indexing in Progress...' : 
               'Create Index'}
            </button>
          </form>

          {/* Last Indexing Result */}
          {indexingStatus && !indexingStatus.is_indexing && indexingStatus.status_message !== 'Ready' && (
            <div className="mt-4 p-4 bg-green-500 bg-opacity-20 rounded">
              <div className="text-white font-semibold">Last Indexing Result:</div>
              <div className="text-gray text-sm">{indexingStatus.status_message}</div>
              {indexingStatus.processed_documents > 0 && (
                <div className="text-sm text-white mt-1">
                  Processed: {indexingStatus.processed_documents} documents, 
                  Failed: {indexingStatus.failed_documents}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Search Tab */}
      {activeTab === 'search' && (
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
            <div className="grid grid-2 gap-4">
              <input
                type="text"
                className="input"
                placeholder="Index name (e.g., my-documents)"
                value={searchIndexName}
                onChange={(e) => setSearchIndexName(e.target.value)}
                disabled={loading}
              />
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-white">
                  <input
                    type="checkbox"
                    checked={useReranking}
                    onChange={(e) => setUseReranking(e.target.checked)}
                    disabled={loading}
                  />
                  Neural Reranking
                </label>
                <label className="flex items-center gap-2 text-white">
                  <span className="text-sm">Results:</span>
                  <select
                    className="bg-gray-800 text-white px-2 py-1 rounded text-sm"
                    value={maxResults}
                    onChange={(e) => setMaxResults(parseInt(e.target.value))}
                    disabled={loading}
                  >
                    <option value={10}>10</option>
                    <option value={25}>25</option>
                    <option value={50}>50</option>
                    <option value={100}>100</option>
                  </select>
                </label>
              </div>
            </div>
            <button
              type="submit"
              className="button w-full"
              disabled={loading || !searchQuery.trim()}
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </form>

          {/* Search Results */}
          {searchResults && (
            <div className="mt-6">
              <h3 className="text-lg font-bold text-white mb-4">
                Results ({searchResults.total_results || 0})
              </h3>
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
                      <p className="text-gray text-sm mb-2">
                        {result.content}
                      </p>
                      <div className="text-xs text-blue mb-1">
                        Source: {result.metadata?.file_path ? 
                          result.metadata.file_path.split('\\').pop().split('/').pop() : 
                          result.doc_id}
                      </div>
                      {result.metadata && (
                        <div className="text-xs text-gray">
                          {result.metadata.chunk_info?.page_number && (
                            <span className="mr-3">Page: {result.metadata.chunk_info.page_number}</span>
                          )}
                          {result.metadata.page_count && (
                            <span className="mr-3">Document: {result.metadata.page_count} pages</span>
                          )}
                          {result.metadata.chunk_info?.word_count && (
                            <span className="mr-3">Chunk: {result.metadata.chunk_info.word_count} words</span>
                          )}
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
        </div>
      )}

      {/* Status Tab */}
      {activeTab === 'status' && status && (
        <div className="glass p-6">
          <h2 className="text-xl font-bold text-white mb-4">System Status</h2>
          <div className="grid grid-2 gap-4">
            <div className="card">
              <h3 className="font-semibold text-white mb-2">System</h3>
              <div className="text-sm space-y-1">
                <div>
                  <span className="text-gray">Status: </span>
                  <span className="text-green">{status.system_status}</span>
                </div>
                <div>
                  <span className="text-gray">Backend Port: </span>
                  <span className="text-white">{status.configuration?.server_port || 8002}</span>
                </div>
                <div>
                  <span className="text-gray">Storage: </span>
                  <span className="text-white text-xs">{status.configuration?.storage_dir || 'Default'}</span>
                </div>
              </div>
            </div>

            <div className="card">
              <h3 className="font-semibold text-white mb-2">Supported Formats</h3>
              <div className="text-sm text-gray">
                {status.configuration?.supported_formats?.join(', ') || 'PDF, DOCX, TXT, MD, HTML, RTF, EPUB'}
              </div>
            </div>
          </div>

          {/* Links */}
          <div className="mt-6 flex gap-4">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="button"
            >
              API Documentation
            </a>
            <a
              href="http://localhost:8000/"
              target="_blank"
              rel="noreferrer"
              className="button"
            >
              API Root
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

export default CompleteApp;
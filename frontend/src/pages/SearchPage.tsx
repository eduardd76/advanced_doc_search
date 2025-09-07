import React, { useState, useEffect } from 'react';
import { useApi } from '../contexts/ApiContext';
import { 
  Search, 
  Settings, 
  FileText, 
  Clock, 
  Star,
  ExternalLink,
  Filter,
  BarChart3
} from 'lucide-react';

const SearchPage: React.FC = () => {
  const { searchDocuments, searchResults, loading, error } = useApi();
  
  const [formData, setFormData] = useState({
    query: '',
    index_name: '',
    top_k: 10,
    use_reranking: true,
    bm25_candidates: 100,
    dense_candidates: 100
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [availableIndexes, setAvailableIndexes] = useState<string[]>([]);

  useEffect(() => {
    const fetchIndexes = async () => {
      try {
        console.log('Fetching indexes...');
        const response = await fetch('http://localhost:8003/api/indexes/list');
        const data = await response.json();
        console.log('Received indexes:', data);
        setAvailableIndexes(data.indexes || []);
      } catch (err) {
        console.error('Failed to fetch indexes:', err);
      }
    };
    fetchIndexes();
  }, []);

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.query.trim()) {
      alert('Please enter a search query');
      return;
    }

    if (!formData.index_name.trim()) {
      alert('Please enter an index name');
      return;
    }

    try {
      await searchDocuments(formData);
    } catch (err) {
      console.error('Search failed:', err);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green';
    if (score >= 0.6) return 'text-blue';
    if (score >= 0.4) return 'text-yellow';
    return 'text-gray';
  };


  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="card">
        <h1 className="heading-2" style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
          <Search style={{ width: '28px', height: '28px', color: 'var(--bg-accent)' }} />
          Hybrid Document Search
        </h1>
        <p className="text-body" style={{ marginBottom: '24px' }}>
          Search your indexed documents using our advanced hybrid retrieval system combining BM25, dense embeddings, and neural reranking.
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-2 gap-4">
            {/* Search Query */}
            <div>
              <label style={{ display: 'block', color: '#000000', fontWeight: '600', marginBottom: '8px' }}>
                Search Query *
              </label>
              <input
                type="text"
                className="input"
                placeholder="What are you looking for?"
                value={formData.query}
                onChange={(e) => handleInputChange('query', e.target.value)}
                disabled={loading}
              />
            </div>

            {/* Index Name */}
            <div>
              <label className="block text-white font-semibold mb-2">
                Index Name *
              </label>
              <select
                className="input"
                value={formData.index_name}
                onChange={(e) => handleInputChange('index_name', e.target.value)}
                disabled={loading}
              >
                <option value="">Select an index...</option>
                {availableIndexes.map((indexName) => (
                  <option key={indexName} value={indexName}>
                    {indexName}
                  </option>
                ))}
              </select>
              {availableIndexes.length === 0 && (
                <p className="text-gray text-sm mt-1">
                  No indexes available. Create an index first.
                </p>
              )}
            </div>
          </div>

          {/* Advanced Settings */}
          <div className="glass p-4">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-white hover:text-gray transition-colors"
            >
              <Settings className="w-4 h-4" />
              Advanced Settings
              <span className="text-xs text-gray">
                ({showAdvanced ? 'Hide' : 'Show'})
              </span>
            </button>

            {showAdvanced && (
              <div className="grid grid-2 gap-4 mt-4">
                <div>
                  <label className="block text-white font-semibold mb-2">
                    Results Count
                  </label>
                  <select
                    className="input"
                    value={formData.top_k}
                    onChange={(e) => handleInputChange('top_k', parseInt(e.target.value))}
                    disabled={loading}
                  >
                    <option value={5}>5 results</option>
                    <option value={10}>10 results</option>
                    <option value={20}>20 results</option>
                    <option value={50}>50 results</option>
                  </select>
                </div>

                <div>
                  <label className="block text-white font-semibold mb-2">
                    BM25 Candidates
                  </label>
                  <input
                    type="number"
                    className="input"
                    min="10"
                    max="1000"
                    value={formData.bm25_candidates}
                    onChange={(e) => handleInputChange('bm25_candidates', parseInt(e.target.value))}
                    disabled={loading}
                  />
                </div>

                <div>
                  <label className="block text-white font-semibold mb-2">
                    Dense Candidates
                  </label>
                  <input
                    type="number"
                    className="input"
                    min="10"
                    max="1000"
                    value={formData.dense_candidates}
                    onChange={(e) => handleInputChange('dense_candidates', parseInt(e.target.value))}
                    disabled={loading}
                  />
                </div>

                <div className="flex items-center">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formData.use_reranking}
                      onChange={(e) => handleInputChange('use_reranking', e.target.checked)}
                      disabled={loading}
                      className="accent-blue-500"
                    />
                    <span className="text-white">Use Neural Reranking</span>
                  </label>
                </div>
              </div>
            )}
          </div>

          <button
            type="submit"
            className="button w-full flex items-center justify-center gap-2"
            disabled={loading}
          >
            {loading ? (
              <>
                <div className="loading"></div>
                Searching...
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                Search Documents
              </>
            )}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-3 bg-red-500 bg-opacity-20 border border-red-500 border-opacity-50 rounded text-red-200">
            {error}
          </div>
        )}
      </div>

      {/* Search Results */}
      {searchResults && (
        <div className="glass p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Search Results
            </h2>
            <div className="text-sm text-gray">
              {searchResults.total_results} results found in {searchResults.search_method} mode
            </div>
          </div>

          {/* Search Summary */}
          <div className="glass p-4 mb-6">
            <div className="grid grid-2 gap-4">
              <div>
                <h3 className="font-semibold text-white mb-2">Query Analysis</h3>
                <p className="text-gray text-sm mb-2">"{searchResults.query}"</p>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-gray">Method: <span className="text-white">{searchResults.search_method}</span></span>
                  <span className="text-gray">Results: <span className="text-white">{searchResults.total_results}</span></span>
                </div>
              </div>

              <div>
                <h3 className="font-semibold text-white mb-2">Search Parameters</h3>
                <div className="text-sm space-y-1">
                  <div className="flex justify-between">
                    <span className="text-gray">Top K:</span>
                    <span className="text-white">{searchResults.parameters?.top_k}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray">Reranking:</span>
                    <span className="text-white">{searchResults.parameters?.use_reranking ? 'Enabled' : 'Disabled'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray">BM25 Candidates:</span>
                    <span className="text-white">{searchResults.parameters?.bm25_candidates}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray">Dense Candidates:</span>
                    <span className="text-white">{searchResults.parameters?.dense_candidates}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Results List */}
          <div className="space-y-4">
            {searchResults.results.map((result: any, index: number) => (
              <div key={result.doc_id} className="card">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-white font-bold">#{index + 1}</span>
                    <FileText className="w-4 h-4 text-blue" />
                    <div className="flex flex-col">
                      <span className="text-white font-semibold text-sm">
                        {result.metadata?.filename || result.metadata?.file_path?.split('\\').pop() || result.metadata?.file_path?.split('/').pop() || `Document ${result.doc_id}`}
                      </span>
                      {(result.metadata?.filename || result.metadata?.file_path) && (
                        <span className="text-gray text-xs">
                          ID: {result.doc_id}
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Star className={`w-4 h-4 ${getScoreColor(result.score)}`} />
                    <span className={`font-bold ${getScoreColor(result.score)}`}>
                      {result.score.toFixed(3)}
                    </span>
                  </div>
                </div>

                <div className="mb-3">
                  <p className="text-gray text-sm leading-relaxed">
                    {result.content}
                  </p>
                </div>

                {/* Essential Metadata */}
                {result.metadata && Object.keys(result.metadata).length > 0 && (
                  <div className="mb-3 p-2 bg-white bg-opacity-5 rounded text-xs">
                    <div className="flex flex-wrap gap-3">
                      {result.metadata.page_count && (
                        <span className="text-gray">
                          📄 {result.metadata.page_count} pages
                        </span>
                      )}
                      {result.metadata.file_path && (
                        <span className="text-gray truncate">
                          📁 {result.metadata.file_path.split('\\').pop() || result.metadata.file_path.split('/').pop()}
                        </span>
                      )}
                    </div>
                  </div>
                )}

              </div>
            ))}
          </div>

          {searchResults.results.length === 0 && (
            <div className="text-center py-12">
              <Search className="w-12 h-12 text-gray mx-auto mb-4" />
              <p className="text-gray">No results found for your query.</p>
              <p className="text-gray text-sm mt-2">
                Try different keywords or check your index name.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SearchPage;
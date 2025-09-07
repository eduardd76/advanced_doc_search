import React, { useState, useEffect } from 'react';
import { useApi } from '../contexts/ApiContext';
import { 
  FolderPlus, 
  Settings, 
  Play, 
  Pause, 
  CheckCircle, 
  AlertCircle,
  FileText,
  Clock,
  Activity,
  Trash2
} from 'lucide-react';

const IndexPage: React.FC = () => {
  const { createIndex, getIndexingStatus, indexingStatus, loading, error, listIndexes, deleteIndex } = useApi();
  
  const [formData, setFormData] = useState({
    name: '',
    directory_paths: [''],
    file_extensions: [] as string[],
    force_rebuild: false,
    chunk_size: 1024
  });

  const [customExtensions, setCustomExtensions] = useState('');
  const [isPolling, setIsPolling] = useState(false);
  const [existingIndexes, setExistingIndexes] = useState<(string | {name: string, documents?: number, status?: string})[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Common file extensions
  const commonExtensions = [
    { value: '.pdf', label: 'PDF' },
    { value: '.docx', label: 'Word' },
    { value: '.txt', label: 'Text' },
    { value: '.md', label: 'Markdown' },
    { value: '.epub', label: 'EPUB' },
    { value: '.rtf', label: 'RTF' },
    { value: '.html', label: 'HTML' }
  ];

  // Poll indexing status when indexing is in progress
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isPolling) {
      interval = setInterval(() => {
        getIndexingStatus();
      }, 2000);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isPolling, getIndexingStatus]);

  // Stop polling when indexing is complete
  useEffect(() => {
    if (indexingStatus && !indexingStatus.is_indexing) {
      setIsPolling(false);
    }
  }, [indexingStatus]);

  // Fetch existing indexes on component mount
  useEffect(() => {
    const fetchIndexes = async () => {
      try {
        const data = await listIndexes();
        setExistingIndexes(data.indexes || []);
      } catch (err) {
        console.error('Failed to fetch indexes:', err);
      }
    };
    fetchIndexes();
  }, [listIndexes]);

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleDirectoryChange = (index: number, value: string) => {
    const newDirectories = [...formData.directory_paths];
    newDirectories[index] = value;
    setFormData(prev => ({
      ...prev,
      directory_paths: newDirectories
    }));
  };

  const addDirectory = () => {
    setFormData((prev: typeof formData) => ({
      ...prev,
      directory_paths: [...prev.directory_paths, '']
    }));
  };

  const removeDirectory = (index: number) => {
    setFormData((prev: typeof formData) => ({
      ...prev,
      directory_paths: prev.directory_paths.filter((_: string, i: number) => i !== index)
    }));
  };

  const handleExtensionToggle = (extension: string) => {
    setFormData((prev: typeof formData) => ({
      ...prev,
      file_extensions: prev.file_extensions.includes(extension)
        ? prev.file_extensions.filter((ext: string) => ext !== extension)
        : [...prev.file_extensions, extension]
    }));
  };

  const handleCustomExtensions = () => {
    const extensions = customExtensions
      .split(',')
      .map(ext => ext.trim())
      .filter(ext => ext.length > 0)
      .map(ext => ext.startsWith('.') ? ext : '.' + ext);

    setFormData((prev: typeof formData) => ({
      ...prev,
      file_extensions: [...new Set([...prev.file_extensions, ...extensions])]
    }));
    setCustomExtensions('');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim()) {
      alert('Please enter an index name');
      return;
    }

    if (formData.directory_paths.every((path: string) => !path.trim())) {
      alert('Please add at least one directory path');
      return;
    }

    try {
      await createIndex({
        ...formData,
        directory_paths: formData.directory_paths.filter((path: string) => path.trim()),
        file_extensions: formData.file_extensions.length > 0 ? formData.file_extensions : null
      });
      
      setIsPolling(true);
      getIndexingStatus();
      
      // Refresh the index list
      const data = await listIndexes();
      setExistingIndexes(data.indexes || []);
      
      alert(`Index "${formData.name}" created successfully!`);
      
    } catch (err: any) {
      console.error('Index creation failed:', err);
      alert(`Failed to create index: ${err?.response?.data?.detail || err.message || 'Unknown error'}`);
    }
  };

  const handleDeleteIndex = async (indexName: string) => {
    if (!window.confirm(`Are you sure you want to delete the index "${indexName}"? This action cannot be undone.`)) {
      return;
    }

    try {
      await deleteIndex(indexName);
      
      // Refresh the index list
      const data = await listIndexes();
      setExistingIndexes(data.indexes || []);
      
      alert(`Index "${indexName}" deleted successfully!`);
      
    } catch (err: any) {
      console.error('Index deletion failed:', err);
      alert(`Failed to delete index: ${err?.response?.data?.detail || err.message || 'Unknown error'}`);
    }
  };

  const getProgressPercentage = () => {
    if (!indexingStatus || indexingStatus.total_documents === 0) return 0;
    return Math.round((indexingStatus.current_progress / indexingStatus.total_documents) * 100);
  };

  return (
    <div className="space-y-6">
      <div className="glass p-6">
        <h1 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
          <FileText className="w-6 h-6" />
          Document Indexing
        </h1>
        <p className="text-gray mb-6">
          Create searchable indexes from your document collections using our advanced hybrid retrieval system.
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Index Name */}
          <div>
            <label className="block text-white font-semibold mb-2">
              Index Name *
            </label>
            <input
              type="text"
              className="input"
              placeholder="e.g., my-documents"
              value={formData.name}
              onChange={(e) => handleInputChange('name', e.target.value)}
              disabled={loading || (indexingStatus?.is_indexing)}
            />
          </div>

          {/* Directory Paths */}
          <div>
            <label className="block text-white font-semibold mb-2">
              Directory Paths *
            </label>
            <div className="space-y-2">
              {formData.directory_paths.map((path: string, index: number) => (
                <div key={index} className="flex gap-2">
                  <input
                    type="text"
                    className="input flex-1"
                    placeholder="C:\Documents\MyFiles"
                    value={path}
                    onChange={(e) => handleDirectoryChange(index, e.target.value)}
                    disabled={loading || (indexingStatus?.is_indexing)}
                  />
                  {formData.directory_paths.length > 1 && (
                    <button
                      type="button"
                      onClick={() => removeDirectory(index)}
                      className="button"
                      disabled={loading || (indexingStatus?.is_indexing)}
                    >
                      Remove
                    </button>
                  )}
                </div>
              ))}
              <button
                type="button"
                onClick={addDirectory}
                className="button flex items-center gap-2"
                disabled={loading || (indexingStatus?.is_indexing)}
              >
                <FolderPlus className="w-4 h-4" />
                Add Directory
              </button>
            </div>
          </div>

          {/* File Extensions */}
          <div>
            <label className="block text-white font-semibold mb-2">
              File Extensions (optional)
            </label>
            <p className="text-gray text-sm mb-3">
              Select file types to index. If none selected, all supported formats will be indexed.
            </p>
            
            <div className="grid grid-3 gap-2 mb-4">
              {commonExtensions.map(({ value, label }) => (
                <label
                  key={value}
                  className="flex items-center gap-2 cursor-pointer p-2 rounded glass hover:bg-white hover:bg-opacity-10 transition-colors"
                >
                  <input
                    type="checkbox"
                    checked={formData.file_extensions.includes(value)}
                    onChange={() => handleExtensionToggle(value)}
                    disabled={loading || (indexingStatus?.is_indexing)}
                    className="accent-blue-500"
                  />
                  <span className="text-white text-sm">{label}</span>
                  <span className="text-gray text-xs">{value}</span>
                </label>
              ))}
            </div>

            <div className="flex gap-2">
              <input
                type="text"
                className="input flex-1"
                placeholder="Custom extensions (comma-separated: .xml, .json)"
                value={customExtensions}
                onChange={(e) => setCustomExtensions(e.target.value)}
                disabled={loading || (indexingStatus?.is_indexing)}
              />
              <button
                type="button"
                onClick={handleCustomExtensions}
                className="button"
                disabled={loading || (indexingStatus?.is_indexing) || !customExtensions.trim()}
              >
                Add
              </button>
            </div>

            {formData.file_extensions.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-2">
                {formData.file_extensions.map((ext: string) => (
                  <span
                    key={ext}
                    className="px-2 py-1 bg-white bg-opacity-20 rounded text-white text-sm flex items-center gap-1"
                  >
                    {ext}
                    <button
                      type="button"
                      onClick={() => handleExtensionToggle(ext)}
                      className="ml-1 text-gray hover:text-white"
                      disabled={loading || (indexingStatus?.is_indexing)}
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Advanced Options */}
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
                    Chunk Size
                  </label>
                  <select
                    className="input"
                    value={formData.chunk_size}
                    onChange={(e) => handleInputChange('chunk_size', parseInt(e.target.value))}
                    disabled={loading || (indexingStatus?.is_indexing)}
                  >
                    <option value={512}>512 tokens</option>
                    <option value={1024}>1024 tokens</option>
                    <option value={2048}>2048 tokens</option>
                  </select>
                </div>

                <div className="flex items-center">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formData.force_rebuild}
                      onChange={(e) => handleInputChange('force_rebuild', e.target.checked)}
                      disabled={loading || (indexingStatus?.is_indexing)}
                      className="accent-blue-500"
                    />
                    <span className="text-white">Force Rebuild</span>
                  </label>
                </div>
              </div>
            )}
          </div>

          <button
            type="submit"
            className="button w-full flex items-center justify-center gap-2"
            disabled={loading || (indexingStatus?.is_indexing)}
          >
            {loading ? (
              <>
                <div className="loading"></div>
                Creating Index...
              </>
            ) : indexingStatus?.is_indexing ? (
              <>
                <Activity className="w-4 h-4" />
                Indexing in Progress...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Create Index
              </>
            )}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-3 bg-red-500 bg-opacity-20 border border-red-500 border-opacity-50 rounded text-red-200">
            <AlertCircle className="w-4 h-4 inline mr-2" />
            {error}
          </div>
        )}
      </div>

      {/* Existing Indexes */}
      {existingIndexes.length > 0 && (
        <div className="glass p-6">
          <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <FileText className="w-5 h-5" />
            Existing Indexes
          </h2>
          <div className="space-y-3">
            {existingIndexes.map((index) => {
              const indexName = typeof index === 'string' ? index : index.name;
              const docCount = typeof index === 'object' && index !== null ? index.documents : undefined;
              
              return (
                <div key={indexName} className="flex items-center justify-between p-3 bg-white bg-opacity-5 rounded border border-white border-opacity-10">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-green rounded-full"></div>
                    <span className="text-white font-semibold">
                      {indexName}
                    </span>
                    {docCount && (
                      <span className="text-gray text-sm">({docCount} docs)</span>
                    )}
                  </div>
                  <button
                    onClick={() => handleDeleteIndex(indexName)}
                    className="flex items-center gap-2 px-3 py-1 bg-red-500 bg-opacity-20 border border-red-500 border-opacity-50 rounded text-red-200 hover:bg-opacity-30 transition-colors"
                    disabled={loading || (indexingStatus?.is_indexing)}
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Indexing Status */}
      {indexingStatus && (
        <div className="glass p-6">
          <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            {indexingStatus.is_indexing ? (
              <Activity className="w-5 h-5 text-blue animate-pulse" />
            ) : (
              <CheckCircle className="w-5 h-5 text-green" />
            )}
            Indexing Status
          </h2>

          {indexingStatus.current_index && (
            <div className="mb-4">
              <span className="text-gray">Current Index:</span>
              <span className="text-white font-semibold ml-2">{indexingStatus.current_index}</span>
            </div>
          )}

          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${getProgressPercentage()}%` }}
            />
          </div>
          
          <div className="grid grid-2 gap-4 mt-4">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray">Progress:</span>
                <span className="text-white">
                  {indexingStatus.current_progress} / {indexingStatus.total_documents}
                  {indexingStatus.total_documents > 0 && ` (${getProgressPercentage()}%)`}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray">Processed:</span>
                <span className="text-green">{indexingStatus.processed_documents}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray">Failed:</span>
                <span className="text-yellow">{indexingStatus.failed_documents}</span>
              </div>
            </div>
            
            <div>
              <span className="text-gray">Status:</span>
              <p className="text-white mt-1 text-sm">
                {indexingStatus.status_message}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default IndexPage;
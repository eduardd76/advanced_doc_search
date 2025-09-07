import React, { createContext, useContext, useState, useCallback } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8003';

interface ApiContextType {
  loading: boolean;
  error: string | null;
  systemStatus: any;
  indexingStatus: any;
  searchResults: any;
  evaluationResults: any;
  chatResponse: any;
  
  getSystemStatus: () => Promise<void>;
  getIndexingStatus: () => Promise<void>;
  createIndex: (data: any) => Promise<void>;
  searchDocuments: (data: any) => Promise<void>;
  evaluateSystem: (data: any) => Promise<void>;
  chatWithDocuments: (data: any) => Promise<any>;
  listIndexes: () => Promise<any>;
  loadIndex: (indexName: string) => Promise<any>;
  deleteIndex: (indexName: string) => Promise<any>;
}

const ApiContext = createContext<ApiContextType | undefined>(undefined);

export const ApiProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [indexingStatus, setIndexingStatus] = useState(null);
  const [searchResults, setSearchResults] = useState(null);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [chatResponse, setChatResponse] = useState(null);

  const handleApiCall = useCallback(async (apiCall: () => Promise<any>) => {
    try {
      setLoading(true);
      setError(null);
      return await apiCall();
    } catch (err: any) {
      let errorMessage = 'An error occurred';
      
      if (err.response?.data?.detail) {
        // Handle FastAPI validation errors (array of error objects)
        if (Array.isArray(err.response.data.detail)) {
          errorMessage = err.response.data.detail.map((e: any) => 
            `${e.loc?.join?.('.') || 'field'}: ${e.msg || 'validation error'}`
          ).join(', ');
        } else if (typeof err.response.data.detail === 'string') {
          errorMessage = err.response.data.detail;
        } else {
          errorMessage = JSON.stringify(err.response.data.detail);
        }
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const getSystemStatus = useCallback(async () => {
    await handleApiCall(async () => {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setSystemStatus(response.data);
      return response.data;
    });
  }, [handleApiCall]);

  const getIndexingStatus = useCallback(async () => {
    await handleApiCall(async () => {
      const response = await axios.get(`${API_BASE_URL}/api/indexes/status`);
      setIndexingStatus(response.data);
      return response.data;
    });
  }, [handleApiCall]);

  const createIndex = useCallback(async (data: any) => {
    await handleApiCall(async () => {
      const response = await axios.post(`${API_BASE_URL}/index`, data);
      return response.data;
    });
  }, [handleApiCall]);

  const searchDocuments = useCallback(async (data: any) => {
    await handleApiCall(async () => {
      const response = await axios.post(`${API_BASE_URL}/search`, data);
      setSearchResults(response.data);
      return response.data;
    });
  }, [handleApiCall]);

  const evaluateSystem = useCallback(async (data: any) => {
    await handleApiCall(async () => {
      const response = await axios.post(`${API_BASE_URL}/api/evaluate`, data);
      setEvaluationResults(response.data);
      return response.data;
    });
  }, [handleApiCall]);

  const chatWithDocuments = useCallback(async (data: any) => {
    return await handleApiCall(async () => {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, data);
      setChatResponse(response.data);
      return response.data;
    });
  }, [handleApiCall]);

  const listIndexes = useCallback(async () => {
    return await handleApiCall(async () => {
      const response = await axios.get(`${API_BASE_URL}/api/indexes/list`);
      return response.data;
    });
  }, [handleApiCall]);

  const loadIndex = useCallback(async (indexName: string) => {
    return await handleApiCall(async () => {
      const response = await axios.post(`${API_BASE_URL}/api/indexes/load`, {
        index_name: indexName
      });
      return response.data;
    });
  }, [handleApiCall]);

  const deleteIndex = useCallback(async (indexName: string) => {
    return await handleApiCall(async () => {
      const response = await axios.delete(`${API_BASE_URL}/api/indexes/${encodeURIComponent(indexName)}`);
      return response.data;
    });
  }, [handleApiCall]);

  const value = {
    loading,
    error,
    systemStatus,
    indexingStatus,
    searchResults,
    evaluationResults,
    chatResponse,
    getSystemStatus,
    getIndexingStatus,
    createIndex,
    searchDocuments,
    evaluateSystem,
    chatWithDocuments,
    listIndexes,
    loadIndex,
    deleteIndex,
  };

  return (
    <ApiContext.Provider value={value}>
      {children}
    </ApiContext.Provider>
  );
};

export const useApi = () => {
  const context = useContext(ApiContext);
  if (context === undefined) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};
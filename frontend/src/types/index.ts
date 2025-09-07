export interface Document {
  doc_id: string;
  content: string;
  metadata: Record<string, any>;
}

export interface SearchResult {
  doc_id: string;
  content: string;
  score: number;
  metadata: Record<string, any>;
  component_scores?: Record<string, number>;
  rank_info?: any;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total_results: number;
  search_method: string;
  parameters: {
    top_k: number;
    use_reranking: boolean;
    bm25_candidates: number;
    dense_candidates: number;
  };
}

export interface IndexingStatus {
  is_indexing: boolean;
  current_progress: number;
  total_documents: number;
  status_message: string;
  current_index: string | null;
  processed_documents: number;
  failed_documents: number;
}

export interface SystemStatus {
  system_status: string;
  indexing_status: IndexingStatus;
  system_statistics: Record<string, any>;
  configuration: {
    server_port: number;
    storage_dir: string;
    supported_formats: string[];
  };
}

export interface EvaluationMetrics {
  recall_at_5: number;
  recall_at_10: number;
  precision_at_5: number;
  precision_at_10: number;
  mrr: number;
  ndcg_at_5: number;
  ndcg_at_10: number;
  map_score: number;
  hit_rate: number;
  avg_retrieval_time: number;
}

export interface EvaluationResults {
  evaluation_method: string;
  benchmark_queries: number;
  metrics: EvaluationMetrics;
  total_queries_evaluated: number;
}

export interface ChatResponse {
  query: string;
  response: string;
  context_used: number;
  sources: Array<{
    doc_id: string;
    score: number;
    metadata: Record<string, any>;
  }>;
  synthesis_mode: boolean;
}
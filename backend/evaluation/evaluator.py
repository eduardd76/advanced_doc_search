"""
Comprehensive evaluation framework for retrieval systems
"""

import json
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt  # Optional for visualization
# import seaborn as sns  # Optional for visualization
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from collections import defaultdict
from sklearn.metrics import ndcg_score
import math

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Single query evaluation result"""
    query_id: str
    query: str
    retrieved_docs: List[str]
    relevance_scores: List[float]
    ground_truth: List[str]
    metrics: Dict[str, float]
    retrieval_time: float
    method: str

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    map_score: float = 0.0  # Mean Average Precision
    hit_rate: float = 0.0
    avg_retrieval_time: float = 0.0
    total_queries: int = 0

@dataclass
class BenchmarkQuery:
    """Benchmark query with ground truth"""
    query_id: str
    query: str
    relevant_docs: List[str]
    relevance_grades: Optional[Dict[str, int]] = None  # For graded relevance

class RetrievalEvaluator:
    """
    Comprehensive evaluation framework for retrieval systems with:
    - Multiple evaluation metrics
    - Ablation studies
    - Performance benchmarking
    - Statistical significance testing
    """
    
    def __init__(self):
        self.benchmark_queries = []
        self.evaluation_results = []
        
    def load_benchmark_queries(self, queries_file: str) -> List[BenchmarkQuery]:
        """Load benchmark queries from JSON file"""
        queries_path = Path(queries_file)
        
        if not queries_path.exists():
            logger.warning(f"Benchmark file not found: {queries_file}")
            return []
        
        try:
            with open(queries_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            queries = []
            for item in data:
                query = BenchmarkQuery(
                    query_id=item.get('query_id', str(len(queries))),
                    query=item['query'],
                    relevant_docs=item['relevant_docs'],
                    relevance_grades=item.get('relevance_grades')
                )
                queries.append(query)
            
            self.benchmark_queries = queries
            logger.info(f"Loaded {len(queries)} benchmark queries")
            return queries
            
        except Exception as e:
            logger.error(f"Error loading benchmark queries: {e}")
            return []
    
    def create_sample_benchmark(self, num_queries: int = 20) -> List[BenchmarkQuery]:
        """Create sample benchmark queries for testing"""
        sample_queries = [
            {
                "query": "machine learning algorithms",
                "relevant_docs": ["ml_guide.pdf", "algorithms.pdf", "deep_learning.pdf"],
                "relevance_grades": {"ml_guide.pdf": 3, "algorithms.pdf": 2, "deep_learning.pdf": 2}
            },
            {
                "query": "neural network architecture",
                "relevant_docs": ["neural_networks.pdf", "deep_learning.pdf", "transformer.pdf"],
                "relevance_grades": {"neural_networks.pdf": 3, "deep_learning.pdf": 2, "transformer.pdf": 3}
            },
            {
                "query": "attention mechanism transformer",
                "relevant_docs": ["transformer.pdf", "attention.pdf", "bert.pdf"],
                "relevance_grades": {"transformer.pdf": 3, "attention.pdf": 3, "bert.pdf": 2}
            },
            {
                "query": "gradient descent optimization",
                "relevant_docs": ["optimization.pdf", "ml_guide.pdf", "training.pdf"],
                "relevance_grades": {"optimization.pdf": 3, "ml_guide.pdf": 2, "training.pdf": 2}
            },
            {
                "query": "natural language processing",
                "relevant_docs": ["nlp.pdf", "bert.pdf", "transformer.pdf"],
                "relevance_grades": {"nlp.pdf": 3, "bert.pdf": 2, "transformer.pdf": 2}
            }
        ]
        
        queries = []
        for i, item in enumerate(sample_queries[:num_queries]):
            query = BenchmarkQuery(
                query_id=f"q_{i+1}",
                query=item["query"],
                relevant_docs=item["relevant_docs"],
                relevance_grades=item["relevance_grades"]
            )
            queries.append(query)
        
        self.benchmark_queries = queries
        logger.info(f"Created {len(queries)} sample benchmark queries")
        return queries
    
    def evaluate_retrieval_system(
        self,
        retrieval_system,
        method_name: str,
        top_k_values: List[int] = None
    ) -> Dict[str, Any]:
        """Evaluate a retrieval system on benchmark queries"""
        if not self.benchmark_queries:
            raise ValueError("No benchmark queries loaded")
        
        if top_k_values is None:
            top_k_values = [5, 10, 20]
        
        logger.info(f"Evaluating {method_name} on {len(self.benchmark_queries)} queries")
        
        query_results = []
        total_time = 0.0
        
        for query in self.benchmark_queries:
            start_time = time.time()
            
            try:
                # Perform search
                search_results = retrieval_system.search(
                    query.query,
                    top_k=max(top_k_values)
                )
                
                retrieval_time = time.time() - start_time
                total_time += retrieval_time
                
                # Extract document IDs and scores
                retrieved_docs = [result.doc_id for result in search_results]
                relevance_scores = [result.final_score if hasattr(result, 'final_score') else result.score for result in search_results]
                
                # Calculate metrics for this query
                query_metrics = self._calculate_query_metrics(
                    query, retrieved_docs, relevance_scores, top_k_values
                )
                
                # Store result
                result = QueryResult(
                    query_id=query.query_id,
                    query=query.query,
                    retrieved_docs=retrieved_docs,
                    relevance_scores=relevance_scores,
                    ground_truth=query.relevant_docs,
                    metrics=query_metrics,
                    retrieval_time=retrieval_time,
                    method=method_name
                )
                query_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating query {query.query_id}: {e}")
                continue
        
        # Calculate aggregate metrics
        aggregate_metrics = self._aggregate_metrics(query_results)
        
        # Store results
        evaluation_result = {
            'method': method_name,
            'query_results': query_results,
            'aggregate_metrics': aggregate_metrics,
            'total_queries': len(query_results),
            'avg_retrieval_time': total_time / len(query_results) if query_results else 0.0
        }
        
        self.evaluation_results.append(evaluation_result)
        
        logger.info(f"Evaluation completed for {method_name}")
        self._print_metrics_summary(aggregate_metrics, method_name)
        
        return evaluation_result
    
    def _calculate_query_metrics(
        self,
        query: BenchmarkQuery,
        retrieved_docs: List[str],
        relevance_scores: List[float],
        top_k_values: List[int]
    ) -> Dict[str, float]:
        """Calculate metrics for a single query"""
        metrics = {}
        relevant_docs_set = set(query.relevant_docs)
        
        # Calculate metrics for different k values
        for k in top_k_values:
            retrieved_k = retrieved_docs[:k]
            relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_docs_set]
            
            # Recall@k
            recall_k = len(relevant_retrieved) / len(relevant_docs_set) if relevant_docs_set else 0.0
            metrics[f'recall@{k}'] = recall_k
            
            # Precision@k
            precision_k = len(relevant_retrieved) / k if k > 0 else 0.0
            metrics[f'precision@{k}'] = precision_k
            
            # NDCG@k
            if query.relevance_grades:
                ndcg_k = self._calculate_ndcg(retrieved_k, query.relevance_grades, k)
                metrics[f'ndcg@{k}'] = ndcg_k
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs_set:
                mrr = 1.0 / (i + 1)
                break
        metrics['mrr'] = mrr
        
        # Hit Rate (whether any relevant doc was found)
        hit_rate = 1.0 if any(doc in relevant_docs_set for doc in retrieved_docs) else 0.0
        metrics['hit_rate'] = hit_rate
        
        # Average Precision
        ap = self._calculate_average_precision(retrieved_docs, relevant_docs_set)
        metrics['average_precision'] = ap
        
        return metrics
    
    def _calculate_ndcg(
        self,
        retrieved_docs: List[str],
        relevance_grades: Dict[str, int],
        k: int
    ) -> float:
        """Calculate NDCG@k"""
        if not retrieved_docs or not relevance_grades:
            return 0.0
        
        # Get relevance scores for retrieved documents
        relevances = []
        for doc in retrieved_docs[:k]:
            relevances.append(relevance_grades.get(doc, 0))
        
        if not any(relevances):
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevances):
            dcg += (2**rel - 1) / math.log2(i + 2)
        
        # Calculate IDCG (perfect ranking)
        ideal_relevances = sorted(relevance_grades.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances):
            idcg += (2**rel - 1) / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_average_precision(
        self,
        retrieved_docs: List[str],
        relevant_docs_set: set
    ) -> float:
        """Calculate Average Precision"""
        if not relevant_docs_set:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs_set) if relevant_docs_set else 0.0
    
    def _aggregate_metrics(self, query_results: List[QueryResult]) -> EvaluationMetrics:
        """Aggregate metrics across all queries"""
        if not query_results:
            return EvaluationMetrics()
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        retrieval_times = []
        
        for result in query_results:
            for metric, value in result.metrics.items():
                all_metrics[metric].append(value)
            retrieval_times.append(result.retrieval_time)
        
        # Calculate averages
        metrics = EvaluationMetrics(
            recall_at_5=np.mean(all_metrics.get('recall@5', [0.0])),
            recall_at_10=np.mean(all_metrics.get('recall@10', [0.0])),
            precision_at_5=np.mean(all_metrics.get('precision@5', [0.0])),
            precision_at_10=np.mean(all_metrics.get('precision@10', [0.0])),
            mrr=np.mean(all_metrics.get('mrr', [0.0])),
            ndcg_at_5=np.mean(all_metrics.get('ndcg@5', [0.0])),
            ndcg_at_10=np.mean(all_metrics.get('ndcg@10', [0.0])),
            map_score=np.mean(all_metrics.get('average_precision', [0.0])),
            hit_rate=np.mean(all_metrics.get('hit_rate', [0.0])),
            avg_retrieval_time=np.mean(retrieval_times),
            total_queries=len(query_results)
        )
        
        return metrics
    
    def _print_metrics_summary(self, metrics: EvaluationMetrics, method_name: str):
        """Print evaluation metrics summary"""
        print(f"\n=== Evaluation Results for {method_name} ===")
        print(f"Queries evaluated: {metrics.total_queries}")
        print(f"Recall@5: {metrics.recall_at_5:.3f}")
        print(f"Recall@10: {metrics.recall_at_10:.3f}")
        print(f"Precision@5: {metrics.precision_at_5:.3f}")
        print(f"Precision@10: {metrics.precision_at_10:.3f}")
        print(f"MRR: {metrics.mrr:.3f}")
        print(f"NDCG@5: {metrics.ndcg_at_5:.3f}")
        print(f"NDCG@10: {metrics.ndcg_at_10:.3f}")
        print(f"MAP: {metrics.map_score:.3f}")
        print(f"Hit Rate: {metrics.hit_rate:.3f}")
        print(f"Avg Retrieval Time: {metrics.avg_retrieval_time:.3f}s")
        print("=" * 50)
    
    def compare_methods(self, methods_to_compare: List[str] = None) -> pd.DataFrame:
        """Compare multiple evaluation results"""
        if not self.evaluation_results:
            logger.warning("No evaluation results to compare")
            return pd.DataFrame()
        
        if methods_to_compare:
            results = [r for r in self.evaluation_results if r['method'] in methods_to_compare]
        else:
            results = self.evaluation_results
        
        # Create comparison dataframe
        comparison_data = []
        for result in results:
            metrics = result['aggregate_metrics']
            data = {
                'Method': result['method'],
                'Recall@5': metrics.recall_at_5,
                'Recall@10': metrics.recall_at_10,
                'Precision@5': metrics.precision_at_5,
                'Precision@10': metrics.precision_at_10,
                'MRR': metrics.mrr,
                'NDCG@5': metrics.ndcg_at_5,
                'NDCG@10': metrics.ndcg_at_10,
                'MAP': metrics.map_score,
                'Hit Rate': metrics.hit_rate,
                'Avg Time (s)': metrics.avg_retrieval_time
            }
            comparison_data.append(data)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """Plot comparison of different methods"""
        if not self.evaluation_results:
            logger.warning("No evaluation results to plot")
            return
        
        df = self.compare_methods()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Recall comparison
        axes[0, 0].bar(df['Method'], df['Recall@5'], alpha=0.7, label='Recall@5')
        axes[0, 0].bar(df['Method'], df['Recall@10'], alpha=0.7, label='Recall@10')
        axes[0, 0].set_title('Recall Comparison')
        axes[0, 0].set_ylabel('Recall')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision comparison
        axes[0, 1].bar(df['Method'], df['Precision@5'], alpha=0.7, label='Precision@5')
        axes[0, 1].bar(df['Method'], df['Precision@10'], alpha=0.7, label='Precision@10')
        axes[0, 1].set_title('Precision Comparison')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # NDCG comparison
        axes[1, 0].bar(df['Method'], df['NDCG@5'], alpha=0.7, label='NDCG@5')
        axes[1, 0].bar(df['Method'], df['NDCG@10'], alpha=0.7, label='NDCG@10')
        axes[1, 0].set_title('NDCG Comparison')
        axes[1, 0].set_ylabel('NDCG')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall metrics
        axes[1, 1].plot(df['Method'], df['MRR'], marker='o', label='MRR')
        axes[1, 1].plot(df['Method'], df['MAP'], marker='s', label='MAP')
        axes[1, 1].plot(df['Method'], df['Hit Rate'], marker='^', label='Hit Rate')
        axes[1, 1].set_title('Overall Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def run_ablation_study(
        self,
        base_system,
        ablation_configs: List[Dict[str, Any]],
        config_names: List[str]
    ) -> Dict[str, Any]:
        """Run ablation study with different configurations"""
        logger.info(f"Running ablation study with {len(ablation_configs)} configurations")
        
        ablation_results = {}
        
        for config, name in zip(ablation_configs, config_names):
            logger.info(f"Evaluating configuration: {name}")
            
            # Apply configuration to system
            # This would need to be customized based on your system's API
            try:
                # Configure system
                for key, value in config.items():
                    if hasattr(base_system, key):
                        setattr(base_system, key, value)
                
                # Evaluate
                result = self.evaluate_retrieval_system(base_system, name)
                ablation_results[name] = result
                
            except Exception as e:
                logger.error(f"Error in ablation study for {name}: {e}")
                continue
        
        logger.info("Ablation study completed")
        return ablation_results
    
    def generate_report(self, output_file: str):
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            logger.warning("No evaluation results to report")
            return
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create comparison dataframe
        df = self.compare_methods()
        
        # Generate HTML report
        html_content = self._generate_html_report(df)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def _generate_html_report(self, df: pd.DataFrame) -> str:
        """Generate HTML evaluation report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Retrieval System Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; }}
                .best {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <h1>Retrieval System Evaluation Report</h1>
            <h2>Summary</h2>
            <p>Evaluated {len(self.evaluation_results)} different methods on {len(self.benchmark_queries)} benchmark queries.</p>
            
            <h2>Metrics Comparison</h2>
            {df.to_html(index=False, classes='table')}
            
            <h2>Metrics Explanation</h2>
            <ul>
                <li><strong>Recall@k:</strong> Proportion of relevant documents retrieved in top-k results</li>
                <li><strong>Precision@k:</strong> Proportion of retrieved documents that are relevant in top-k results</li>
                <li><strong>MRR:</strong> Mean Reciprocal Rank - average of reciprocal ranks of first relevant result</li>
                <li><strong>NDCG@k:</strong> Normalized Discounted Cumulative Gain at k</li>
                <li><strong>MAP:</strong> Mean Average Precision across all queries</li>
                <li><strong>Hit Rate:</strong> Proportion of queries with at least one relevant result</li>
            </ul>
        </body>
        </html>
        """
        return html
    
    def export_results(self, output_file: str):
        """Export evaluation results to JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        export_data = []
        for result in self.evaluation_results:
            export_result = {
                'method': result['method'],
                'aggregate_metrics': asdict(result['aggregate_metrics']),
                'total_queries': result['total_queries'],
                'avg_retrieval_time': result['avg_retrieval_time'],
                'query_results': []
            }
            
            for query_result in result['query_results']:
                query_data = {
                    'query_id': query_result.query_id,
                    'query': query_result.query,
                    'retrieved_docs': query_result.retrieved_docs,
                    'relevance_scores': query_result.relevance_scores,
                    'ground_truth': query_result.ground_truth,
                    'metrics': query_result.metrics,
                    'retrieval_time': query_result.retrieval_time
                }
                export_result['query_results'].append(query_data)
            
            export_data.append(export_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Evaluation results exported to {output_path}")

class BenchmarkGenerator:
    """Generate benchmark queries from document collections"""
    
    def __init__(self, retrieval_system):
        self.retrieval_system = retrieval_system
    
    def generate_queries_from_documents(
        self,
        num_queries: int = 50,
        query_types: List[str] = None
    ) -> List[BenchmarkQuery]:
        """Generate benchmark queries from indexed documents"""
        if query_types is None:
            query_types = ['keyword', 'phrase', 'question']
        
        # This is a simplified version - in practice you'd use more sophisticated methods
        generated_queries = []
        
        # Sample implementation
        sample_queries = [
            "machine learning algorithms",
            "neural network architecture", 
            "deep learning methods",
            "natural language processing",
            "computer vision techniques"
        ]
        
        for i, query in enumerate(sample_queries[:num_queries]):
            # Get relevant documents by running the query
            results = self.retrieval_system.search(query, top_k=10)
            relevant_docs = [r.doc_id for r in results[:3]]  # Top 3 as relevant
            
            benchmark_query = BenchmarkQuery(
                query_id=f"gen_{i+1}",
                query=query,
                relevant_docs=relevant_docs
            )
            generated_queries.append(benchmark_query)
        
        logger.info(f"Generated {len(generated_queries)} benchmark queries")
        return generated_queries
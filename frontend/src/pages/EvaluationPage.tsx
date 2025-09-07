import React, { useState, useEffect } from 'react';
import { useApi } from '../contexts/ApiContext';
import { 
  BarChart3, 
  Target, 
  TrendingUp, 
  Clock, 
  CheckCircle,
  AlertCircle,
  Info,
  PlayCircle,
  FileText
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadialBarChart, RadialBar } from 'recharts';

const EvaluationPage: React.FC = () => {
  const { evaluateSystem, evaluationResults, loading, error, listIndexes } = useApi();
  
  const [formData, setFormData] = useState({
    index_name: '',
    benchmark_file: '',
    create_sample_benchmark: true,
    num_sample_queries: 20
  });

  const [availableIndexes, setAvailableIndexes] = useState<string[]>([]);

  // Fetch available indexes on component mount
  useEffect(() => {
    const fetchIndexes = async () => {
      try {
        const data = await listIndexes();
        setAvailableIndexes(data.indexes || []);
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.index_name.trim()) {
      alert('Please enter an index name');
      return;
    }

    if (!formData.create_sample_benchmark && !formData.benchmark_file.trim()) {
      alert('Please provide a benchmark file or enable sample benchmark creation');
      return;
    }

    try {
      await evaluateSystem(formData);
    } catch (err) {
      console.error('Evaluation failed:', err);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return '#10b981'; // green
    if (score >= 0.6) return '#3b82f6'; // blue
    if (score >= 0.4) return '#f59e0b'; // yellow
    return '#6b7280'; // gray
  };

  const formatMetricName = (metric: string) => {
    return metric
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase())
      .replace('At', '@');
  };

  const prepareChartData = (metrics: any) => {
    if (!metrics) return [];
    
    return Object.entries(metrics)
      .filter(([key]) => !key.includes('time'))
      .map(([key, value]) => ({
        name: formatMetricName(key),
        value: Number(value),
        fill: getScoreColor(Number(value))
      }));
  };

  const prepareRadialData = (metrics: any) => {
    if (!metrics) return [];
    
    const keyMetrics = ['recall_at_10', 'precision_at_10', 'mrr', 'ndcg_at_10'];
    return keyMetrics
      .filter(key => metrics[key] !== undefined)
      .map((key, index) => ({
        name: formatMetricName(key),
        value: Number(metrics[key]) * 100,
        fill: [`#3b82f6`, `#10b981`, `#f59e0b`, `#8b5cf6`][index]
      }));
  };

  return (
    <div className="space-y-6">
      <div className="glass p-6">
        <h1 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
          <BarChart3 className="w-6 h-6" />
          System Evaluation
        </h1>
        <p className="text-gray mb-6">
          Evaluate your retrieval system's performance using comprehensive metrics including Recall, Precision, MRR, nDCG, and MAP.
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-2 gap-4">
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

            {/* Sample Queries Count */}
            <div>
              <label className="block text-white font-semibold mb-2">
                Sample Queries Count
              </label>
              <select
                className="input"
                value={formData.num_sample_queries}
                onChange={(e) => handleInputChange('num_sample_queries', parseInt(e.target.value))}
                disabled={loading || !formData.create_sample_benchmark}
              >
                <option value={10}>10 queries</option>
                <option value={20}>20 queries</option>
                <option value={50}>50 queries</option>
                <option value={100}>100 queries</option>
              </select>
            </div>
          </div>

          {/* Benchmark Options */}
          <div className="glass p-4">
            <h3 className="text-white font-semibold mb-3">Benchmark Configuration</h3>
            
            <div className="space-y-3">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="benchmark_type"
                  checked={formData.create_sample_benchmark}
                  onChange={() => handleInputChange('create_sample_benchmark', true)}
                  disabled={loading}
                  className="accent-blue-500"
                />
                <span className="text-white">Create sample benchmark automatically</span>
              </label>
              <p className="text-gray text-sm ml-6">
                Generate synthetic queries from your document collection for evaluation
              </p>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="benchmark_type"
                  checked={!formData.create_sample_benchmark}
                  onChange={() => handleInputChange('create_sample_benchmark', false)}
                  disabled={loading}
                  className="accent-blue-500"
                />
                <span className="text-white">Use custom benchmark file</span>
              </label>
              
              {!formData.create_sample_benchmark && (
                <div className="ml-6">
                  <input
                    type="text"
                    className="input"
                    placeholder="Path to benchmark file (JSON format)"
                    value={formData.benchmark_file}
                    onChange={(e) => handleInputChange('benchmark_file', e.target.value)}
                    disabled={loading}
                  />
                  <p className="text-gray text-sm mt-1">
                    JSON file containing query-document relevance pairs
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Information Panel */}
          <div className="glass p-4 border-l-4 border-blue-500">
            <div className="flex items-start gap-2">
              <Info className="w-4 h-4 text-blue mt-0.5" />
              <div>
                <h4 className="text-white font-semibold mb-1">Evaluation Metrics</h4>
                <ul className="text-gray text-sm space-y-1">
                  <li><strong>Recall@k:</strong> Fraction of relevant documents retrieved in top-k results</li>
                  <li><strong>Precision@k:</strong> Fraction of retrieved documents that are relevant</li>
                  <li><strong>MRR:</strong> Mean Reciprocal Rank of the first relevant document</li>
                  <li><strong>nDCG@k:</strong> Normalized Discounted Cumulative Gain (ranking quality)</li>
                  <li><strong>MAP:</strong> Mean Average Precision across all queries</li>
                  <li><strong>Hit Rate:</strong> Percentage of queries with at least one relevant result</li>
                </ul>
              </div>
            </div>
          </div>

          <button
            type="submit"
            className="button w-full flex items-center justify-center gap-2"
            disabled={loading}
          >
            {loading ? (
              <>
                <div className="loading"></div>
                Evaluating System...
              </>
            ) : (
              <>
                <PlayCircle className="w-4 h-4" />
                Start Evaluation
              </>
            )}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-3 bg-red-500 bg-opacity-20 border border-red-500 border-opacity-50 rounded text-red-200 flex items-center gap-2">
            <AlertCircle className="w-4 h-4" />
            {error}
          </div>
        )}
      </div>

      {/* Evaluation Results */}
      {evaluationResults && (
        <div className="space-y-6">
          {/* Results Overview */}
          <div className="glass p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green" />
                Evaluation Results
              </h2>
              <div className="text-sm text-gray">
                Method: {evaluationResults.evaluation_method}
              </div>
            </div>

            <div className="grid grid-3 gap-4 mb-6">
              <div className="card">
                <div className="flex items-center gap-2 mb-2">
                  <FileText className="w-4 h-4 text-blue" />
                  <span className="text-white font-semibold">Benchmark</span>
                </div>
                <div className="text-2xl font-bold text-white">
                  {evaluationResults.benchmark_queries}
                </div>
                <div className="text-gray text-sm">Queries Tested</div>
              </div>

              <div className="card">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-4 h-4 text-green" />
                  <span className="text-white font-semibold">Overall Score</span>
                </div>
                <div className="text-2xl font-bold text-green">
                  {evaluationResults.metrics.mrr?.toFixed(3) || 'N/A'}
                </div>
                <div className="text-gray text-sm">Mean Reciprocal Rank</div>
              </div>

              <div className="card">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-yellow" />
                  <span className="text-white font-semibold">Performance</span>
                </div>
                <div className="text-2xl font-bold text-yellow">
                  {evaluationResults.metrics.avg_retrieval_time?.toFixed(3) || 'N/A'}s
                </div>
                <div className="text-gray text-sm">Avg Retrieval Time</div>
              </div>
            </div>
          </div>

          {/* Key Metrics Radial Chart */}
          <div className="glass p-6">
            <h3 className="text-xl font-bold text-white mb-4">Key Performance Metrics</h3>
            <div className="grid grid-2 gap-6">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <RadialBarChart data={prepareRadialData(evaluationResults.metrics)}>
                    <RadialBar dataKey="value" cornerRadius={10} fill="#8884d8" />
                    <Tooltip 
                      formatter={(value: any) => [`${(value / 100).toFixed(3)}`, 'Score']}
                      labelFormatter={(label) => `Metric: ${label}`}
                    />
                  </RadialBarChart>
                </ResponsiveContainer>
              </div>

              <div className="space-y-3">
                {prepareRadialData(evaluationResults.metrics).map((metric, index) => (
                  <div key={metric.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div 
                        className="w-3 h-3 rounded"
                        style={{ backgroundColor: metric.fill }}
                      />
                      <span className="text-white text-sm">{metric.name}</span>
                    </div>
                    <span className="text-white font-semibold">
                      {(metric.value / 100).toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Detailed Metrics Bar Chart */}
          <div className="glass p-6">
            <h3 className="text-xl font-bold text-white mb-4">Detailed Metrics Comparison</h3>
            <div className="h-64 mb-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={prepareChartData(evaluationResults.metrics)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="name" stroke="rgba(255,255,255,0.7)" fontSize={12} />
                  <YAxis stroke="rgba(255,255,255,0.7)" fontSize={12} />
                  <Tooltip 
                    formatter={(value: any) => [value.toFixed(3), 'Score']}
                    labelFormatter={(label) => `Metric: ${label}`}
                    contentStyle={{
                      backgroundColor: 'rgba(0,0,0,0.8)',
                      border: '1px solid rgba(255,255,255,0.2)',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Metrics Table */}
          <div className="glass p-6">
            <h3 className="text-xl font-bold text-white mb-4">Complete Metrics Breakdown</h3>
            <div className="space-y-2">
              {Object.entries(evaluationResults.metrics).map(([key, value]) => (
                <div key={key} className="metric">
                  <span className="text-white font-semibold">{formatMetricName(key)}</span>
                  <span className={`font-bold ${getScoreColor(Number(value)) === '#10b981' ? 'text-green' : 
                                                getScoreColor(Number(value)) === '#3b82f6' ? 'text-blue' :
                                                getScoreColor(Number(value)) === '#f59e0b' ? 'text-yellow' : 'text-gray'}`}>
                    {typeof value === 'number' ? 
                      (key.includes('time') ? `${(value as number).toFixed(3)}s` : (value as number).toFixed(3)) 
                      : String(value)
                    }
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Evaluation Summary */}
          <div className="glass p-6">
            <h3 className="text-xl font-bold text-white mb-4">Evaluation Summary</h3>
            <div className="grid grid-2 gap-4">
              <div>
                <h4 className="font-semibold text-white mb-2">Performance Highlights</h4>
                <ul className="text-sm space-y-1">
                  <li className="flex justify-between">
                    <span className="text-gray">Hit Rate:</span>
                    <span className="text-green">{(evaluationResults.metrics.hit_rate * 100).toFixed(1)}%</span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-gray">Best Recall:</span>
                    <span className="text-blue">
                      Recall@10: {evaluationResults.metrics.recall_at_10?.toFixed(3) || 'N/A'}
                    </span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-gray">Best Precision:</span>
                    <span className="text-blue">
                      Precision@5: {evaluationResults.metrics.precision_at_5?.toFixed(3) || 'N/A'}
                    </span>
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-white mb-2">System Information</h4>
                <ul className="text-sm space-y-1">
                  <li className="flex justify-between">
                    <span className="text-gray">Total Queries:</span>
                    <span className="text-white">{evaluationResults.total_queries_evaluated}</span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-gray">Method:</span>
                    <span className="text-white">{evaluationResults.evaluation_method}</span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-gray">Avg Response Time:</span>
                    <span className="text-white">{evaluationResults.metrics.avg_retrieval_time?.toFixed(3) || 'N/A'}s</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EvaluationPage;
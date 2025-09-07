import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useApi } from '../contexts/ApiContext';
import { 
  Search, 
  FileText, 
  BarChart3, 
  MessageSquare, 
  Zap, 
  Brain, 
  Target,
  Activity,
  CheckCircle,
  Sparkles,
  TrendingUp,
  Database,
  Cpu
} from 'lucide-react';

const HomePage: React.FC = () => {
  const { systemStatus, getSystemStatus, loading } = useApi();

  useEffect(() => {
    getSystemStatus();
  }, [getSystemStatus]);

  const features = [
    {
      icon: Zap,
      title: 'Hybrid BM25 + Dense Retrieval',
      description: 'State-of-the-art combination of keyword and semantic search for optimal recall and precision.',
      gradient: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
    },
    {
      icon: Brain,
      title: 'Cross-encoder Reranking',
      description: 'Advanced neural reranking using MS-MARCO models for superior relevance scoring.',
      gradient: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)'
    },
    {
      icon: Target,
      title: 'Structure-aware Chunking',
      description: 'Intelligent document segmentation that preserves context and semantic boundaries.',
      gradient: 'linear-gradient(135deg, #10b981 0%, #047857 100%)'
    },
    {
      icon: Activity,
      title: 'Multi-format Processing',
      description: 'Support for PDF, DOCX, TXT, MD, EPUB, and RTF with OCR capabilities.',
      gradient: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)'
    }
  ];

  const quickActions = [
    {
      to: '/index',
      icon: Database,
      title: 'Create Index',
      description: 'Index your documents for powerful search capabilities',
      gradient: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)'
    },
    {
      to: '/search',
      icon: Search,
      title: 'Search Documents',
      description: 'Find relevant information across your document collection',
      gradient: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)'
    },
    {
      to: '/evaluate',
      icon: TrendingUp,
      title: 'Evaluate System',
      description: 'Benchmark retrieval performance and quality metrics',
      gradient: 'linear-gradient(135deg, #10b981 0%, #047857 100%)'
    },
    {
      to: '/chat',
      icon: MessageSquare,
      title: 'Chat Interface',
      description: 'Interactive conversation with your documents',
      gradient: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
    }
  ];

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      {/* Hero Section */}
      <div className="card" style={{
        textAlign: 'center',
        padding: '48px',
        background: 'var(--gradient-card)',
        border: '2px solid var(--border-light)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          width: '80px',
          height: '80px',
          background: 'var(--gradient-accent)',
          borderRadius: '50%',
          opacity: 0.1,
          animation: 'pulse 3s infinite'
        }} />
        <div style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          width: '60px',
          height: '60px',
          background: 'var(--gradient-accent)',
          borderRadius: '50%',
          opacity: 0.1,
          animation: 'pulse 3s infinite 1s'
        }} />
        
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '16px', marginBottom: '24px' }}>
          <div style={{
            background: 'var(--gradient-accent)',
            padding: '16px',
            borderRadius: '20px',
            boxShadow: 'var(--shadow-medium)'
          }}>
            <Sparkles style={{ width: '40px', height: '40px', color: 'white' }} />
          </div>
        </div>
        
        <h1 className="heading-1" style={{ fontSize: '3rem', marginBottom: '16px' }}>
          Advanced Document Search
        </h1>
        <p className="text-body" style={{ fontSize: '1.25rem', marginBottom: '32px', maxWidth: '600px', margin: '0 auto 32px' }}>
          State-of-the-art hybrid retrieval system with BM25, dense embeddings, and neural reranking
        </p>
        
        {systemStatus && (
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            background: '#ecfdf5',
            color: '#10b981',
            padding: '12px 20px',
            borderRadius: '12px',
            border: '1px solid #d1fae5',
            fontWeight: '600'
          }}>
            <CheckCircle style={{ width: '20px', height: '20px' }} />
            <span>System Online - Ready for Action</span>
          </div>
        )}
      </div>

      {/* System Status */}
      {systemStatus && (
        <div className="card">
          <h2 className="heading-2" style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
            <Cpu style={{ width: '28px', height: '28px', color: 'var(--bg-accent)' }} />
            System Status
          </h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
            <div className="card" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-light)' }}>
              <h3 className="heading-3" style={{ fontSize: '1.125rem', marginBottom: '16px', color: '#000000' }}>
                Configuration
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="text-small">Server Port:</span>
                  <span style={{ 
                    background: 'var(--bg-accent)', 
                    color: 'white', 
                    padding: '4px 8px', 
                    borderRadius: '6px',
                    fontSize: '12px',
                    fontWeight: '600'
                  }}>
                    {systemStatus.configuration?.server_port || 'N/A'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="text-small">Storage Directory:</span>
                  <span className="text-small" style={{ textAlign: 'right', maxWidth: '200px', wordBreak: 'break-all' }}>
                    {systemStatus.configuration?.storage_dir || 'N/A'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="text-small">Supported Formats:</span>
                  <span style={{ 
                    background: 'var(--bg-accent)', 
                    color: 'white', 
                    padding: '4px 8px', 
                    borderRadius: '6px',
                    fontSize: '12px',
                    fontWeight: '600'
                  }}>
                    {systemStatus.configuration?.supported_formats?.length || 0}
                  </span>
                </div>
              </div>
            </div>

            <div className="card" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-light)' }}>
              <h3 className="heading-3" style={{ fontSize: '1.125rem', marginBottom: '16px', color: '#000000' }}>
                System Statistics
              </h3>
              {systemStatus.system_statistics && Object.keys(systemStatus.system_statistics).length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {Object.entries(systemStatus.system_statistics).map(([key, value]) => (
                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span className="text-small">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</span>
                      <span style={{ fontWeight: '600', color: '#000000' }}>{String(value)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-small" style={{ fontStyle: 'italic' }}>
                  No statistics available. Create an index to see system metrics.
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Features */}
      <div className="card">
        <h2 className="heading-2" style={{ marginBottom: '24px' }}>Key Features</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          {features.map(({ icon: Icon, title, description, gradient }, index) => (
            <div key={index} className="card slide-in" style={{
              background: 'var(--bg-card)',
              border: '2px solid var(--border-light)',
              transition: 'all 0.3s ease',
              cursor: 'pointer',
              animationDelay: `${index * 0.1}s`
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLElement).style.transform = 'translateY(-8px)';
              (e.currentTarget as HTMLElement).style.boxShadow = 'var(--shadow-large)';
              (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-medium)';
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLElement).style.transform = 'translateY(0)';
              (e.currentTarget as HTMLElement).style.boxShadow = 'var(--shadow-soft)';
              (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-light)';
            }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
                <div style={{
                  background: gradient,
                  padding: '12px',
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: 'var(--shadow-soft)'
                }}>
                  <Icon style={{ width: '24px', height: '24px', color: 'white' }} />
                </div>
                <div style={{ flex: 1 }}>
                  <h3 className="heading-3" style={{ fontSize: '1.125rem', marginBottom: '8px' }}>{title}</h3>
                  <p className="text-body" style={{ fontSize: '0.875rem', lineHeight: '1.5' }}>{description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h2 className="heading-2" style={{ marginBottom: '24px' }}>Quick Actions</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
          {quickActions.map(({ to, icon: Icon, title, description, gradient }) => (
            <Link
              key={to}
              to={to}
              style={{ textDecoration: 'none' }}
              className="slide-in"
            >
              <div className="card" style={{
                background: 'var(--bg-card)',
                border: '2px solid var(--border-light)',
                transition: 'all 0.3s ease',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.transform = 'translateY(-6px) scale(1.02)';
                (e.currentTarget as HTMLElement).style.boxShadow = 'var(--shadow-large)';
                (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-medium)';
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.transform = 'translateY(0) scale(1)';
                (e.currentTarget as HTMLElement).style.boxShadow = 'var(--shadow-soft)';
                (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-light)';
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <div style={{
                    background: gradient,
                    padding: '16px',
                    borderRadius: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: 'var(--shadow-medium)'
                  }}>
                    <Icon style={{ width: '28px', height: '28px', color: 'white' }} />
                  </div>
                  <div>
                    <h3 className="heading-3" style={{ fontSize: '1.125rem', marginBottom: '4px', color: '#000000' }}>
                      {title}
                    </h3>
                    <p className="text-body" style={{ fontSize: '0.875rem', margin: 0 }}>{description}</p>
                  </div>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {loading && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '20px' }}>
          <div className="spinner" />
        </div>
      )}
    </div>
  );
};

export default HomePage;
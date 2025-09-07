import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Search, FileText, BarChart3, MessageSquare, Home, Sparkles } from 'lucide-react';

const Navigation: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/index', label: 'Index', icon: FileText },
    { path: '/search', label: 'Search', icon: Search },
    { path: '/evaluate', label: 'Evaluate', icon: BarChart3 },
    { path: '/chat', label: 'Chat', icon: MessageSquare },
  ];

  return (
    <nav style={{
      background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
      padding: '20px 0',
      borderRadius: '0 0 24px 24px',
      boxShadow: 'var(--shadow-large)',
      marginBottom: '30px',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        padding: '0 20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            background: 'rgba(255, 255, 255, 0.2)',
            padding: '12px',
            borderRadius: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Sparkles style={{ width: '32px', height: '32px', color: 'white' }} />
          </div>
          <div>
            <h1 className="heading-2" style={{ 
              color: 'white', 
              margin: 0, 
              fontSize: '1.75rem',
              fontWeight: '800',
              textShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}>
              Advanced Document Search
            </h1>
            <p style={{ 
              color: 'rgba(255, 255, 255, 0.8)', 
              margin: 0,
              fontSize: '14px',
              fontWeight: '500'
            }}>
              AI-Powered Search & Analytics
            </p>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {navItems.map(({ path, label, icon: Icon }) => (
            <Link
              key={path}
              to={path}
              className="fade-in"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '12px 16px',
                borderRadius: '12px',
                textDecoration: 'none',
                fontWeight: '600',
                fontSize: '14px',
                transition: 'all 0.3s ease',
                background: location.pathname === path 
                  ? 'rgba(255, 255, 255, 0.25)' 
                  : 'transparent',
                color: location.pathname === path 
                  ? 'white' 
                  : 'rgba(255, 255, 255, 0.8)',
                border: location.pathname === path 
                  ? '2px solid rgba(255, 255, 255, 0.3)' 
                  : '2px solid transparent',
                backdropFilter: location.pathname === path ? 'blur(10px)' : 'none'
              }}
              onMouseEnter={(e) => {
                if (location.pathname !== path) {
                  (e.target as HTMLElement).style.background = 'rgba(255, 255, 255, 0.15)';
                  (e.target as HTMLElement).style.color = 'white';
                  (e.target as HTMLElement).style.transform = 'translateY(-2px)';
                }
              }}
              onMouseLeave={(e) => {
                if (location.pathname !== path) {
                  (e.target as HTMLElement).style.background = 'transparent';
                  (e.target as HTMLElement).style.color = 'rgba(255, 255, 255, 0.8)';
                  (e.target as HTMLElement).style.transform = 'translateY(0)';
                }
              }}
            >
              <Icon style={{ width: '18px', height: '18px' }} />
              <span style={{ 
                display: window.innerWidth > 768 ? 'inline' : 'none'
              }}>{label}</span>
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
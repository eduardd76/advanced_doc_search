import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import HomePage from './pages/HomePage';
import IndexPage from './pages/IndexPage';
import SearchPage from './pages/SearchPage';
import EvaluationPage from './pages/EvaluationPage';
import ChatPage from './pages/ChatPage';
import { ApiProvider } from './contexts/ApiContext';
import './styles/globals.css';

function App() {
  return (
    <ApiProvider>
      <Router>
        <div className="min-h-screen" style={{ background: 'var(--gradient-primary)' }}>
          <Navigation />
          <main style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/index" element={<IndexPage />} />
              <Route path="/search" element={<SearchPage />} />
              <Route path="/evaluate" element={<EvaluationPage />} />
              <Route path="/chat" element={<ChatPage />} />
            </Routes>
          </main>
        </div>
      </Router>
    </ApiProvider>
  );
}

export default App;
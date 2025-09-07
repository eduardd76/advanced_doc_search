import React, { useState, useRef, useEffect } from 'react';
import { useApi } from '../contexts/ApiContext';
import { 
  MessageSquare, 
  Send, 
  User, 
  Bot, 
  FileText, 
  Settings,
  Key,
  Brain,
  ExternalLink,
  Copy,
  Check
} from 'lucide-react';

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  message: string;
  timestamp: Date;
  sources?: any[];
  contextUsed?: number;
  synthesisMode?: boolean;
}

const ChatPage: React.FC = () => {
  const { chatWithDocuments, loading, error } = useApi();
  
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [settings, setSettings] = useState({
    index_name: '',
    api_key: '',
    model: 'gpt-3.5-turbo',
    use_synthesis: true
  });
  const [showSettings, setShowSettings] = useState(false);
  const [copiedStates, setCopiedStates] = useState<{[key: string]: boolean}>({});
  const [availableIndexes, setAvailableIndexes] = useState<string[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const fetchIndexes = async () => {
      try {
        console.log('Fetching indexes for chat...');
        const response = await fetch('http://localhost:8003/api/indexes/list');
        const data = await response.json();
        console.log('Received indexes for chat:', data);
        setAvailableIndexes(data.indexes || []);
      } catch (err) {
        console.error('Failed to fetch indexes for chat:', err);
      }
    };
    fetchIndexes();
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!currentMessage.trim()) return;
    if (!settings.index_name.trim()) {
      alert('Please enter an index name in settings');
      return;
    }

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      message: currentMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');

    try {
      const response = await chatWithDocuments({
        message: currentMessage,
        index_name: settings.index_name,
        api_key: settings.api_key || undefined,
        model: settings.model,
        use_synthesis: settings.use_synthesis
      });

      const botMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        message: response.response,
        timestamp: new Date(),
        sources: response.sources,
        contextUsed: response.context_used,
        synthesisMode: response.synthesis_mode
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      console.error('Chat failed:', err);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        message: 'Sorry, I encountered an error while processing your request. Please check your settings and try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const copyToClipboard = async (text: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedStates(prev => ({ ...prev, [messageId]: true }));
      setTimeout(() => {
        setCopiedStates(prev => ({ ...prev, [messageId]: false }));
      }, 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const formatTimestamp = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="flex flex-col h-screen max-h-[calc(100vh-12rem)]">
      {/* Header */}
      <div className="glass p-4 mb-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              <MessageSquare className="w-6 h-6" />
              Chat with Documents
            </h1>
            <p className="text-gray text-sm">
              Have conversations with your indexed documents using AI-powered search and synthesis
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="button flex items-center gap-2"
            >
              <Settings className="w-4 h-4" />
              Settings
            </button>
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="button bg-red-500 bg-opacity-20 border-red-500"
              >
                Clear Chat
              </button>
            )}
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="mt-4 p-4 glass border-t border-white border-opacity-10">
            <div className="grid grid-2 gap-4">
              <div>
                <label className="block text-white font-semibold mb-2">
                  Index Name *
                </label>
                <select
                  className="input"
                  value={settings.index_name}
                  onChange={(e) => setSettings(prev => ({ ...prev, index_name: e.target.value }))}
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

              <div>
                <label className="block text-white font-semibold mb-2">
                  OpenAI API Key (Optional)
                </label>
                <div className="relative">
                  <Key className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray" />
                  <input
                    type="password"
                    className="input pl-10"
                    placeholder="sk-..."
                    value={settings.api_key}
                    onChange={(e) => setSettings(prev => ({ ...prev, api_key: e.target.value }))}
                  />
                </div>
                <p className="text-gray text-xs mt-1">
                  Required for advanced synthesis. Without it, you'll get basic document context.
                </p>
              </div>

              <div>
                <label className="block text-white font-semibold mb-2">
                  Model
                </label>
                <select
                  className="input"
                  value={settings.model}
                  onChange={(e) => setSettings(prev => ({ ...prev, model: e.target.value }))}
                >
                  <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                  <option value="gpt-4">GPT-4</option>
                  <option value="gpt-4-turbo">GPT-4 Turbo</option>
                </select>
              </div>

              <div className="flex items-center">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.use_synthesis}
                    onChange={(e) => setSettings(prev => ({ ...prev, use_synthesis: e.target.checked }))}
                    className="accent-blue-500"
                  />
                  <span className="text-white">Enable AI Synthesis</span>
                </label>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Chat Messages */}
      <div className="flex-1 glass p-4 overflow-y-auto space-y-4 mb-4">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <Bot className="w-16 h-16 text-gray mx-auto mb-4" />
            <h3 className="text-xl font-bold text-white mb-2">Start a Conversation</h3>
            <p className="text-gray mb-4">
              Ask questions about your indexed documents. I'll search through them and provide detailed answers.
            </p>
            <div className="grid grid-2 gap-2 max-w-md mx-auto text-sm">
              <div className="glass p-3 rounded">
                <p className="text-white">"What are the main topics in my documents?"</p>
              </div>
              <div className="glass p-3 rounded">
                <p className="text-white">"Find information about machine learning"</p>
              </div>
              <div className="glass p-3 rounded">
                <p className="text-white">"Summarize the key findings in my research papers"</p>
              </div>
              <div className="glass p-3 rounded">
                <p className="text-white">"What do my documents say about data analysis?"</p>
              </div>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
            >
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                message.type === 'user' 
                  ? 'bg-blue-500 bg-opacity-20' 
                  : 'bg-green-500 bg-opacity-20'
              }`}>
                {message.type === 'user' ? (
                  <User className="w-4 h-4 text-blue" />
                ) : (
                  <Bot className="w-4 h-4 text-green" />
                )}
              </div>

              <div className={`flex-1 max-w-3xl ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-white font-semibold text-sm">
                    {message.type === 'user' ? 'You' : 'Assistant'}
                  </span>
                  <span className="text-gray text-xs">
                    {formatTimestamp(message.timestamp)}
                  </span>
                  {message.synthesisMode && (
                    <div className="flex items-center gap-1">
                      <Brain className="w-3 h-3 text-blue" />
                      <span className="text-blue text-xs">AI Synthesis</span>
                    </div>
                  )}
                </div>

                <div className={`card ${message.type === 'user' ? 'bg-blue-500 bg-opacity-10' : 'bg-white bg-opacity-5'}`}>
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-white leading-relaxed text-sm flex-1">
                      {message.message}
                    </p>
                    <button
                      onClick={() => copyToClipboard(message.message, message.id)}
                      className="text-gray hover:text-white transition-colors flex-shrink-0"
                      title="Copy message"
                    >
                      {copiedStates[message.id] ? (
                        <Check className="w-4 h-4 text-green" />
                      ) : (
                        <Copy className="w-4 h-4" />
                      )}
                    </button>
                  </div>

                  {/* Context Information */}
                  {message.type === 'bot' && message.contextUsed && (
                    <div className="mt-3 pt-3 border-t border-white border-opacity-10">
                      <div className="flex items-center justify-between text-xs mb-2">
                        <span className="text-gray flex items-center gap-1">
                          <FileText className="w-3 h-3" />
                          {message.contextUsed} documents used as context
                        </span>
                        {message.synthesisMode && (
                          <span className="text-blue">Advanced synthesis enabled</span>
                        )}
                      </div>

                      {/* Sources */}
                      {message.sources && message.sources.length > 0 && (
                        <div className="space-y-2">
                          <h4 className="text-white font-semibold text-xs">Sources:</h4>
                          {message.sources.slice(0, 3).map((source, index) => (
                            <div key={index} className="bg-white bg-opacity-5 rounded p-2">
                              <div className="flex items-center justify-between">
                                <span className="text-white text-xs font-mono">
                                  {source.doc_id}
                                </span>
                                <span className="text-green text-xs">
                                  Score: {source.score.toFixed(3)}
                                </span>
                              </div>
                              {source.metadata && (
                                <div className="text-gray text-xs mt-1">
                                  {Object.entries(source.metadata).slice(0, 2).map(([key, value]) => (
                                    <span key={key} className="mr-2">
                                      {key}: {String(value).slice(0, 30)}
                                      {String(value).length > 30 ? '...' : ''}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="glass p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            className="input flex-1"
            placeholder={settings.index_name ? "Ask a question about your documents..." : "Enter index name in settings first"}
            value={currentMessage}
            onChange={(e) => setCurrentMessage(e.target.value)}
            disabled={loading || !settings.index_name.trim()}
          />
          <button
            type="submit"
            className="button flex items-center gap-2"
            disabled={loading || !currentMessage.trim() || !settings.index_name.trim()}
          >
            {loading ? (
              <div className="loading w-4 h-4"></div>
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </form>

        {error && (
          <div className="mt-2 text-red-200 text-sm">
            Error: {error}
          </div>
        )}

        <div className="flex items-center justify-between text-xs text-gray mt-2">
          <span>
            {settings.index_name ? `Using index: ${settings.index_name}` : 'No index selected'}
          </span>
          <span>
            {settings.api_key ? 'AI synthesis available' : 'Basic context mode'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
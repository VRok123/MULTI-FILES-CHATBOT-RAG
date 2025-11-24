// src/App.js
import React, { useState, useEffect } from 'react';
import UploadPDFs from './components/UploadPDFs';
import ChatBox from './components/ChatBox';
import DocumentAnalysis from './components/DocumentAnalysis';
import Login from './components/Login';
import UserSessions from './components/UserSessions';
import { Moon, Sun } from 'lucide-react';
import './index.css';

function App() {
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('theme') === 'dark');
  const [sessionId, setSessionId] = useState(() => localStorage.getItem('sessionId') || null);
  const [loadedMessages, setLoadedMessages] = useState(() => {
    const saved = localStorage.getItem('loadedMessages');
    return saved ? JSON.parse(saved) : [];
  });
  const [user, setUser] = useState(() => localStorage.getItem('userId') || null);
  const [sessionToken, setSessionToken] = useState(() => localStorage.getItem('sessionToken') || null);
  const [activeTab, setActiveTab] = useState('chat');

  // 🌙 Theme handling
  useEffect(() => {
    const root = window.document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [darkMode]);

  // 🧠 Persist session data (sessionId + messages)
  useEffect(() => {
    if (sessionId) localStorage.setItem('sessionId', sessionId);
    localStorage.setItem('loadedMessages', JSON.stringify(loadedMessages));
  }, [sessionId, loadedMessages]);

  // 🧩 Login handler
  const handleLogin = (token, userId) => {
    setSessionToken(token);
    setUser(userId);
    localStorage.setItem('sessionToken', token);
    localStorage.setItem('userId', userId);
  };

  // 🚪 Logout handler
  const handleLogout = () => {
    setSessionToken(null);
    setUser(null);
    setSessionId(null);
    setLoadedMessages([]);
    localStorage.removeItem('sessionToken');
    localStorage.removeItem('userId');
    localStorage.removeItem('sessionId');
    localStorage.removeItem('loadedMessages');
  };

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-white transition-colors duration-300">
      <nav className="flex justify-between items-center px-6 py-4 shadow-md bg-gray-100 dark:bg-gray-800 rounded-b-2xl">
        <div className="flex items-center gap-2">
          <span className="text-3xl">📚</span>
          <h1 className="text-2xl font-bold">ASKEASE</h1>
        </div>
        <div className="flex items-center gap-4">
          {user ? (
            <div className="flex items-center gap-2">
              <span className="hidden sm:inline">User: {user}</span>
              <button
                onClick={handleLogout}
                className="px-3 py-1 bg-red-600 text-white rounded text-sm"
              >
                Logout
              </button>
            </div>
          ) : (
            <span className="hidden sm:inline">Guest User</span>
          )}
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition"
            aria-label="Toggle Theme"
          >
            {darkMode ? <Sun className="w-6 h-6" /> : <Moon className="w-6 h-6" />}
          </button>
        </div>
      </nav>

      <main className="p-4 md:p-6 max-w-7xl mx-auto">
        {!user ? (
          <div className="flex justify-center">
            <Login onLogin={handleLogin} />
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* 📂 Left Sidebar */}
            <div className="lg:col-span-1 space-y-6">
              <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-2xl shadow-lg">
                <h2 className="text-xl font-semibold mb-4">📤 Upload PDFs</h2>
                <UploadPDFs setSessionId={setSessionId} sessionToken={sessionToken} />
              </div>

              <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-2xl shadow-lg">
              <UserSessions
                userId={user}
                sessionToken={sessionToken}
                onLoadSession={(session) => {
                  setSessionId(session.session_id); // ✅ corrected field
                  setLoadedMessages(session.messages || []);
                  localStorage.setItem('sessionId', session.session_id);
                  localStorage.setItem('loadedMessages', JSON.stringify(session.messages || []));
                }}
              />

              </div>
            </div>

            {/* 💬 Main Content */}
            <div className="lg:col-span-3">
              <div className="bg-gray-50 dark:bg-gray-800 rounded-2xl shadow-lg p-4">
                <div className="flex border-b border-gray-200 dark:border-gray-700 mb-4">
                  <button
                    onClick={() => setActiveTab('chat')}
                    className={`px-4 py-2 font-medium ${
                      activeTab === 'chat'
                        ? 'border-b-2 border-blue-600 text-blue-600'
                        : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                    }`}
                    disabled={!sessionId}
                  >
                    💬 Chat
                  </button>
                  <button
                    onClick={() => setActiveTab('analysis')}
                    className={`px-4 py-2 font-medium ${
                      activeTab === 'analysis'
                        ? 'border-b-2 border-blue-600 text-blue-600'
                        : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                    }`}
                  >
                    📊 Analysis
                  </button>
                </div>

                {activeTab === 'chat' ? (
                  <div>
                    <h2 className="text-xl font-semibold mb-4">Ask Questions</h2>
                    <ChatBox
                      key={sessionId || 'new'}
                      sessionId={sessionId}
                      sessionToken={sessionToken}
                      loadedMessages={loadedMessages}
                      onMessagesUpdate={(msgs) => setLoadedMessages(msgs)} // 🔁 Update when chat changes
                    />
                  </div>
                ) : (
                  <div>
                    <h2 className="text-xl font-semibold mb-4">Document Analysis</h2>
                    <DocumentAnalysis sessionId={sessionId} />
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;

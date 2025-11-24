// src/components/ChatBox.js
import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import VoiceInput from "./VoiceInput";
import VoiceOutput from "./VoiceOutput";
import { API_BASE_URL, apiStream } from "../config";

function ChatBox({ sessionId, sessionToken, loadedMessages = [], onMessagesUpdate }) {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState(
    (loadedMessages || []).map(m =>
      ({ ...m, text: m.text ?? m.messagetext ?? m.message_text ?? "" })
    )
  );
  const [loading, setLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Auto-scroll to bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // Sync messages state on loadedMessages prop change
  useEffect(() => {
    setMessages(
      (loadedMessages || []).map(m =>
        ({ ...m, text: m.text ?? m.messagetext ?? m.message_text ?? "" })
      )
    );
  }, [loadedMessages]);

  // Load chat from localStorage or backend when sessionId changes
  useEffect(() => {
    const saved = localStorage.getItem(`chat_${sessionId}`);
    if (saved) {
      const parsed = JSON.parse(saved);
      const normalized = parsed.map(m =>
        ({ ...m, text: m.text ?? m.messagetext ?? m.message_text ?? "" })
      );
      setMessages(normalized);
    } else if (sessionId) {
      fetchChatFromBackend(sessionId);
    }
  }, [sessionId, sessionToken]);

  // Save chat to localStorage on every change
  useEffect(() => {
    if (sessionId && messages.length > 0) {
      localStorage.setItem(`chat_${sessionId}`, JSON.stringify(messages));
    }
  }, [messages, sessionId]);

  const fetchChatFromBackend = async (id) => {
    try {
      const res = await fetch(`${API_BASE_URL}/chat-messages/${id}`, {
        headers: { "Authorization": `Bearer ${sessionToken}` },
      });
      const data = await res.json();
      if (data.messages) {
        const normalized = data.messages.map(m =>
          ({ ...m, text: m.text ?? m.messagetext ?? m.message_text ?? "" })
        );
        setMessages(normalized);
      }
    } catch (err) {
      console.error("Error loading chat:", err);
    }
  };

  const handleAsk = async () => {
    if (!sessionId || !question.trim()) {
      alert("Please upload a file or load a session first.");
      return;
    }
  
    const userMsg = { 
      sender: "You", 
      text: question, 
      timestamp: new Date(),
      id: Date.now()
    };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);
    setIsTyping(true);
  
    try {
      const response = await apiStream("/ask", {
        method: "POST",
        body: JSON.stringify({ question, session_id: sessionId }),
        headers: { "Authorization": `Bearer ${sessionToken}` },
      });
  
      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
  
      const aiMsg = { 
        sender: "AI", 
        text: "", 
        citations: [], 
        timestamp: new Date(),
        id: Date.now() + 1
      };
      let aiMsgIndex = -1;
      setMessages(prev => {
        const upd = [...prev, aiMsg];
        aiMsgIndex = upd.length - 1;
        return upd;
      });
  
      let buffer = "";
      let accumulatedText = "";
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        
        if (!buffer.includes("---__CITATIONS__---")) {
          accumulatedText = buffer;
          setMessages(current => {
            const upd = [...current];
            const idx = upd.findIndex(m => m.id === aiMsg.id);
            if (idx !== -1) {
              upd[idx] = { ...upd[idx], text: accumulatedText };
            }
            return upd;
          });
        }
      }
  
      let [text, cites] = buffer.split("---__CITATIONS__---");
      let citations = [];
      if (cites) {
        try { citations = JSON.parse(cites); } catch { citations = []; }
      }
  
      setMessages(current => {
        const upd = [...current];
        const idx = upd.findIndex(m => m.id === aiMsg.id);
        if (idx !== -1) {
          upd[idx] = {
            ...upd[idx],
            text: text || "",
            citations: citations,
            timestamp: new Date(),
          };
        }
        if (onMessagesUpdate) onMessagesUpdate(upd);
        return upd;
      });
  
    } catch (err) {
      console.error("Error asking question:", err);
      setMessages(prev => [
        ...prev,
        { 
          sender: "AI", 
          text: `Error: ${err.message}`, 
          timestamp: new Date(),
          id: Date.now()
        },
      ]);
    } finally {
      setLoading(false);
      setIsTyping(false);
      setQuestion("");
    }
  };

  const clearChat = () => {
    if (window.confirm("Are you sure you want to clear this chat?")) {
      setMessages([]);
      localStorage.removeItem(`chat_${sessionId}`);
      if (onMessagesUpdate) onMessagesUpdate([]);
    }
  };

  const handleSaveChat = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/save-chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${sessionToken}`,
        },
        body: JSON.stringify({
          session_id: sessionId,
          messages: messages.map((m) => ({
            sender: m.sender,
            text: m.text,
            citations: m.citations || [],
          })),
        }),
      });

      if (!res.ok) throw new Error("Failed to save chat");
      
      const saveBtn = document.querySelector('.save-btn');
      const originalText = saveBtn.textContent;
      saveBtn.textContent = "✓ Saved!";
      saveBtn.classList.add('bg-green-600');
      setTimeout(() => {
        saveBtn.textContent = originalText;
        saveBtn.classList.remove('bg-green-600');
      }, 2000);
      
    } catch (err) {
      alert(`Error saving chat: ${err.message}`);
    }
  };

  const handleDownloadTable = async (citation) => {
    const res = await fetch(`${API_BASE_URL}/download-table`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(citation),
    });
    if (!res.ok) {
      alert("Failed to download table.");
      return;
    }
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = citation.source?.replace(/\.[^/.]+$/, "") + ".csv" || "table.csv";
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const ordered = [...messages].sort((a, b) =>
    new Date(a.timestamp || a.createdat) - new Date(b.timestamp || b.createdat)
  );

  const paired = [];
  for (let i = 0; i < ordered.length; i++) {
    const msg = ordered[i];
    if (msg.sender === "You" || msg.sender === "user") {
      paired.push(msg);
      if (
        i + 1 < ordered.length &&
        (ordered[i + 1].sender === "AI" || ordered[i + 1].sender === "ai")
      ) {
        paired.push(ordered[i + 1]);
        i++;
      }
    }
  }

  const CitationCards = ({ citations }) => (
    <div className="mt-4 border-t border-gray-200 dark:border-gray-600 pt-3">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
        <div className="font-semibold text-sm text-blue-700 dark:text-blue-300">Sources & References</div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {citations.map((cite, idx) => (
          <div
            key={idx}
            className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-800 border border-gray-200 dark:border-gray-600 rounded-xl shadow-sm p-4 hover:shadow-md transition-shadow duration-200"
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex-1">
                <div className="text-sm font-medium text-blue-900 dark:text-blue-200 truncate">
                  {cite.source}
                </div>
                {cite.location && (
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    📍 {cite.location}
                  </div>
                )}
                {cite.type && (
                  <div className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 mt-1">
                    {cite.type}
                  </div>
                )}
              </div>
            </div>
            
            <div className="text-sm text-gray-700 dark:text-gray-300 line-clamp-3 leading-relaxed">
              {cite.preview}
            </div>
            
            {cite.type === "table" && Array.isArray(cite.table_data) && cite.table_data.length > 0 && (
              <button
                className="mt-3 px-3 py-1.5 text-xs bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg hover:from-green-600 hover:to-green-700 transition-all duration-200 flex items-center gap-1"
                onClick={() => handleDownloadTable(cite)}
              >
                <span>📥</span>
                Download Table
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  const MessageBubble = ({ message }) => (
    <div
      className={`flex gap-3 mb-6 ${
        message.sender === "You" ? "flex-row-reverse" : "flex-row"
      }`}
    >
      {/* Avatar */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold ${
        message.sender === "You" 
          ? "bg-gradient-to-br from-blue-500 to-blue-600" 
          : "bg-gradient-to-br from-purple-500 to-purple-600"
      }`}>
        {message.sender === "You" ? "👤" : "🤖"}
      </div>

      {/* Message Content */}
      <div className={`flex flex-col max-w-[85%] ${
        message.sender === "You" ? "items-end" : "items-start"
      }`}>
        {/* Header with sender and time */}
        <div className={`flex items-center gap-2 mb-1 ${
          message.sender === "You" ? "flex-row-reverse" : "flex-row"
        }`}>
          <span className="font-semibold text-gray-900 dark:text-white">
            {message.sender}
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {formatTime(message.timestamp)}
          </span>
        </div>

        {/* Message Bubble */}
        <div className={`rounded-2xl px-4 py-3 shadow-sm ${
          message.sender === "You"
            ? "bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-br-md"
            : "bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white rounded-bl-md"
        }`}>
          {/* FIXED: Removed className prop from ReactMarkdown */}
          <ReactMarkdown 
            remarkPlugins={[remarkGfm]} 
            rehypePlugins={[rehypeRaw]}
            components={{
              // Custom components for styling
              code: ({ node, inline, className, children, ...props }) => {
                const isUserMessage = message.sender === "You";
                return (
                  <code
                    className={`${
                      inline
                        ? "px-1.5 py-0.5 rounded text-sm"
                        : "block p-3 rounded-lg my-2 text-sm overflow-x-auto"
                    } ${
                      isUserMessage
                        ? "bg-blue-400/20 text-blue-100"
                        : "bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200"
                    }`}
                    {...props}
                  >
                    {children}
                  </code>
                );
              },
              // Style other elements
              p: ({ node, children, ...props }) => (
                <p className="mb-2 last:mb-0 leading-relaxed" {...props}>
                  {children}
                </p>
              ),
              ul: ({ node, children, ...props }) => (
                <ul className="list-disc list-inside mb-2 space-y-1" {...props}>
                  {children}
                </ul>
              ),
              ol: ({ node, children, ...props }) => (
                <ol className="list-decimal list-inside mb-2 space-y-1" {...props}>
                  {children}
                </ol>
              ),
              li: ({ node, children, ...props }) => (
                <li className="pl-1" {...props}>
                  {children}
                </li>
              ),
              blockquote: ({ node, children, ...props }) => (
                <blockquote className="border-l-4 border-gray-300 dark:border-gray-600 pl-4 italic my-2" {...props}>
                  {children}
                </blockquote>
              ),
              table: ({ node, children, ...props }) => (
                <div className="overflow-x-auto my-2">
                  <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-600">
                    {children}
                  </table>
                </div>
              ),
              th: ({ node, children, ...props }) => (
                <th className="border border-gray-300 dark:border-gray-600 px-3 py-2 bg-gray-100 dark:bg-gray-700 font-semibold" {...props}>
                  {children}
                </th>
              ),
              td: ({ node, children, ...props }) => (
                <td className="border border-gray-300 dark:border-gray-600 px-3 py-2" {...props}>
                  {children}
                </td>
              ),
            }}
          >
            {message.text}
          </ReactMarkdown>

          {/* Voice output for AI messages */}
          {message.sender === "AI" && message.text && (
            <div className="mt-3 pt-3 border-t border-white/20">
              <VoiceOutput text={message.text} />
            </div>
          )}
        </div>

        {/* Citations for AI messages */}
        {message.sender === "AI" && Array.isArray(message.citations) && message.citations.length > 0 && (
          <div className="w-full mt-3">
            <CitationCards citations={message.citations} />
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="flex flex-col h-[700px] bg-white dark:bg-gray-900 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="flex justify-between items-center p-6 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900">
        <div>
          <h3 className="text-xl font-bold text-gray-900 dark:text-white">Chat with your Files</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            {sessionId ? "Ask questions about your uploaded documents" : "Upload files to start chatting"}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={clearChat}
            disabled={messages.length === 0}
            className="px-4 py-2 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-xl text-sm font-medium hover:from-gray-600 hover:to-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2"
          >
            <span>🗑️</span>
            Clear
          </button>
          <button
            onClick={handleSaveChat}
            className="save-btn px-4 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl text-sm font-medium hover:from-blue-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2"
            disabled={!sessionId || messages.length === 0}
          >
            <span>💾</span>
            Save Chat
          </button>
        </div>
      </div>

      {/* Chat Window */}
      <div 
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-6 bg-gray-50/50 dark:bg-gray-800/50"
      >
        {paired.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-blue-200 dark:from-blue-900 dark:to-blue-800 rounded-full flex items-center justify-center mb-4">
              <span className="text-2xl">💬</span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              {sessionId ? "Start a conversation!" : "Welcome!"}
            </h3>
            <p className="text-gray-600 dark:text-gray-400 max-w-md">
              {sessionId 
                ? "Ask questions about your uploaded documents and get AI-powered answers with citations."
                : "Please upload files or load a session to begin chatting."
              }
            </p>
          </div>
        ) : (
          <>
            {paired.map((message, index) => (
              <MessageBubble key={message.id || index} message={message} />
            ))}
            
            {/* Typing indicator */}
            {isTyping && (
              <div className="flex gap-3 mb-6">
                <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-gradient-to-br from-purple-500 to-purple-600 text-white text-sm font-bold">
                  🤖
                </div>
                <div className="flex flex-col max-w-[85%]">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold text-gray-900 dark:text-white">AI</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">Now</span>
                  </div>
                  <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-6 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
        <div className="flex items-center gap-3">
          <VoiceInput setQuestion={setQuestion} />
          <div className="flex-1 relative">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleAsk()}
              placeholder={sessionId ? "Ask a question about your documents..." : "Upload files to enable chat..."}
              className="w-full p-4 pr-12 border border-gray-300 dark:border-gray-600 rounded-xl text-gray-900 dark:text-white bg-gray-50 dark:bg-gray-800 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
              disabled={!sessionId || loading}
            />
            {question && (
              <button
                onClick={() => setQuestion("")}
                className="absolute right-14 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                ✕
              </button>
            )}
          </div>
          <button
            onClick={handleAsk}
            disabled={loading || !sessionId || !question.trim()}
            className="px-6 py-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl font-medium hover:from-blue-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2 min-w-[100px] justify-center"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Asking...</span>
              </>
            ) : (
              <>
                <span>📤</span>
                <span>Ask</span>
              </>
            )}
          </button>
        </div>
        
        {/* Quick Tips */}
        {sessionId && paired.length === 0 && (
          <div className="mt-3 text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400">
              💡 Try asking: "What are the main topics in my documents?" or "Summarize the key points"
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default ChatBox;
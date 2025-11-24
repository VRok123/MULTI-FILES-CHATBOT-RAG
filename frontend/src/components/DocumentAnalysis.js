// src/components/DocumentAnalysis.js
import React, { useState, useEffect, useCallback } from 'react';
import { apiFetch } from '../config';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend,
  PieChart, Pie, Cell, LineChart, Line, AreaChart, Area
} from 'recharts';

function DocumentAnalysis({ sessionId }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('overview');

  const fetchAnalysis = useCallback(async () => {
    if (!sessionId) {
      setError('No session ID available');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const data = await apiFetch(`/document-analysis/${sessionId}`);
      setAnalysis(data);
    } catch (err) {
      console.error("Error fetching analysis:", err);
      setError(err.message || 'Error loading document analysis');
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    if (sessionId) fetchAnalysis();
  }, [sessionId, fetchAnalysis]);

  // Enhanced metrics calculations
  const calculateEnhancedMetrics = (analysisData) => {
    if (!analysisData || !analysisData.sources) return {};
    
    const sources = analysisData.sources;
    const fileTypes = {};
    const contentTypes = {};
    let totalContentLength = 0;
    const chunkSizes = [];
    const fileChunkDistribution = [];

    Object.entries(sources).forEach(([source, chunks]) => {
      // File type analysis
      const fileExt = source.split('.').pop()?.toLowerCase() || 'unknown';
      fileTypes[fileExt] = (fileTypes[fileExt] || 0) + chunks.length;
      
      // Content type analysis and statistics
      chunks.forEach(chunk => {
        const contentType = chunk.type || 'text';
        contentTypes[contentType] = (contentTypes[contentType] || 0) + 1;
        
        const contentLength = chunk.content_preview?.length || 0;
        totalContentLength += contentLength;
        chunkSizes.push(contentLength);
      });

      fileChunkDistribution.push({
        name: source.length > 20 ? source.substring(0, 20) + '...' : source,
        chunks: chunks.length,
        fullName: source
      });
    });

    // Calculate averages
    const avgChunkSize = chunkSizes.length > 0 ? 
      Math.round(chunkSizes.reduce((a, b) => a + b, 0) / chunkSizes.length) : 0;
    
    const maxChunkSize = Math.max(...chunkSizes, 0);
    const minChunkSize = Math.min(...chunkSizes, 0);

    return {
      fileTypes,
      contentTypes,
      totalContentLength,
      avgChunkSize,
      maxChunkSize,
      minChunkSize,
      fileChunkDistribution: fileChunkDistribution.sort((a, b) => b.chunks - a.chunks).slice(0, 10),
      chunkSizes
    };
  };

  const metrics = analysis ? calculateEnhancedMetrics(analysis) : {};
  
  // Data for charts
  const metricsData = analysis
    ? [
        { name: 'Text Chunks', value: analysis.document_count || 0 },
        { name: 'Source Files', value: analysis.source_count || 0 },
      ]
    : [];

  // File type distribution data
  const fileTypeData = Object.entries(metrics.fileTypes || {}).map(([name, value]) => ({
    name: name.toUpperCase(),
    value,
    count: value
  }));

  // Content type distribution data
  const contentTypeData = Object.entries(metrics.contentTypes || {}).map(([name, value]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    value,
    count: value
  }));

  // Colors for charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];
  const CONTENT_COLORS = {
    text: '#0088FE',
    code: '#00C49F',
    table: '#FFBB28',
    ocr: '#FF8042',
    image: '#8884D8'
  };

  if (!sessionId) {
    return (
      <div className="p-4 border rounded-2xl shadow-md bg-gray-50 dark:bg-gray-800">
        <h3 className="text-lg font-bold mb-2">Document Analysis</h3>
        <p className="text-gray-500 dark:text-gray-400">
          Upload PDFs or load a session to see document analysis.
        </p>
      </div>
    );
  }

  const renderOverviewTab = () => (
    <div className="space-y-4">
      {/* Enhanced Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <div className="bg-blue-100 dark:bg-blue-900 p-3 rounded-lg text-center">
          <div className="text-2xl font-bold">{analysis.document_count || 0}</div>
          <div className="text-sm">Total Chunks</div>
        </div>
        <div className="bg-green-100 dark:bg-green-900 p-3 rounded-lg text-center">
          <div className="text-2xl font-bold">{analysis.source_count || 0}</div>
          <div className="text-sm">Source Files</div>
        </div>
        <div className="bg-purple-100 dark:bg-purple-900 p-3 rounded-lg text-center">
          <div className="text-2xl font-bold">{metrics.avgChunkSize || 0}</div>
          <div className="text-sm">Avg Chunk Size</div>
        </div>
        <div className="bg-orange-100 dark:bg-orange-900 p-3 rounded-lg text-center">
          <div className="text-2xl font-bold">
            {metrics.totalContentLength ? Math.round(metrics.totalContentLength / 1024) : 0}KB
          </div>
          <div className="text-sm">Total Content</div>
        </div>
      </div>

      {/* Main Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Document Metrics Bar Chart */}
        <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow">
          <h4 className="font-semibold mb-2 text-gray-800 dark:text-gray-200">Session Overview</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={metricsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="value" fill="#2563eb" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Content Type Distribution */}
        <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow">
          <h4 className="font-semibold mb-2 text-gray-800 dark:text-gray-200">Content Types</h4>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={contentTypeData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {contentTypeData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={CONTENT_COLORS[entry.name.toLowerCase()] || COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => [`${value} chunks`, 'Count']} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  const renderFilesTab = () => (
    <div className="space-y-4">
      {/* File Type Distribution */}
      {fileTypeData.length > 0 && (
        <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow">
          <h4 className="font-semibold mb-2 text-gray-800 dark:text-gray-200">File Type Distribution</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={fileTypeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis allowDecimals={false} />
              <Tooltip formatter={(value) => [`${value} chunks`, 'Count']} />
              <Bar dataKey="value" fill="#8884d8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Top Files by Chunk Count */}
      {metrics.fileChunkDistribution && metrics.fileChunkDistribution.length > 0 && (
        <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow">
          <h4 className="font-semibold mb-2 text-gray-800 dark:text-gray-200">Top Files by Chunk Count</h4>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart 
              data={metrics.fileChunkDistribution} 
              layout="vertical"
              margin={{ left: 100 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis 
                type="category" 
                dataKey="name" 
                width={90}
                tick={{ fontSize: 12 }}
              />
              <Tooltip 
                formatter={(value) => [`${value} chunks`, 'Count']}
                labelFormatter={(value, payload) => payload[0]?.payload.fullName || value}
              />
              <Bar dataKey="chunks" fill="#00C49F" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );

  const renderDocumentsTab = () => (
    <div className="space-y-4">
      {analysis.sources && Object.keys(analysis.sources).length > 0 ? (
        <div className="max-h-96 overflow-y-auto">
          <div className="space-y-3">
            {Object.entries(analysis.sources).map(([source, pages]) => {
              const fileExt = source.split('.').pop()?.toLowerCase();
              const chunkTypes = {};
              pages.forEach(page => {
                const type = page.type || 'text';
                chunkTypes[type] = (chunkTypes[type] || 0) + 1;
              });

              return (
                <div key={source} className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
                  <div className="flex justify-between items-start mb-2">
                    <h5 className="font-medium text-gray-800 dark:text-gray-200 truncate flex-1">
                      {source}
                    </h5>
                    <span className="bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-2 py-1 rounded text-xs ml-2">
                      .{fileExt}
                    </span>
                  </div>
                  
                  <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
                    <span>{pages.length} chunk{pages.length !== 1 ? 's' : ''}</span>
                    <div className="flex space-x-2">
                      {Object.entries(chunkTypes).map(([type, count]) => (
                        <span 
                          key={type}
                          className="px-2 py-1 rounded text-xs"
                          style={{ 
                            backgroundColor: `${CONTENT_COLORS[type]}20`,
                            color: CONTENT_COLORS[type]
                          }}
                        >
                          {type}: {count}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    <div className="font-medium mb-1">Sample content:</div>
                    <div className="italic bg-white dark:bg-gray-800 p-2 rounded border">
                      {pages[0]?.content_preview || 'No content available'}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <p className="text-gray-500 dark:text-gray-400 text-center py-8">
          No documents uploaded in this session.
        </p>
      )}
    </div>
  );

  return (
    <div className="p-4 border rounded-2xl shadow-md bg-gray-50 dark:bg-gray-800">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-lg font-bold">Document Analysis</h3>
        <button 
          onClick={fetchAnalysis}
          disabled={loading || !sessionId}
          className="px-3 py-1 bg-blue-600 text-white rounded text-sm disabled:opacity-50 hover:bg-blue-700 transition-colors"
        >
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 p-2 rounded mb-3">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex justify-center items-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      ) : analysis ? (
        <div>
          {/* Tab Navigation */}
          <div className="flex border-b border-gray-200 dark:border-gray-700 mb-4">
            {['overview', 'files', 'documents'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 font-medium text-sm capitalize transition-colors ${
                  activeTab === tab
                    ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
                    : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          {activeTab === 'overview' && renderOverviewTab()}
          {activeTab === 'files' && renderFilesTab()}
          {activeTab === 'documents' && renderDocumentsTab()}
        </div>
      ) : (
        <p className="text-gray-500 dark:text-gray-400">No analysis available. Click refresh to load.</p>
      )}
    </div>
  );
}

export default DocumentAnalysis;
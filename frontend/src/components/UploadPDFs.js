import React, { useState } from "react";
import { API_BASE_URL } from "../config";

function UploadPDFs({ setSessionId, sessionToken, currentSessionId }) {  // ← NEW: Add currentSessionId prop
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files)); // Convert FileList to array
  };

  const handleUpload = async () => {
    if (!files.length) {
      alert("Please select files first!");
      return;
    }
    if (!sessionToken) {
      alert("You must be logged in to upload files.");
      return;
    }
  
    setUploading(true);
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("session_token", sessionToken);
    
    // NEW: Include current session ID if it exists
    if (currentSessionId) {
      formData.append("session_id", currentSessionId);
    }
  
    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });
  
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Upload failed");
  
      setSessionId(data.session_id);
      
      // NEW: Show appropriate message based on action
      if (data.action === "added_to_existing") {
        alert(`✅ Files added to existing session!\n${data.message}`);
      } else {
        alert(data.message);
      }
      
    } catch (err) {
      console.error("Upload error:", err);
      alert(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const clearSelection = () => {
    setFiles([]);
    // Clear the file input
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) fileInput.value = '';
  };

  return (
    <div className="p-4 border rounded-lg bg-gray-50 dark:bg-gray-700">
      <h3 className="text-lg font-semibold mb-2">Upload Files</h3>
      
      {/* NEW: Session info display */}
      {currentSessionId && (
        <div className="mb-3 p-2 bg-blue-100 dark:bg-blue-900 rounded text-sm">
          <span className="font-medium">Active Session:</span> 
          <br />
          <span className="text-xs opacity-75">
            New files will be added to existing session
          </span>
        </div>
      )}
      
      <input
        type="file"
        accept=".pdf,.docx,.pptx,.ppt,.jpg,.jpeg,.png,.bmp,.tiff,.csv,.xlsx,.xls,.html,.htm,.txt,.md"
        multiple
        onChange={handleFileChange}
        className="mb-3 w-full"
      />
      
      {files.length > 0 && (
        <div className="mb-3 p-3 bg-white dark:bg-gray-600 rounded">
          <div className="flex justify-between items-center mb-2">
            <strong>Selected files:</strong>
            <button
              onClick={clearSelection}
              className="text-sm text-red-600 hover:text-red-800"
            >
              Clear
            </button>
          </div>
          <ul className="list-disc list-inside max-h-32 overflow-y-auto">
            {files.map((file, idx) => (
              <li key={idx} className="text-sm truncate">{file.name}</li>
            ))}
          </ul>
        </div>
      )}
      
      <button
        onClick={handleUpload}
        disabled={uploading || !files.length}
        className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50 hover:bg-blue-700 transition-colors w-full"
      >
        {uploading ? "Uploading..." : 
         currentSessionId ? "Add to Session" : "Upload Files"}
      </button>
      
      {/* NEW: Help text */}
      <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
        {currentSessionId 
          ? "Files will be added to your current session" 
          : "This will create a new session"}
      </p>
    </div>
  );
}

export default UploadPDFs;
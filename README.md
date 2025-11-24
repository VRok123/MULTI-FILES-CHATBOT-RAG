# 💬 Multi-File Chatbot with GPT-OSS/20b RAG Architecture

This project is an advanced **Retrieval-Augmented Generation (RAG)** application designed to process and chat with data from **multiple file types** (PDFs, DOCX, TXT, etc.).

It utilizes the **Mistral** model family (or a Mistral-based model) via a flexible API (like OpenRouter or a self-hosted endpoint) to provide context-aware, source-attributed answers.

---

## ✨ Features

* **📄 Multi-File Ingestion:** Process and chat with data from multiple document types (e.g., PDF, DOCX, TXT).
* **🧠 GPT-OSS/20b-Powered RAG:** Uses a powerful GPT-OSS/20b-based model for high-quality, relevant response generation.
* **💬 Session-aware Chat:** Maintains conversational context throughout the user session.
* **⚡ Streaming Responses:** Provides fast, real-time response generation for a better user experience.
* **📑 Source Attribution:** Links answers directly back to the specific file or section they originated from.
* **🎨 Modern Stack:** Built with a React Frontend and a FastAPI Backend.

## 💻 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend** | Python, FastAPI, LangChain | Handles document processing, embedding, RAG pipeline, and API endpoints. |
| **LLM/API** | GPT-OSS/20b (via OpenRouter/Self-Hosted) | The core generative language model. |
| **Frontend** | React.js, Tailwind CSS | Responsive and interactive user interface. |
| **Vector DB** | FAISS / Other | Used for local storage of document embeddings. |
| **Session** | SQLite/MySQL (Future) | For managing and persisting session and chat history. |

## 🚀 Getting Started

Follow these steps to get the application running locally.

### 1. Set up Environment Variables

Before running, you must set up your API key. Create a file named **`.env`** in your `backend` directory and add your key:

```.env
# Replace with your actual key and preferred API/Model endpoint
OPENROUTER_API_KEY="YOUR_API_KEY_HERE"
# Or any other API key relevant to your specific Mistral setup

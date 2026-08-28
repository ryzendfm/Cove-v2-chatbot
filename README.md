# Cove RAG ⚡

A modern, full-stack AI Assistant & RAG platform built with **Django**, **LangChain**, and **Groq**. Cove features per-user isolated Retrieval-Augmented Generation (RAG), live tool execution, real-time reasoning/thought process streaming, session-based conversation memory, rich markdown formatting, and a sleek dark-themed UI.

---

## 📸 Previews

<div align="center">

| New Chat Dashboard | RAG Q&A & Thought Process |
| :---: | :---: |
| <img src="screenshots/Screenshot%202026-08-28%20225021.png" width="420" alt="Dashboard Preview" /> | <img src="screenshots/Screenshot%202026-08-28%20224929.png" width="420" alt="RAG and Thought Process Preview" /> |

| Knowledge Base (Multi-Source RAG) |
| :---: |
| <img src="screenshots/Screenshot%202026-08-28%20223601.png" width="550" alt="Knowledge Base Modal" /> |

| Sign In | Sign Up |
| :---: | :---: |
| <img src="screenshots/Screenshot%202026-08-28%20223229.png" width="420" alt="Sign In Preview" /> | <img src="screenshots/Screenshot%202026-08-28%20223238.png" width="420" alt="Sign Up Preview" /> |

</div>

---

## ✨ Key Features

### 🧠 LangChain Orchestration & Deep Reasoning
- **LangChain Chains**: Flexible, modular prompt engineering, history injection, and tool-augmented generation.
- **Transparent Thought Process**: Streams the model's internal reasoning tokens in real-time within collapsible `<thinking>` accordion blocks.
- **Built-in Tool Calling**: Seamlessly runs tools during conversation (e.g., real-time datetime lookup, mathematical expression evaluation).
- **Automated Thread Title Generation**: Generates concise, context-aware thread titles dynamically using LLM chains.

### 📚 Isolated Multi-Format RAG (Knowledge Base)
- **Multi-Format Ingestion**: Upload and index **PDFs**, **TXT / Markdown**, **CSV spreadsheets**, and direct **Web URLs**.
- **User-Isolated FAISS Vector Store**: Each user has an isolated vector store partition ensuring complete data privacy and fast similarity searches.
- **HuggingFace Embeddings**: Local high-performance embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- **Accurate Source Attribution**: RAG answers include document citations with specific page numbers and row references.
- **One-Click RAG Toggle**: Seamlessly toggle RAG grounding on or off directly from the chat input bar.

### ⚡ Real-Time Streaming & Interaction
- **Server-Sent Events (SSE)**: Ultra-responsive token-by-token streaming with live status updates (*"Searching your documents..."*, *"Running tool..."*).
- **Rich Markdown & Code Rendering**: Full support for markdown tables, syntax-highlighted code blocks, lists, blockquotes, and one-click code copy.
- **Session Memory Management**: Timeline-categorized conversation history (*Today, Yesterday, This Week, Older*) with pinned threads.
- **User Authentication**: Secure multi-user registration, login, and session management.

---

## 🛠️ Tech Stack

- **Backend Framework**: Django 5.x (Python)
- **AI & LLM Engine**: LangChain, Groq API (`openai/gpt-oss-120b` / configurable)
- **Embeddings & Vector Search**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`), FAISS Vector Store
- **Document Processing**: `pypdf`, `BeautifulSoup4`, LangChain Document Loaders & Splitters
- **Frontend**: Vanilla JavaScript, Server-Sent Events (SSE), Marked.js, Highlight.js, CSS3 Dark Mode Design System
- **Database**: SQLite3 (Django ORM)

---

## 📁 Project Structure

```text
Cove_RAG/
├── ChatBot/                  # Django project configuration
│   ├── settings.py           # Settings, Groq API config, & media paths
│   ├── urls.py               # Root URL configuration
│   └── wsgi.py
├── chat/                     # Main Chat & RAG application
│   ├── chains.py             # LangChain streaming orchestration & reasoning flow
│   ├── llm.py                # LLM initialization & prompt templates
│   ├── rag.py                # Document loaders, FAISS vector store & indexing
│   ├── tools.py              # LangChain tool bindings & execution
│   ├── memory.py             # Chat history & message serialization
│   ├── models.py             # ChatSession, ChatMessage, UserDocument models
│   ├── views.py              # SSE stream views, upload & auth endpoints
│   ├── urls.py               # Application routing
│   └── templates/            # UI templates (Chat interface, Sign in, Sign up)
├── media/                    # Uploaded user documents & isolated FAISS indices
├── screenshots/              # UI preview screenshots
├── manage.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Cove_RAG
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install django langchain langchain-community langchain-groq langchain-huggingface sentence-transformers faiss-cpu pypdf beautifulsoup4 requests
```

### 4. Configure API Keys
Set your Groq API Key in your environment or in `ChatBot/settings.py`:
```python
# ChatBot/settings.py
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL   = "openai/gpt-oss-120b"
```

### 5. Apply Database Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 6. Run the Development Server
```bash
python manage.py runserver
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser, create an account, upload documents to your Knowledge Base, and start chatting with Cove!

---

## 💡 How It Works

1. **Document Indexing**: Upload documents or provide a URL in the Knowledge Base modal. The backend chunks the content using `RecursiveCharacterTextSplitter`, computes dense vector embeddings, and stores them in the user's dedicated FAISS index.
2. **Context-Aware Retrieval**: When `RAG ON` is active, user queries query the isolated FAISS vector database to retrieve the top $k$ most relevant document chunks.
3. **Reasoning & Tool Execution**: LangChain binds tools and constructs the conversational prompt with retrieved context. If tool calls or reasoning tokens are generated, they stream in real-time through SSE.
4. **Interactive Response**: Responses render markdown, tables, citations, and expandable thought processes dynamically.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

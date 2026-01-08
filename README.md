# 🤖 Desolve-mini – AI-Powered GitHub Issue Solver

An AI assistant that analyzes GitHub repositories and helps solve issues using **Gemini** or **Groq** AI models with semantic search.

---

## 🚀 Features

- 🔗 Clone & analyze GitHub repositories
- 📂 Extract and index repository files
- 🧠 Generate AI summaries of repositories
- 💬 Chat with AI about issues and code
- ⚡ Dual AI model support (Gemini & Groq)
- 🎨 Modern React UI with Tailwind CSS

---

## 🛠 Tech Stack

**Backend:** FastAPI, Python, FAISS, Sentence Transformers  
**Frontend:** React 19, Tailwind CSS, Vite  
**AI Models:** Google Gemini, Groq LLaMA

---

## ⚙️ Setup

### 1. Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with API keys
echo "GEMINI_API_KEY=your_key_here" > .env
echo "GROQ_API_KEY=your_key_here" >> .env
echo "GITHUB_TOKEN=your_token_here" >> .env
```

### 2. Frontend Setup

```bash
cd react-frontend
npm install
```

---

## 🚀 Run Application

**Terminal 1 – Backend:**
```bash
python main.py
# Backend runs on http://127.0.0.1:8000
```

**Terminal 2 – Frontend:**
```bash
cd react-frontend
npm run dev
# Frontend runs on http://localhost:5173
```

---

## 📖 How It Works

1. **Enter GitHub URL** → Select AI model (Gemini or Groq)
2. **Fetch & Analyze** → Repository is cloned, files extracted, issues fetched
3. **View Summary** → AI generates repository overview
4. **Chat** → Ask questions, AI retrieves relevant code and answers

---

## 📁 Project Structure

```
Desolve-mini/
├── main.py                    # FastAPI backend
├── requirements.txt           # Python dependencies
├── .env                       # API keys (create this)
│
├── models/
│   ├── gemini_models_rag.py   # Gemini AI pipeline
│   └── groq_models_rag.py     # Groq AI pipeline
│
├── services/
│   ├── clone.py               # Clone repositories
│   ├── file_contents.py       # Extract files
│   ├── issues.py              # Fetch issues
│   └── repo_summarizer.py     # Generate summaries
│
└── react-frontend/            # React app
    ├── src/
    │   ├── components/        # UI components
    │   ├── pages/            # Page views
    │   └── App.jsx           # Main app
    └── package.json
```

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/process-repo` | Analyze repository |
| POST | `/ask-ai` | Chat with AI about code |
| GET | `/models` | Get available models |
| POST | `/cleanup` | Clear temp data |

---

## 📋 Requirements

- Python 3.8+
- Node.js 16+
- Gemini API Key ([get free](https://ai.google.dev))
- Groq API Key ([get free](https://groq.com))
- GitHub Token (optional, for higher API limits)

---

## 💡 Usage Example

1. Go to `http://localhost:5173`
2. Paste a GitHub URL: `https://github.com/username/repo`
3. Select an AI model
4. Click "Fetch Issues & Analyze"
5. Select an issue and start chatting!

---

## 📝 License

Open source for educational and commercial use.

---

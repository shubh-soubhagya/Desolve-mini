import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_NAME = "openai/gpt-oss-120b"
FILES_CSV = r"data/repo_files_data.csv"
ISSUES_CSV = r"data/repo_issues.csv"
CUSTOM_MODEL_PATH = r"sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = r"embeddings/repo_index.pkl"

TOP_K = 30  # Number of most relevant files to retrieve per issue

# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================
def load_env_and_configure():
    """Loads .env file and configures the GROQ API key (Groq client)."""
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in .env file.")
        raise ValueError("GROQ_API_KEY not found in .env file.")

    client = Groq(api_key=GROQ_API_KEY)
    print("✅ GROQ client configured.")
    return client


# =====================================================
# BUILD VECTOR INDEX (RUN ONCE PER REPO)
# =====================================================
def build_vector_index():
    """Builds and saves a FAISS vector index from the repo files CSV."""
    print("🧠 Building vector index from repo files...")

    # Ensure directories exist
    files_dir = os.path.dirname(FILES_CSV) or "."
    index_dir = os.path.dirname(INDEX_PATH) or "."
    os.makedirs(files_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    if not os.path.exists(FILES_CSV):
        print(f"❌ Files CSV not found at {FILES_CSV}. Cannot build index.")
        raise FileNotFoundError(f"Required file not found: {FILES_CSV}")

    df = pd.read_csv(FILES_CSV, encoding="utf-8")
    if df.empty:
        print("❌ No files found in repo_files_data.csv")
        raise ValueError("No files found in repo_files_data.csv to build index.")

    print(f"📄 Total files: {len(df)}")

    model = SentenceTransformer(CUSTOM_MODEL_PATH)
    # ensure all file_content are strings
    file_texts = df["file_content"].fillna("").astype(str).tolist()
    embeddings = model.encode(file_texts, convert_to_numpy=True, show_progress_bar=True)

    # create faiss index (L2). If you want cosine later, normalize and use IndexFlatIP.
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    # save index + metadata using pickle (matching your gemini script style)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump((index, df), f)

    print(f"✅ Vector index saved at: {INDEX_PATH}")


# =====================================================
# LOAD INDEX & RETRIEVE RELEVANT FILES
# =====================================================
def retrieve_relevant_files(query: str, top_k: int = TOP_K):
    """Retrieves context from the vector index based on a query."""
    if not os.path.exists(INDEX_PATH):
        print("⚠️ Index not found. Building one now...")
        build_vector_index()

    with open(INDEX_PATH, "rb") as f:
        index, df = pickle.load(f)

    model = SentenceTransformer(CUSTOM_MODEL_PATH)
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")

    # search; if top_k > num entries, faiss returns padded indices — handle that
    D, I = index.search(query_vec, top_k)
    indices = [i for i in I[0] if i != -1]
    top_files = df.iloc[indices].to_dict(orient="records")

    repo_context = ""
    for file in top_files:
        repo_context += (
            f"\n\n### File: {file.get('file_name', '')} ({file.get('file_path', '')})\n"
            f"```{file.get('file_extension', '')}\n"
            f"{file.get('file_content', '')[:2500]}\n```\n"
        )
    return repo_context


# =====================================================
# CREATE PROMPT FOR GROQ / MODEL
# =====================================================
def create_prompt(issue: dict, repo_context: str) -> str:
    """Creates the initial system-level prompt for the AI."""
    issue_title = issue.get("title", "Untitled Issue")
    issue_body = issue.get("body", "No description provided.")
    issue_number = issue.get("number", "N/A")

    return f"""
You are Desolve AI — an expert AI developer assistant that helps fix software repository issues.

### Issue #{issue_number}: {issue_title}
{issue_body}

### Relevant Repository Files
{repo_context}

Your task:
- Analyze the given files and issue.
- Suggest precise code changes or fixes.
- Provide reasoning and corrected code snippets.
- Maintain concise, professional formatting.
"""


# =====================================================
# LOAD ISSUE
# =====================================================
def load_issue(issue_csv: str, row_index: int = 0):
    """Loads a specific issue by its row index in the CSV."""
    if not os.path.exists(issue_csv):
        print(f"❌ Issues CSV not found at {issue_csv}")
        raise FileNotFoundError(f"Issues CSV not found: {issue_csv}")

    df = pd.read_csv(issue_csv)
    if len(df) <= row_index:
        print(f"❌ CSV has only {len(df)} issues. Row {row_index + 1} not found.")
        raise IndexError(f"Row index {row_index} out of bounds for issues CSV.")
    return df.to_dict(orient="records")[row_index]


# =====================================================
# NOTE
# =====================================================
# CLI/chat loop and __main__ were intentionally omitted (like your Gemini script).
# Integrate these functions into your FastAPI main.py (or call them from tests/CLI).
#
# Example usage inside your server code:
# client = load_env_and_configure()
# if not os.path.exists(INDEX_PATH): build_vector_index()
# issue = load_issue(ISSUES_CSV, row_index=3)
# ctx = retrieve_relevant_files(issue["body"])
# prompt = create_prompt(issue, ctx)
#
# Then pass `prompt` to your Groq client or whatever model interface you use.

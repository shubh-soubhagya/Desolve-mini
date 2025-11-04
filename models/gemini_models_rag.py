import os
import pickle
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_NAME = "gemini-2.5-flash-lite"
FILES_CSV = r"C:\Users\hp\Desktop\MinorProj\Desolve-mini\data\repo_files_data.csv"
ISSUES_CSV = r"C:\Users\hp\Desktop\MinorProj\Desolve-mini\data\repo_issues.csv"
CUSTOM_MODEL_PATH = r"C:\Users\hp\Desktop\prashna\models\all-MiniLM-L6-v2"
INDEX_PATH = r"C:\Users\hp\Desktop\MinorProj\Desolve-mini\embeddings\repo_index.pkl"

# FILES_CSV = "repo_files_data.csv"
# ISSUES_CSV = "repo_issues.csv"
# CUSTOM_MODEL_PATH = "all-MiniLM-L6-v2"
TOP_K = 30  # Number of most relevant files to retrieve per issue
ROW_INDEX = 3  # 4th issue (0-based)

# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================
def load_env_and_configure():
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GOOGLE_API_KEY:
        print("❌ GEMINI_API_KEY not found in .env file.")
        exit(1)

    genai.configure(api_key=GOOGLE_API_KEY)
    return GOOGLE_API_KEY


# =====================================================
# BUILD VECTOR INDEX (RUN ONCE)
# =====================================================
def build_vector_index():
    print("🧠 Building vector index from repo files...")

    df = pd.read_csv(FILES_CSV, encoding="utf-8")
    if df.empty:
        print("❌ No files found in repo_files_data.csv")
        exit(1)

    print(f"📄 Total files: {len(df)}")

    model = SentenceTransformer(CUSTOM_MODEL_PATH)
    embeddings = model.encode(df["file_content"].astype(str).tolist(), convert_to_numpy=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump((index, df), f)

    print(f"✅ Vector index saved at: {INDEX_PATH}")


# =====================================================
# LOAD INDEX & RETRIEVE RELEVANT FILES
# =====================================================
def retrieve_relevant_files(query: str, top_k: int = TOP_K):
    if not os.path.exists(INDEX_PATH):
        print("⚠️ Index not found. Building one now...")
        build_vector_index()

    with open(INDEX_PATH, "rb") as f:
        index, df = pickle.load(f)

    model = SentenceTransformer(CUSTOM_MODEL_PATH)
    query_vec = model.encode([query], convert_to_numpy=True)

    D, I = index.search(query_vec, top_k)
    top_files = df.iloc[I[0]].to_dict(orient="records")

    repo_context = ""
    for file in top_files:
        repo_context += (
            f"\n\n### File: {file.get('file_name', '')} ({file.get('file_path', '')})\n"
            f"```{file.get('file_extension', '')}\n"
            f"{file.get('file_content', '')[:2500]}\n```\n"
        )
    return repo_context


# =====================================================
# CREATE PROMPT FOR GEMINI
# =====================================================
def create_prompt(issue: dict, repo_context: str) -> str:
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
    df = pd.read_csv(issue_csv)
    if len(df) <= row_index:
        print(f"❌ CSV has only {len(df)} issues. Row {row_index + 1} not found.")
        exit(1)
    return df.to_dict(orient="records")[row_index]


# =====================================================
# GEMINI CHAT SESSION
# =====================================================
def start_chat(system_prompt: str, model_name: str):
    """Starts CLI-based Gemini chat session."""
    model = genai.GenerativeModel(model_name)
    chat = model.start_chat(history=[{"role": "user", "parts": system_prompt}])

    print("\n🤖 Desolve AI: RAG-based contextual chat initialized!\n")

    while True:
        user_input = input("👨‍💻 You: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting chat. Goodbye!")
            break

        try:
            response = chat.send_message(user_input)
            print("\n🤖 Desolve AI:\n")
            print(response.text)
            print("\n" + "-" * 100 + "\n")
        except Exception as e:
            print(f"⚠️ Error: {e}")


# =====================================================
# MAIN
# =====================================================
# if __name__ == "__main__":
#     load_env_and_configure()

#     if not os.path.exists(INDEX_PATH):
#         build_vector_index()

#     issue = load_issue(ISSUES_CSV, row_index=ROW_INDEX)
#     repo_context = retrieve_relevant_files(issue["body"])
#     system_prompt = create_prompt(issue, repo_context)

#     print(f"\n📂 Loaded Issue #{issue.get('number', 'N/A')} — “{issue.get('title', '')}”")
#     start_chat(system_prompt, MODEL_NAME)

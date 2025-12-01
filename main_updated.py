# main.py
import os
import shutil
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import time

# Local Imports
from clone import clone_repo
from file_contents import extract_files_to_csv
from issues import extract_issues

# Add summarizer import (new)
from repo_summarizer import summarize_repository

# Gemini RAG model helpers (keep names distinct)
from models.gemini_models_rag import (
    load_env_and_configure as load_gemini_env,
    build_vector_index as build_gemini_index,
    create_prompt as create_gemini_prompt,
    retrieve_relevant_files as retrieve_gemini_files,
    MODEL_NAME as GEMINI_MODEL_NAME,
)

# Groq (updated chunked RAG) helpers
from models.groq_model_using_chunks import (
    load_env_and_configure as load_groq_env,
    build_vector_index as build_groq_index,
    create_prompt as create_groq_prompt,
    retrieve_relevant_files as retrieve_groq_files,
    MODEL_NAME as GROQ_MODEL_NAME,
    TokenRateLimiter,
    count_tokens as count_tokens_groq,
)



# CONFIGURATION
TEMP_DATA_DIR = "data"
FILES_CSV = os.path.join(TEMP_DATA_DIR, "repo_files_data.csv")
ISSUES_CSV = os.path.join(TEMP_DATA_DIR, "repo_issues.csv")
INDEX_PATH = os.path.join(TEMP_DATA_DIR, "repo_index.pkl")
CLONE_DIR = "cloned_repo"

# Ensure temp dir exists
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# Configure Gemini and Groq clients (if keys available)
try:
    load_gemini_env()
    print("✅ Gemini API key configured.")
except Exception as e:
    print(f"⚠️ Gemini API key not configured: {e}")

groq_client = None
try:
    groq_client = load_groq_env()
    print("✅ Groq API key configured.")
except Exception as e:
    print(f"⚠️ Groq API key not configured: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
chat_sessions: dict = {}
# single token limiter instance for Groq TPM guarding (optional)
token_limiter = TokenRateLimiter()

# Pydantic request models
class RepoRequest(BaseModel):
    url: str
    model: str = "gemini"  # "gemini" or "groq"

class AiRequest(BaseModel):
    issue_id: int
    prompt: str
    model: str = "gemini"  # "gemini" or "groq"


# Helper: safe remove readonly (Windows)
def remove_readonly(func, path, exc):
    import stat
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        func(path)
    else:
        raise


# Cleanup helper with retries
def cleanup_temp_data():
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            if os.path.exists(CLONE_DIR):
                print(f"🗑️ Cleaning up {CLONE_DIR}...")
                shutil.rmtree(CLONE_DIR, onerror=remove_readonly)

            if os.path.exists(TEMP_DATA_DIR):
                print(f"🗑️ Cleaning up {TEMP_DATA_DIR}...")
                shutil.rmtree(TEMP_DATA_DIR, onerror=remove_readonly)

            os.makedirs(TEMP_DATA_DIR, exist_ok=True)
            print("✅ Cleanup complete")
            return
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"⚠️ Cleanup failed (attempt {retry_count}/{max_retries}), retrying: {e}")
                time.sleep(1)
            else:
                print(f"❌ Cleanup failed after {max_retries} attempts: {e}")
                if not os.path.exists(TEMP_DATA_DIR):
                    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
                return


# Load issue by GitHub issue number (safe coercions)
def load_issue_by_id(csv_path: str, issue_id: int) -> dict | None:
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if 'number' not in df.columns:
        return None

    df = df.fillna("")

    try:
        df['number'] = pd.to_numeric(df['number'], errors='coerce')
    except Exception:
        pass

    issue_df = df[df['number'] == issue_id]
    if issue_df.empty:
        return None

    record = issue_df.to_dict(orient="records")[0]
    for k, v in record.items():
        if v is None:
            record[k] = ""
        elif isinstance(v, float) and pd.isna(v):
            record[k] = ""
        elif not isinstance(v, (str, int, bool, list, dict)):
            record[k] = str(v)
    return record


# API: process repo (clone, extract, index, summarize)
@app.post("/process-repo")
async def process_repo(request: RepoRequest):
    print(f"🚀 Processing repo with {request.model} model: {request.url}")

    if request.model not in ["gemini", "groq"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model must be 'gemini' or 'groq'")

    chat_sessions.clear()
    cleanup_temp_data()

    token = os.getenv("GITHUB_TOKEN")

    try:
        print(f"Cloning {request.url}...")
        clone_repo(request.url, CLONE_DIR)

        repo_name = request.url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_path = os.path.join(CLONE_DIR, repo_name)

        print("📂 Extracting repository files...")
        extract_files_to_csv(repo_path, FILES_CSV)

        print("🐞 Extracting repository issues...")
        if not token:
            print("⚠️ GITHUB_TOKEN not set. May be rate-limited.")
        extract_issues(request.url, output_file=ISSUES_CSV, token=token)

        print("🧠 Building vector indexes...")
        # build gemini index (if available)
        try:
            build_gemini_index()
            print("✅ Gemini index built.")
        except Exception as e_index:
            print(f"⚠️ Warning: failed to build Gemini index: {e_index}")

        # build groq index (chunked)
        try:
            build_groq_index()
            print("✅ Groq chunked index built.")
        except Exception as e_index:
            print(f"⚠️ Warning: failed to build Groq index: {e_index}")

        # NEW: Generate repository summary (robust handling)
        print("📖 Generating repository summary...")
        summary = ""
        try:
            # summarize_repository accepts a 'model' argument in your original implementation
            summary = summarize_repository(model=request.model)
            print("✅ Repository summary generated.")
        except Exception as e_summary:
            print(f"⚠️ Warning: failed to generate repository summary: {e_summary}")
            summary = "Summary generation failed. Please check API keys and repo contents."

        print("✅ Processing complete.")
        if not os.path.exists(ISSUES_CSV):
            raise FileNotFoundError(f"Issues CSV not created at {ISSUES_CSV}")

        df_issues = pd.read_csv(ISSUES_CSV)
        if 'number' not in df_issues.columns:
            raise KeyError("Issues CSV must contain a 'number' column.")

        df_issues = df_issues.fillna("")
        df_issues['id'] = df_issues['number']

        issues_df_subset = df_issues[['id', 'title', 'body']].copy()
        for col in ['id', 'title', 'body']:
            issues_df_subset[col] = issues_df_subset[col].apply(
                lambda v: "" if v is None else (str(v) if not isinstance(v, (str, int, bool, list, dict)) else v)
            )

        issues_list = issues_df_subset.to_dict(orient="records")
        safe_payload = jsonable_encoder({
            "issues": issues_list,
            "summary": summary,
            "model": request.model
        })
        return JSONResponse(content=safe_payload)

    except Exception as e:
        print(f"❌ Error during repo processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {str(e)}")


# API: ask-ai (handles Gemini and Groq)
@app.post("/ask-ai")
async def ask_ai(request: AiRequest):
    print(f"🤖 Chat request (model: {request.model}) for issue {request.issue_id}")

    if request.model not in ["gemini", "groq"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model must be 'gemini' or 'groq'")

    try:
        session_key = f"{request.issue_id}_{request.model}"

        # Create new chat session if missing
        if session_key not in chat_sessions:
            print(f"Creating new chat session for issue {request.issue_id} ({request.model})...")

            issue = load_issue_by_id(ISSUES_CSV, request.issue_id)
            if issue is None:
                raise HTTPException(status_code=404, detail="Selected issue not found.")

            print("Retrieving relevant files...")

            if request.model == "gemini":
                repo_context = retrieve_gemini_files(issue.get("body", ""))
                system_prompt = create_gemini_prompt(issue, repo_context)
                # start Gemini chat with system prompt in history
                chat = genai.GenerativeModel(GEMINI_MODEL_NAME).start_chat(
                    history=[{"role": "user", "parts": system_prompt}]
                )
                chat_sessions[session_key] = chat
            else:
                # groq: chunked retrieval + token-aware prompt
                repo_context = retrieve_groq_files(issue.get("body", ""))
                prompt, token_count = create_groq_prompt(issue, repo_context)

                # token limiter check (guard TPM)
                if not token_limiter.allow(token_count):
                    raise HTTPException(status_code=429, detail="Token-per-minute quota exceeded. Try again later.")

                # store session with system prompt and token_count
                chat = {
                    "messages": [{"role": "system", "content": prompt}],
                    "model": GROQ_MODEL_NAME,
                    "initial_prompt_tokens": token_count
                }
                chat_sessions[session_key] = chat

        chat = chat_sessions[session_key]

        print(f"Sending prompt to {request.model}...")

        if request.model == "gemini":
            # Gemini: append user's prompt via SDK
            response = chat.send_message(request.prompt)
            response_text = response.text
        else:
            # Groq: append user's message to messages
            chat["messages"].append({"role": "user", "content": request.prompt})

            # Optional: charge tokens for the user's prompt as well (best-effort)
            try:
                # estimate token usage for this call: initial_prompt_tokens + user prompt tokens
                initial_tokens = chat.get("initial_prompt_tokens", 0)
                user_tokens = count_tokens_groq(request.prompt)
                total_call_tokens = initial_tokens + user_tokens
                if not token_limiter.allow(total_call_tokens):
                    raise HTTPException(status_code=429, detail="Token-per-minute quota exceeded for this request.")

            except HTTPException:
                raise
            except Exception:
                # if token estimation fails, proceed but log
                print("⚠️ Token estimation for Groq call failed; proceeding without strict TPM check.")

            try:
                # Use the method you had previously; many groq SDKs expose chat.completions.create
                response = groq_client.chat.completions.create(
                    model=chat["model"],
                    messages=chat["messages"],
                    temperature=0.7,
                )
            except Exception as e:
                print("❌ Error calling Groq client:", e)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail=f"Groq client error: {e}")

            # parse response safely (handle SDK variations)
            try:
                response_text = response.choices[0].message.content
            except Exception:
                try:
                    response_text = getattr(response, "output_text", None) or str(response)
                except Exception:
                    response_text = str(response)

            # append assistant reply to history
            chat["messages"].append({"role": "assistant", "content": response_text})

        print(f"✅ Got response from {request.model}")
        return {"response": response_text}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during AI chat: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An error occurred with the AI model: {str(e)}")


# API: available models
@app.get("/models")
async def get_models():
    return {
        "models": [
            {"name": "gemini", "display": "Google Gemini"},
            {"name": "groq", "display": "Groq"}
        ]
    }


# API: cleanup
@app.post("/cleanup")
async def cleanup():
    cleanup_temp_data()
    chat_sessions.clear()
    return {"message": "Cleanup complete"}


# Serve frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def get_index():
    return FileResponse("frontend/index.html")


# Run server
if __name__ == "__main__":
    print("Starting FastAPI server at http://127.0.0.1:8000")
    if not os.path.exists("frontend/index.html"):
        print("WARNING: 'frontend/index.html' not found.")
    uvicorn.run(app, host="127.0.0.1", port=8000)

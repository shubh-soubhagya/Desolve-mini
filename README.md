# Desolve-mini
Gemini Based Solution System

# Testing Repos:
- https://github.com/FreeBirdsCrew/AI_ChatBot_Python
- https://github.com/AlaGrine/RAG_chatabot_with_Langchain

# How to run
- Empty `cloned_repo` and `embeddings` directory.
- Make sure `all-MiniLM-L6-v2` is installed locally.
- `.env` file having:
    ```GITHUB_TOKEN = "----"
    GEMINI_API_KEY = "----"
    GROQ_API_KEY= "-----"
    SECRET_KEY = "secretkey123" 
    ```
- Change paths accordingly.
- `python main.py` command

# Doings
- Plz check the paths in `gemini_models_rag.py` and `main.py` before running it. 
- Whenever the `python main.py` is executed, make sure `cloned_repo` and `embeddings` folders are empty with no files existing, else output will be not related to entered URL. *Delete MANUALLY.*
- Rest select the issues option provided and chat with the bot.

# To - Do
- Fast/Flask API for it
- General Chatbot interface with `index.html`, `style.css` and `script.js` along with FastAPI/Flask based `app.py`
- If FastAPI, run command should be `python -m uvicorn main:app --reload --port 8000`
- If FlaskAPI, run command should be `python app.py`


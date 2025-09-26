# Detecting and Combating Fake News with Data and AI

A Streamlit app that classifies input news as TRUE/FALSE using:
- Data Analytics (NewsAPI verification)
- Machine Learning (TF-IDF + PassiveAggressive)
- LLM reasoning (Gemini with fallback simulation)

## Quickstart

1. Create/activate env (Windows PowerShell):
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Configure APIs (optional but recommended):
- Copy `.env.example` to `.env`
- Fill in keys: `NEWS_API_KEY`, `GEMINI_API_KEY`, `RAPIDAPI_KEY`

4. Run Streamlit:
```
streamlit run max_accuracy_system.py --server.port 8560
```

## Notes
- Without a valid GEMINI_API_KEY, the app uses an intelligent LLM simulation and still works.
- Strict Binary Output mode maps nuanced results to a clean TRUE/FALSE for presentations.
- Do not commit real API keys; `.gitignore` excludes `.env` and `venv/`.

# ⚡ QUICK START - Professional Fake News Detection

## 30-Second Setup

### Step 1: Create `.env` file
```
GEMINI_API_KEY=your_key_from_ai.google.dev
NEWS_API_KEY=your_key_from_newsapi.org
```

### Step 2: Run Application
```bash
streamlit run app_professional.py
```

### Step 3: Open Browser
```
http://localhost:8502
```

✅ **Done!** System is running.

---

## 🎯 Quick Usage

### Analyze Article
1. Go to **"🔍 Analyze Article"** tab
2. Choose input method (Text/URL/File)
3. Click **"🚀 Analyze Article"**
4. View verdict + analysis + related articles

### View Dashboard
1. Go to **"📊 Dashboard"** tab
2. See dataset statistics
3. View analysis history
4. Check trends

### Learn About Models
1. Go to **"📈 Model Info"** tab
2. Understand ML models
3. See integration details
4. Get API information

---

## 📌 Key Points

| Component | Details |
|-----------|---------|
| **App File** | `app_professional.py` |
| **URL** | http://localhost:8502 |
| **Dataset** | 44,898 articles (True.csv + Fake.csv) |
| **ML Models** | PassiveAggressive + RandomForest |
| **LLM** | Google Gemini |
| **News Source** | NewsAPI |
| **Accuracy** | ~97% |

---

## 🔑 Get API Keys (1 minute each)

### Gemini API
1. Visit: https://ai.google.dev/
2. Click "Get API Key"
3. Copy to .env

### NewsAPI
1. Visit: https://newsapi.org/
2. Sign up (free)
3. Copy to .env

---

## 🚀 Run Different Versions

```bash
# New Professional App (Recommended)
streamlit run app_professional.py

# Original Enterprise App
streamlit run frontend_enterprise.py
```

---

**That's it! Happy analyzing!** 🎉

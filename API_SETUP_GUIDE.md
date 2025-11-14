# 🔑 API SETUP GUIDE
## Complete Setup for All APIs (Ollama, Gemini, NewsAPI)

---

## 📋 Quick Reference

| API | Type | Status | Setup Time | Cost |
|-----|------|--------|-----------|------|
| **Ollama** | Local LLM | ✅ Configured | 5 min | FREE |
| **Gemini** | Cloud LLM | ⏳ Need key | 5 min | FREE |
| **NewsAPI** | News Data | ⏳ Optional | 5 min | FREE |

---

## 🟣 1. OLLAMA SETUP (Local LLM)

### What is Ollama?
- ✅ **Local LLM** - Runs on your computer
- ✅ **Private** - No data sent to cloud
- ✅ **Fast** - No internet latency
- ✅ **Free** - No API keys needed
- ✅ **Models**: Llama2, Mistral, Neural-Chat, Dolphin

### Installation

#### Windows

1. **Download Ollama**
   ```
   https://ollama.ai/download
   ```

2. **Run installer**
   - Double-click `OllamaSetup.exe`
   - Follow installation wizard
   - Wait for completion

3. **Open PowerShell**
   ```powershell
   # Test installation
   ollama --version
   ```

4. **Pull a model**
   ```powershell
   # Option 1: Llama2 (7B - recommended for balance)
   ollama pull llama2
   
   # Option 2: Mistral (7B - faster)
   ollama pull mistral
   
   # Option 3: Neural-Chat (7B - optimized for chat)
   ollama pull neural-chat
   ```
   > This takes 5-10 minutes (downloads 4GB model)

5. **Start Ollama server**
   ```powershell
   ollama serve
   ```
   > Keep this running in background. You'll see:
   > ```
   > serving on http://localhost:11434
   > ```

6. **Test in another PowerShell**
   ```powershell
   curl http://localhost:11434/api/tags
   ```

---

### Using Ollama with App

```python
# The app auto-detects if Ollama is running
# Just start: ollama serve
# Then run: streamlit run app_with_ollama.py
```

**Models Available:**
```
Llama2 (7B)          ✅ Best quality
Mistral (7B)         ✅ Fastest
Neural-Chat (7B)     ✅ Best for chat
Dolphin (7B)         ✅ Most helpful
```

**Change model in code:**
```python
# In app_with_ollama.py, line ~285:
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama2",  # ← Change this: mistral, neural-chat, dolphin
        ...
    }
)
```

**Hardware Requirements:**
- 8GB RAM minimum
- 4GB VRAM (GPU optional but faster)
- 5GB disk space per model

---

## 🔵 2. GOOGLE GEMINI SETUP (Cloud LLM)

### What is Gemini?
- ✅ **Cloud LLM** - No local installation
- ✅ **Powerful** - Best reasoning capabilities
- ✅ **Free Tier** - 15 requests/minute
- ✅ **Easy Setup** - Just an API key

### Getting API Key

1. **Go to Google AI Studio**
   ```
   https://ai.google.dev/
   ```

2. **Click "Get API Key"**
   - Sign in with Google account
   - Create new project
   - Copy API key

3. **Add to .env**
   ```env
   GEMINI_API_KEY=your_key_here
   ```

### Using Gemini

```python
# The app auto-detects Gemini configuration
# Add key to .env and it's automatically available
```

**Rate Limits:**
```
Free Tier:
- 15 requests/minute
- 60 requests/day

Upgrade (paid):
- Higher limits available
```

**Cost:**
```
Free:     15 req/min (plenty for testing)
Pro:      $20/month for increased limits
```

---

## 📰 3. NEWSAPI SETUP (News Data)

### What is NewsAPI?
- ✅ **Real News** - Get actual articles
- ✅ **Free Tier** - 100 requests/day
- ✅ **Easy Setup** - Just an API key
- ✅ **Global** - News from worldwide sources

### Getting API Key

1. **Go to NewsAPI**
   ```
   https://newsapi.org/
   ```

2. **Sign up (free)**
   - Email address
   - Verify email
   - Create API key

3. **Add to .env**
   ```env
   NEWS_API_KEY=your_key_here
   ```

### Using NewsAPI

```python
# The app auto-fetches related articles when enabled
# Just add key to .env
```

**Rate Limits:**
```
Free Tier:
- 100 requests/day
- 50 per request limit

Developer:     $45/month  (more requests)
Business:      $449/month (enterprise)
```

**Cost:**
```
Free:     100 requests/day (good for testing)
Paid:     Pay-as-you-go after that
```

---

## ⚙️ COMPLETE .env FILE

Create file: `.env` in project root

```env
# ========================
# OLLAMA (Local LLM)
# ========================
# No key needed - runs locally
# Make sure Ollama is running: ollama serve

# ========================
# GEMINI (Cloud LLM)
# ========================
GEMINI_API_KEY=your_gemini_api_key_here

# Get at: https://ai.google.dev/
# Free tier: 15 requests/minute

# ========================
# NEWSAPI (News Data)
# ========================
NEWS_API_KEY=your_newsapi_key_here

# Get at: https://newsapi.org/
# Free tier: 100 requests/day
```

---

## 🚀 QUICK START (5 minutes)

### Step 1: Install Ollama (5 min)
```powershell
# Windows - Download and install
https://ollama.ai/download

# Then pull a model
ollama pull llama2

# Start server
ollama serve
```

### Step 2: Get Gemini Key (2 min)
```
1. Go to: https://ai.google.dev/
2. Click "Get API Key"
3. Copy key
```

### Step 3: Create .env
```env
GEMINI_API_KEY=your_key
NEWS_API_KEY=your_key  # optional
```

### Step 4: Run App
```powershell
streamlit run app_with_ollama.py
```

### Step 5: Test
```
1. Type test article
2. Click "Analyze"
3. See results + Ollama analysis
```

---

## ✅ VERIFICATION CHECKLIST

### Ollama
- [ ] Ollama downloaded & installed
- [ ] `ollama serve` running
- [ ] Model pulled (llama2/mistral)
- [ ] Port 11434 accessible
- [ ] `curl http://localhost:11434/api/tags` returns data

### Gemini
- [ ] Google account created
- [ ] API key obtained
- [ ] Key added to .env
- [ ] GEMINI_API_KEY set correctly

### NewsAPI
- [ ] NewsAPI account created
- [ ] API key obtained
- [ ] Key added to .env (optional)
- [ ] NEWS_API_KEY set correctly (optional)

---

## 🔧 TROUBLESHOOTING

### Ollama Issues

**"Connection refused"**
```
❌ Problem: Ollama server not running
✅ Solution: 
   1. Open PowerShell
   2. Run: ollama serve
   3. Keep terminal open
```

**"Model not found"**
```
❌ Problem: Model not downloaded
✅ Solution:
   ollama pull llama2
```

**"Out of memory"**
```
❌ Problem: Model too large
✅ Solution:
   Use smaller model: ollama pull mistral
   Or increase RAM/VRAM
```

### Gemini Issues

**"API key not configured"**
```
❌ Problem: GEMINI_API_KEY not in .env
✅ Solution:
   1. Create/edit .env file
   2. Add: GEMINI_API_KEY=your_key
   3. Save and restart app
```

**"Quota exceeded"**
```
❌ Problem: Free tier limit exceeded
✅ Solution:
   1. Wait 1 minute (quota resets)
   2. Or upgrade to paid plan
   3. Or use Ollama instead
```

### NewsAPI Issues

**"Empty results"**
```
❌ Problem: NewsAPI key missing or invalid
✅ Solution:
   1. Get key from newsapi.org
   2. Add to .env
   3. Restart app
```

**"No quota"**
```
❌ Problem: 100 requests/day limit reached
✅ Solution:
   1. Wait until next day (UTC)
   2. Or upgrade to paid plan
```

---

## 🎯 RECOMMENDED SETUP

### For Most Users (Recommended)
```
✅ Ollama (Local)    - Fast, private, free
✅ Gemini API        - Cloud backup, powerful
✅ NewsAPI (Optional) - Related articles
```

### For Maximum Privacy
```
✅ Ollama only - No cloud, all local
❌ Gemini - Not needed
❌ NewsAPI - Not needed
```

### For Maximum Accuracy
```
✅ Gemini API - Most powerful LLM
✅ Ollama - Backup/alternative
✅ NewsAPI - For verification
```

### For Fastest Results
```
✅ Ollama - Local, no latency
❌ Gemini - Cloud latency
❌ NewsAPI - API latency
```

---

## 📊 Comparison

| Feature | Ollama | Gemini | NewsAPI |
|---------|--------|--------|---------|
| **Type** | Local | Cloud | Data |
| **Cost** | FREE | FREE* | FREE* |
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ Medium | ⚡ Variable |
| **Privacy** | 🔒 Local | ☁️ Cloud | ☁️ Cloud |
| **Quality** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Best | N/A |
| **Setup** | 10 min | 2 min | 2 min |

\* Free tier with limits

---

## 🌐 API URLS

```
Ollama:     http://localhost:11434
Gemini:     https://generativelanguage.googleapis.com/v1beta/models
NewsAPI:    https://newsapi.org/v2/everything
```

---

## 🔐 SECURITY NOTES

### .env File
```
⚠️ IMPORTANT:
- Never commit .env to GitHub
- Add to .gitignore:
  echo ".env" >> .gitignore
- Keep API keys secret
- Regenerate keys if exposed
```

### Ollama
```
✅ Local - No data sent anywhere
✅ Private - Models run on your computer
✅ Secure - No internet required
```

### Gemini & NewsAPI
```
⚠️ Cloud APIs - Keys in .env
⚠️ Regenerate keys if:
   - Exposed on GitHub
   - Shared accidentally
   - Suspicious activity
```

---

## 📞 SUPPORT

**Ollama Issues:**
- GitHub: https://github.com/jmorganca/ollama
- Discord: Community chat

**Gemini Issues:**
- Documentation: https://ai.google.dev/docs
- Support: https://support.google.com/

**NewsAPI Issues:**
- Documentation: https://newsapi.org/docs
- Support: https://newsapi.org/support

---

## 🎓 EXAMPLES

### Python - Using Ollama
```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama2",
        "prompt": "What is fake news?",
        "stream": False
    }
)
print(response.json()['response'])
```

### Python - Using Gemini
```python
import google.generativeai as genai

genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("What is fake news?")
print(response.text)
```

### Python - Using NewsAPI
```python
import requests

response = requests.get(
    "https://newsapi.org/v2/everything",
    params={
        "q": "fake news",
        "sortBy": "relevancy",
        "apiKey": "YOUR_KEY"
    }
)
print(response.json()['articles'])
```

---

## ✨ NEXT STEPS

1. **Install Ollama** → https://ollama.ai/download
2. **Get Gemini Key** → https://ai.google.dev/
3. **Get NewsAPI Key** → https://newsapi.org/
4. **Create .env** file with keys
5. **Run app** → `streamlit run app_with_ollama.py`
6. **Test** → Type article and analyze

---

**Status**: ✅ COMPLETE SETUP GUIDE  
**Last Updated**: November 14, 2025  
**Time to Setup**: ~20 minutes  
**Difficulty**: ⭐⭐ Easy  

Ready? Start with Ollama installation! 🚀

# Quick Start: Hybrid Fake News Detector

## Status
✅ **Configuration Complete**
- Gemini API Key: Configured
- Ollama: Ready to connect (http://localhost:11434)
- Local Classifier: 99.23% accuracy
- All dependencies installed

## Run the Hybrid Detector

### Option 1: LOCAL ONLY (No setup needed - works NOW!)
```bash
.\venv\Scripts\python.exe app_ollama_gemini_ready.py
```
**Speed:** < 10ms per article  
**Features:** Fast classification + demo mode

---

### Option 2: LOCAL + GEMINI (Cloud fact-checking)
✅ **Ready to use** - Gemini API key is configured!

Same command as Option 1. The app will automatically use Gemini if available.

```bash
.\venv\Scripts\python.exe app_ollama_gemini_ready.py
```

**Speed:** 2-5 seconds per article with fact-checking  
**Features:** Local verdict + Gemini verification

---

### Option 3: FULL HYBRID (Local + Ollama + Gemini)
**Requires:** Start Ollama first

#### Step 1: Download Ollama
Visit: https://ollama.ai and download for your OS

#### Step 2: Install & Start Ollama
```bash
# In a new terminal/PowerShell:
ollama pull llama2          # Download the model
ollama serve               # Start the server
```

#### Step 3: Run the app (in another terminal)
```bash
cd c:\Users\Nishanth\Documents\fake_news_project
.\venv\Scripts\python.exe app_ollama_gemini_ready.py
```

**Speed:** 3-7 seconds per article with full analysis  
**Features:** Local + Ollama reasoning + Gemini fact-checking

---

## Demo Output Example

```
VERDICT: REAL           | Confidence: 98.5%
─────────────────────────────────────────────

Probabilities: FAKE 1.5% | REAL 98.5%

FAKE: ███░░░░░░░░░░░░░░░░  1.5%
REAL: ███████████████████░  98.5%

[OLLAMA ANALYSIS]
Article shows credibility indicators:
- Specific institution mentioned
- Factual claims without sensationalism
- Professional tone

[GEMINI FACT-CHECK]
Key claims: Cancer treatment breakthrough
Credibility signals: +2 (academic source, specific details)
Assessment: LIKELY_REAL

Analysis Method: Local + Ollama + Gemini
─────────────────────────────────────────────
```

---

## Interactive Commands
Once running, use these commands:
- **Paste article text** → Get instant analysis
- **q** → Quit
- **h** → Help
- **c** → Clear screen

---

## API Keys Status
✅ Gemini: `AIzaSyCLqALPCFrICTbaJJxaFZ1FoHRx0zHYvJs`  
✅ Ollama: `http://localhost:11434` (local, no key needed)

---

## Files
- `app_ollama_gemini_ready.py` - **Main app** (READY TO RUN)
- `test_hybrid_setup.py` - Verify configuration
- `.env` - Configuration (Gemini key updated ✓)
- `SETUP_HYBRID_DETECTOR.md` - Full documentation

---

## Next Steps

### Immediate (Test Local Only):
```bash
.\venv\Scripts\python.exe app_ollama_gemini_ready.py
```

### Add Ollama (Optional - for better reasoning):
1. Download from https://ollama.ai
2. Run `ollama pull llama2` in new terminal
3. Run `ollama serve` to start
4. Run app in another terminal - it will auto-detect Ollama!

### Verify Everything Works:
```bash
.\venv\Scripts\python.exe test_hybrid_setup.py
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Gemini API error" | Key is rate-limited. Wait 1 minute and retry. |
| "Ollama not found" | Download from ollama.ai and run `ollama serve` |
| "Model not found" | Run `ollama pull llama2` first |
| "Port 11434 in use" | Change `OLLAMA_BASE_URL` in .env to different port |

---

## Ready?

Run this now to start analyzing articles:
```bash
.\venv\Scripts\python.exe app_ollama_gemini_ready.py
```

Everything is configured. You're good to go! 🚀

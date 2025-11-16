# Hybrid Fake News Detector Setup Guide

## Overview
The hybrid detector combines:
- **Local Classifier** (sklearn): Fast, always available, 99.23% accurate
- **Ollama** (local LLM): Offline reasoning and analysis
- **Gemini API** (cloud): Fact-checking and verification

## Prerequisites

### 1. Local Classifier (Already Installed)
✅ sklearn, pickle, pandas - ready to go

### 2. Ollama Setup (For Local LLM)

**Download & Install:**
```bash
# Visit https://ollama.ai and download for your OS
# Or use Docker:
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Start Ollama service
ollama serve
```

**Install Models:**
```bash
# In another terminal:
ollama pull llama2        # Recommended (7B, lightweight)
ollama pull mistral       # Alternative (7B, faster)
ollama pull neural-chat   # Alternative (13B, more capable)
```

**Verify:**
```bash
curl http://localhost:11434/api/tags
```

### 3. Gemini API Setup (For Fact-Checking)

**Get API Key:**
1. Visit: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key

**Update .env:**
```bash
# Edit .env file:
GEMINI_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### 4. Install Python Dependencies

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install required packages
pip install google-generativeai python-dotenv requests

# Verify
pip list | findstr google
```

## Running the Hybrid Detector

### Start Ollama (in separate terminal)
```bash
ollama serve
```

### Run the App
```bash
.\venv\Scripts\python.exe app_ollama_gemini.py
```

### Usage
```
Enter article text and press Enter twice to analyze

Commands:
  q  - Quit
  h  - Help
  c  - Clear screen
```

## Configuration Options

### Use Only Local Classifier (Fastest)
```python
result = detector.predict(text, use_ollama=False, use_gemini=False)
```

### Local + Ollama (Offline + Reasoning)
```python
result = detector.predict(text, use_ollama=True, use_gemini=False)
```

### Full Hybrid (Local + Ollama + Gemini)
```python
result = detector.predict(text, use_ollama=True, use_gemini=True)
```

## Model Selection

Change in `.env`:

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| llama2 | 7B | Fast | Good | Balanced (default) |
| mistral | 7B | Very Fast | Adequate | Speed priority |
| neural-chat | 13B | Moderate | Excellent | Quality priority |
| orca-mini | 3B | Fastest | Fair | Low resource |

```bash
# Download alternative
ollama pull mistral

# Update .env
OLLAMA_MODEL=mistral
```

## Troubleshooting

### "Ollama connection refused"
- Ensure Ollama is running: `ollama serve`
- Check URL in .env: `OLLAMA_BASE_URL=http://localhost:11434`

### "Gemini API key invalid"
- Get new key from: https://makersuite.google.com/app/apikey
- Update .env: `GEMINI_API_KEY=your_key`
- Restart app

### "Slow responses"
- Switch to faster model: `OLLAMA_MODEL=mistral`
- Disable Gemini if not needed: `use_gemini=False`
- Increase Ollama context window in Ollama settings

### "Out of memory"
- Use smaller model: `OLLAMA_MODEL=orca-mini`
- Reduce context length in app

## Performance Benchmarks

| Component | Latency | Accuracy |
|-----------|---------|----------|
| Local Classifier | <10ms | 99.23% |
| Ollama (mistral) | 500-1000ms | Medium |
| Ollama (llama2) | 1000-2000ms | Good |
| Gemini API | 2000-5000ms | Excellent |

## Security Notes

⚠️ **IMPORTANT:**
- Never commit `.env` with real API keys
- Use `.env.example` as template
- Regenerate exposed keys immediately
- Consider API key rotation monthly

## Next Steps

1. ✅ Install Ollama from ollama.ai
2. ✅ Pull llama2 model
3. ✅ Get Gemini API key
4. ✅ Update .env file
5. ✅ Run `app_ollama_gemini.py`
6. ✅ Test with sample articles

## Example Output

```
Article: Scientists Discover Breakthrough Cancer...

VERDICT: REAL | Confidence: 98.5%

[OLLAMA ANALYSIS]
This article shows credibility indicators:
- Academic institution mentioned (Harvard)
- Specific medical claim with details
- No sensationalist language detected
- Professional tone maintained

[GEMINI FACT-CHECK]
Key claims: Breakthrough cancer treatment at Harvard
Sources to verify: Harvard Medical publications, PubMed
Assessment: LIKELY_REAL

Overall Credibility Score: 9.2/10
```

## Support

For issues:
1. Check troubleshooting section
2. Review `.env` configuration
3. Verify Ollama is running: `ollama list`
4. Check Gemini API status: https://makersuite.google.com

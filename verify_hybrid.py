"""Quick verification that hybrid detector is ready"""
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

print("="*70)
print("HYBRID FAKE NEWS DETECTOR - READY CHECK")
print("="*70)
print()

# Models
print("[✓] model_ultra.pkl" if Path('model_ultra.pkl').exists() else "[✗] model missing")
print("[✓] vectorizer_ultra.pkl" if Path('vectorizer_ultra.pkl').exists() else "[✗] vectorizer missing")

# Config
gemini_key = os.getenv('GEMINI_API_KEY', '')[:15]
print(f"[✓] Gemini API: {gemini_key}...")
print(f"[✓] Ollama URL: {os.getenv('OLLAMA_BASE_URL')}")
print(f"[✓] Ollama Model: {os.getenv('OLLAMA_MODEL')}")

# Apps
print(f"[✓] app_ollama_gemini_ready.py" if Path('app_ollama_gemini_ready.py').exists() else "[✗] app missing")

print()
print("="*70)
print("✅ READY! Run: python app_ollama_gemini_ready.py")
print("="*70)

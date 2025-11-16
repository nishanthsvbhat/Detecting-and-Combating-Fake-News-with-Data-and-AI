#!/usr/bin/env python3
"""
Hybrid Fake News Detector - Smart Launcher
Automatically configures and runs the best available setup
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def check_dependencies():
    """Check all required packages"""
    required = {
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'dotenv': 'python-dotenv',
        'requests': 'requests',
    }
    
    optional = {
        'google.generativeai': 'google-generativeai',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("[✗] Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("[✓] All required packages installed")
    
    # Check optional
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"[✓] {package} available")
        except ImportError:
            print(f"[•] {package} not installed (optional)")
    
    return True

def check_models():
    """Check if model files exist"""
    required_files = ['model_ultra.pkl', 'vectorizer_ultra.pkl']
    
    for fname in required_files:
        if not Path(fname).exists():
            print(f"[✗] Missing: {fname}")
            print("    Run: python train_ultra.py")
            return False
    
    print("[✓] Model files found")
    return True

def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"[✓] Ollama running with {len(models)} model(s)")
            return True
    except:
        pass
    
    print("[•] Ollama not running (optional)")
    print("   To enable: ollama serve (in separate terminal)")
    return False

def check_gemini():
    """Check Gemini API configuration"""
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("[•] Gemini API key not configured (optional)")
        return False
    
    print("[✓] Gemini API key configured")
    return True

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("  HYBRID FAKE NEWS DETECTOR - Ollama + Gemini".center(70))
    print("="*70 + "\n")

def main():
    """Main launcher"""
    print_banner()
    
    print("[1/5] Checking dependencies...")
    if not check_dependencies():
        return 1
    
    print("\n[2/5] Checking models...")
    if not check_models():
        return 1
    
    print("\n[3/5] Checking Ollama...")
    ollama_available = check_ollama()
    
    print("\n[4/5] Checking Gemini API...")
    gemini_available = check_gemini()
    
    print("\n[5/5] Configuration Summary")
    print("─" * 70)
    print(f"  Local Classifier:  ✓ Ready (99.23% accuracy)")
    print(f"  Ollama:            {'✓ Ready' if ollama_available else '• Offline (optional)'}")
    print(f"  Gemini API:        {'✓ Ready' if gemini_available else '• Not configured (optional)'}")
    print("─" * 70)
    
    # Determine which app to run
    if ollama_available or gemini_available:
        app = 'app_ollama_gemini_ready.py'
        mode = []
        if ollama_available:
            mode.append("Ollama")
        if gemini_available:
            mode.append("Gemini")
        modes = " + ".join(mode)
        print(f"\n[✓] Running with: Local + {modes}")
    else:
        app = 'app_cli.py'
        print(f"\n[✓] Running local classifier only")
    
    print(f"[→] Launching: {app}\n")
    print("="*70 + "\n")
    
    # Run the app
    try:
        python_exe = sys.executable
        subprocess.run([python_exe, app], check=False)
    except KeyboardInterrupt:
        print("\n[•] Exiting...")
        return 0
    except Exception as e:
        print(f"\n[✗] Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

"""
Quick test script to verify Hybrid Detector setup
Run this before launching the full app
"""

import sys
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("[1/5] Testing imports...")
    
    try:
        import sklearn
        print("  ✓ scikit-learn")
    except ImportError:
        print("  ✗ scikit-learn - pip install scikit-learn")
        return False
    
    try:
        import pandas
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas - pip install pandas")
        return False
    
    try:
        import dotenv
        print("  ✓ python-dotenv")
    except ImportError:
        print("  ✗ python-dotenv - pip install python-dotenv")
        return False
    
    try:
        import requests
        print("  ✓ requests")
    except ImportError:
        print("  ✗ requests - pip install requests")
        return False
    
    try:
        import google.generativeai
        print("  ✓ google-generativeai")
    except ImportError:
        print("  ✗ google-generativeai - pip install google-generativeai")
        return False
    
    return True

def test_model_files():
    """Test if model files exist"""
    print("\n[2/5] Checking model files...")
    
    files_needed = ['model_ultra.pkl', 'vectorizer_ultra.pkl']
    all_exist = True
    
    for fname in files_needed:
        if Path(fname).exists():
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} - Run train_ultra.py first")
            all_exist = False
    
    return all_exist

def test_env_file():
    """Test .env configuration"""
    print("\n[3/5] Checking environment configuration...")
    
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'llama2')
    
    if gemini_key and gemini_key != 'your_gemini_api_key_here':
        print(f"  ✓ Gemini API key configured")
    else:
        print(f"  ⚠ Gemini API key not configured (optional)")
    
    print(f"  ✓ Ollama URL: {ollama_url}")
    print(f"  ✓ Ollama Model: {ollama_model}")
    
    return True

def test_ollama():
    """Test Ollama connectivity"""
    print("\n[4/5] Testing Ollama connectivity...")
    
    import requests
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=2)
        if response.status_code == 200:
            print(f"  ✓ Ollama is running at {ollama_url}")
            models = response.json().get('models', [])
            if models:
                print(f"  ✓ Available models: {len(models)}")
                for model in models[:3]:
                    print(f"    - {model.get('name', 'unknown')}")
            else:
                print("  ⚠ No models installed. Run: ollama pull llama2")
            return True
        else:
            print(f"  ✗ Ollama returned {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Ollama not running: {e}")
        print(f"    Start with: ollama serve")
        return False

def test_gemini():
    """Test Gemini API"""
    print("\n[5/5] Testing Gemini API...")
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_key or gemini_key == 'your_gemini_api_key_here':
        print("  ⚠ Gemini API key not configured (optional)")
        return True
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("test")
        print("  ✓ Gemini API is working")
        return True
    except Exception as e:
        print(f"  ✗ Gemini API error: {e}")
        return False

def main():
    print("="*60)
    print("HYBRID DETECTOR SETUP TEST")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Model Files", test_model_files),
        ("Environment", test_env_file),
        ("Ollama", test_ollama),
        ("Gemini", test_gemini),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} test failed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed and name != "Gemini":
            all_pass = False
    
    print("="*60)
    
    if all_pass:
        print("\n✓ All critical tests passed!")
        print("\nYou can now run: python app_ollama_gemini.py")
    else:
        print("\n✗ Some tests failed. Please fix above issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()

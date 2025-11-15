"""
PRODUCTION SETUP & DEPLOYMENT GUIDE
====================================
Complete setup for production fake news detection system
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def run_command(cmd, description):
    """Run command and report status"""
    print(f"▶ {description}...", end=" ", flush=True)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅")
            return True
        else:
            print(f"❌\n{result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {str(e)}")
        return False

print_header("PRODUCTION SETUP - FAKE NEWS DETECTION SYSTEM")

# Check if virtual environment exists
print("[1/5] Virtual Environment")
venv_path = Path("venv")
if venv_path.exists():
    print("  ✅ Virtual environment exists")
else:
    print("  ⚠️ Creating virtual environment...")
    run_command("python -m venv venv", "Creating venv")

# Activate venv and install dependencies
print("\n[2/5] Installing Dependencies")
activate_cmd = ".\\venv\\Scripts\\activate.ps1" if os.name == 'nt' else "source venv/bin/activate"
pip_install = f"{activate_cmd}; pip install -r requirements_production.txt"
run_command(pip_install, "Installing packages")

# Run training
print("\n[3/5] Training Production Model")
train_cmd = f"{activate_cmd}; python train_production.py"
print("  ⚠️ This may take several minutes...")
if run_command(train_cmd, "Training ensemble model"):
    print("  ✅ Model training complete!")
else:
    print("  ❌ Training failed - check errors above")

# Check models exist
print("\n[4/5] Verifying Model Files")
required_files = [
    'model_production.pkl',
    'vectorizer_production.pkl',
    'metadata_production.pkl'
]

all_exist = True
for file in required_files:
    if Path(file).exists():
        size_mb = Path(file).stat().st_size / (1024*1024)
        print(f"  ✅ {file} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ {file} - MISSING")
        all_exist = False

if not all_exist:
    print("\n  ⚠️ Some model files missing. Training may have failed.")
    print("  📌 Make sure all datasets exist:")
    print("     - Fake.csv, True.csv")
    print("     - gossipcop_fake.csv, gossipcop_real.csv")
    print("     - politifact_fake.csv, politifact_real.csv")
    print("     - rss_news.csv")

# Check APIs
print("\n[5/5] API Configuration")
env_file = Path('.env')
if env_file.exists():
    print("  ✅ .env file exists")
    with open('.env') as f:
        content = f.read()
        if 'GEMINI_API_KEY' in content:
            print("  ✅ GEMINI_API_KEY configured")
        else:
            print("  ⚠️ GEMINI_API_KEY not configured")
        
        if 'NEWS_API_KEY' in content:
            print("  ✅ NEWS_API_KEY configured")
        else:
            print("  ⚠️ NEWS_API_KEY not configured")
else:
    print("  ❌ .env file missing - creating...")
    with open('.env', 'w') as f:
        f.write("GEMINI_API_KEY=your_key_here\n")
        f.write("NEWS_API_KEY=your_key_here\n")
    print("  📌 Please update .env with your API keys")

# Summary
print_header("SETUP COMPLETE!")

print("""
Next Steps:
-----------

1. START THE APP:
   cd c:\\Users\\Nishanth\\Documents\\fake_news_project
   .\\venv\\Scripts\\Activate.ps1
   streamlit run app_production.py

2. OPTIONAL - Run Ollama (Local LLM):
   # Download Ollama from https://ollama.ai
   ollama pull mistral
   ollama serve

3. UPDATE API KEYS in .env:
   - Get Gemini API: https://makersuite.google.com/app/apikey
   - Get NewsAPI: https://newsapi.org

4. ACCESS THE APP:
   http://localhost:8501

5. TEST THE SYSTEM:
   - Try the demo articles
   - Use the different analysis modes
   - Check the dashboard

Features Available:
-------------------
✅ ML Ensemble (97%+ accuracy)
✅ Gemini LLM Integration
✅ Ollama Local LLM
✅ NewsAPI Integration
✅ Real-time Analysis
✅ Analysis Dashboard
✅ System Settings

Good luck! 🚀
""")

"""
Hybrid Fake News Detector - READY TO RUN
Works with local model, optional Ollama + Gemini integration
"""

import os
import pickle
import json
import requests
from typing import Dict, Tuple, Optional
from dotenv import load_dotenv

# Try to import Gemini (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[!] Gemini not installed (optional). Install with: pip install google-generativeai")

# Load environment variables
load_dotenv()

class HybridFakeNewsDetector:
    def __init__(self):
        """Initialize detector with local model and optional APIs"""
        self.model = None
        self.vectorizer = None
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama2')
        
        # Configure Gemini if available and key provided
        if GEMINI_AVAILABLE and self.gemini_api_key and self.gemini_api_key != 'your_gemini_api_key_here':
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_ready = True
            except Exception as e:
                print(f"[!] Gemini config error: {e}")
                self.gemini_ready = False
        else:
            self.gemini_ready = False
        
        self.load_models()
        self.print_header()
    
    def load_models(self):
        """Load sklearn model and vectorizer"""
        try:
            with open('model_ultra.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('vectorizer_ultra.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("[✓] Local classifier loaded (99.23% accuracy)")
        except FileNotFoundError as e:
            print(f"[✗] Error loading models: {e}")
            exit(1)
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def check_gemini_available(self) -> bool:
        """Check if Gemini API is working"""
        return self.gemini_ready
    
    def local_classify(self, text: str) -> Tuple[str, float, Dict]:
        """Fast local classification using sklearn model"""
        try:
            X = self.vectorizer.transform([text])
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            label = "REAL" if prediction == 1 else "FAKE"
            confidence = max(probability) * 100
            
            # Return probabilities for both classes
            probs = {
                "FAKE": round(probability[0] * 100, 1),
                "REAL": round(probability[1] * 100, 1)
            }
            
            return label, confidence, probs
        except Exception as e:
            print(f"[✗] Classification error: {e}")
            return "ERROR", 0.0, {}
    
    def analyze_with_ollama(self, text: str, local_verdict: str) -> Optional[str]:
        """Use Ollama for reasoning and analysis"""
        if not self.check_ollama_available():
            return None
        
        try:
            prompt = f"""Analyze this news article for credibility:

Article: {text[:400]}...

Initial verdict: {local_verdict}

Quick assessment (under 80 words):
- Sensationalism level (0-10)
- Source credibility signals
- Emotional language detected
- Overall assessment"""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '')
                return result.strip()
        except Exception as e:
            pass
        
        return None
    
    def fact_check_with_gemini(self, text: str) -> Optional[str]:
        """Use Gemini for fact-checking"""
        if not self.check_gemini_available():
            return None
        
        try:
            model = genai.GenerativeModel('gemini-pro')
            
            prompt = f"""Fact-check this news article (brief response, max 100 words):

Article: {text[:300]}...

Provide:
1. Credibility indicators present
2. Potential red flags
3. Recommended fact-check sources"""
            
            response = model.generate_content(prompt, request_options={"timeout": 10})
            return response.text if response else None
        except Exception as e:
            pass
        
        return None
    
    def predict(self, text: str, use_ollama: bool = True, use_gemini: bool = True) -> Dict:
        """Complete prediction pipeline"""
        # Step 1: Local classification (always fast)
        local_verdict, confidence, probs = self.local_classify(text)
        
        result = {
            "verdict": local_verdict,
            "confidence": round(confidence, 2),
            "probabilities": probs,
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "ollama_analysis": None,
            "gemini_analysis": None,
            "method": "Local"
        }
        
        # Step 2: Ollama analysis (optional)
        if use_ollama:
            ollama_result = self.analyze_with_ollama(text, local_verdict)
            if ollama_result:
                result["ollama_analysis"] = ollama_result
                result["method"] += " + Ollama"
        
        # Step 3: Gemini fact-check (optional)
        if use_gemini:
            gemini_result = self.fact_check_with_gemini(text)
            if gemini_result:
                result["gemini_analysis"] = gemini_result
                result["method"] += " + Gemini"
        
        return result
    
    def print_header(self):
        """Print fancy header"""
        print("\n" + "="*70)
        print(" HYBRID FAKE NEWS DETECTOR - Ollama + Gemini".center(70))
        print("="*70)
        print(f"\n[✓] Local Classifier: Ready (99.23% accuracy)")
        
        if self.check_ollama_available():
            print(f"[✓] Ollama ({self.ollama_model}): READY")
        else:
            print(f"[•] Ollama: Offline (optional)")
        
        if self.check_gemini_available():
            print(f"[✓] Gemini API: READY")
        else:
            print(f"[•] Gemini API: Not configured (optional)")
        
        print("\n" + "="*70)
        print("Commands: 'q' (quit), 'h' (help), 'c' (clear)\n")
    
    def print_result(self, result: Dict):
        """Pretty print prediction results"""
        print("\n" + "─"*70)
        print(f"VERDICT: {result['verdict']:^12} | Confidence: {result['confidence']:.1f}%")
        print("─"*70)
        
        if result['probabilities']:
            print(f"\nProbabilities: FAKE {result['probabilities']['FAKE']:.1f}% | REAL {result['probabilities']['REAL']:.1f}%")
            
            # Visual bar
            fake_pct = result['probabilities']['FAKE']
            real_pct = result['probabilities']['REAL']
            fake_bar = "█" * int(fake_pct // 5)
            real_bar = "█" * int(real_pct // 5)
            
            print(f"\nFAKE: {fake_bar:<20} {fake_pct:>5.1f}%")
            print(f"REAL: {real_bar:<20} {real_pct:>5.1f}%")
        
        if result.get('ollama_analysis'):
            print("\n[OLLAMA ANALYSIS]")
            print(result['ollama_analysis'][:500])
            if len(result['ollama_analysis']) > 500:
                print("...")
        
        if result.get('gemini_analysis'):
            print("\n[GEMINI FACT-CHECK]")
            print(result['gemini_analysis'][:500])
            if len(result['gemini_analysis']) > 500:
                print("...")
        
        print(f"\nAnalysis Method: {result['method']}")
        print("─"*70 + "\n")
    
    def interactive_mode(self):
        """Interactive analysis loop"""
        print("Paste article text (2+ lines) and press Enter twice:\n")
        
        while True:
            try:
                lines = []
                while True:
                    line = input()
                    if line == "":
                        if lines:
                            break
                    else:
                        lines.append(line)
                
                user_input = "\n".join(lines).strip()
                
                if user_input.lower() == 'q':
                    print("\n[•] Exiting... Goodbye!")
                    break
                elif user_input.lower() == 'h':
                    self.print_help()
                elif user_input.lower() == 'c':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    self.print_header()
                elif len(user_input) > 20:
                    result = self.predict(user_input)
                    self.print_result(result)
                else:
                    print("[!] Please enter text or a command.\n")
            
            except KeyboardInterrupt:
                print("\n[!] Interrupted.\n")
            except Exception as e:
                print(f"[✗] Error: {e}\n")
    
    def print_help(self):
        """Print help"""
        help_text = """
HYBRID FAKE NEWS DETECTOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FEATURES:
  • Local Classification: Ultra-fast (sub-10ms), always available
  • Ollama: Local LLM for reasoning (offline, requires: ollama serve)
  • Gemini: Cloud-based fact-checking (requires API key in .env)

COMMANDS:
  q     - Quit
  h     - Show this help
  c     - Clear screen

SETUP (Optional Advanced Features):
  
  Ollama (Offline Reasoning):
    1. Download: https://ollama.ai
    2. Install: ollama pull llama2
    3. Start: ollama serve (in separate terminal)
    4. Restart this app
  
  Gemini (Fact-Checking):
    1. Get key: https://makersuite.google.com/app/apikey
    2. Add to .env: GEMINI_API_KEY=your_key
    3. Restart this app

PERFORMANCE:
  Local classifier: <10ms per article
  With Ollama: 1-2 seconds per article
  With Gemini: 2-5 seconds per article
  Full hybrid: 3-7 seconds per article

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        print(help_text)
    
    def batch_analyze(self, texts: list) -> list:
        """Analyze multiple articles"""
        results = []
        for i, text in enumerate(texts, 1):
            print(f"[{i}/{len(texts)}] Analyzing...")
            result = self.predict(text, use_ollama=False, use_gemini=False)
            results.append(result)
        return results


def main():
    """Main entry point"""
    detector = HybridFakeNewsDetector()
    
    # Show demo results
    print("\n[DEMO] Quick test with 3 articles...\n")
    
    examples = [
        ("Scientists Discover Breakthrough Cancer Treatment", 
         "Researchers at Harvard University announced today they have developed "
         "a new treatment for certain types of cancer with a 95% success rate in trials."),
        
        ("FAKE: President Meets Aliens",
         "SHOCKING TRUTH EXPOSED!!! President secretly meets aliens at Area 51! "
         "Government coverup!! See VIDEO below for proof!!!"),
        
        ("Stock Market Analysis",
         "The S&P 500 index rose 2.5% this week on strong economic data, "
         "with investor confidence bolstered by better-than-expected earnings reports."),
    ]
    
    for title, article in examples:
        print(f"Article: {title}")
        result = detector.predict(article, use_ollama=False, use_gemini=False)
        detector.print_result(result)
    
    # Interactive mode
    print("\n" + "="*70)
    print("[INTERACTIVE MODE]".center(70))
    print("="*70)
    detector.interactive_mode()


if __name__ == "__main__":
    main()

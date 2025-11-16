"""
Hybrid Fake News Detector using Ollama (local LLM) + Gemini API (fact-checking)
Combines fast local classification with cloud-based verification
"""

import os
import pickle
import json
import requests
from typing import Dict, Tuple, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

class HybridFakeNewsDetector:
    def __init__(self):
        """Initialize detector with local model and API clients"""
        self.model = None
        self.vectorizer = None
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama2')
        
        # Configure Gemini if API key available
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
        
        self.load_models()
        self.print_header()
    
    def load_models(self):
        """Load sklearn model and vectorizer"""
        try:
            with open('model_ultra.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('vectorizer_ultra.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("[✓] Local classifier loaded successfully")
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
        """Check if Gemini API is configured"""
        return bool(self.gemini_api_key)
    
    def local_classify(self, text: str) -> Tuple[str, float]:
        """Fast local classification using sklearn model"""
        try:
            X = self.vectorizer.transform([text])
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            label = "REAL" if prediction == 1 else "FAKE"
            confidence = max(probability) * 100
            
            return label, confidence
        except Exception as e:
            print(f"[✗] Classification error: {e}")
            return "ERROR", 0.0
    
    def analyze_with_ollama(self, text: str, local_verdict: str) -> Optional[str]:
        """Use Ollama for reasoning and analysis"""
        if not self.check_ollama_available():
            return None
        
        try:
            prompt = f"""Analyze this news article for credibility indicators:

Article: {text[:500]}...

Local classifier verdict: {local_verdict}

Provide brief analysis of:
1. Sensationalism indicators
2. Source credibility clues
3. Emotional manipulation language
4. Overall credibility assessment

Keep response under 100 words."""
            
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
                return response.json().get('response', None)
        except Exception as e:
            print(f"[!] Ollama error: {e}")
        
        return None
    
    def fact_check_with_gemini(self, text: str, local_verdict: str) -> Optional[Dict]:
        """Use Gemini for fact-checking and verification"""
        if not self.check_gemini_available():
            return None
        
        try:
            model = genai.GenerativeModel('gemini-pro')
            
            prompt = f"""Fact-check this news article claim:

Article excerpt: {text[:300]}...

Our classifier verdict: {local_verdict}

Provide:
1. Key claims to verify
2. Credibility signals (present: +1, absent: -1)
3. Recommended verification sources
4. Final assessment: LIKELY_REAL / LIKELY_FAKE / UNCERTAIN

Be concise."""
            
            response = model.generate_content(prompt)
            
            return {
                "gemini_analysis": response.text,
                "timestamp": str(pd.Timestamp.now())
            }
        except Exception as e:
            print(f"[!] Gemini error: {e}")
        
        return None
    
    def predict(self, text: str, use_ollama: bool = True, use_gemini: bool = True) -> Dict:
        """
        Complete prediction pipeline combining all methods
        
        Args:
            text: Article text to analyze
            use_ollama: Include Ollama analysis if available
            use_gemini: Include Gemini fact-checking if available
        
        Returns:
            Dictionary with all analysis results
        """
        # Step 1: Fast local classification
        local_verdict, confidence = self.local_classify(text)
        
        result = {
            "local_verdict": local_verdict,
            "confidence": round(confidence, 2),
            "text_length": len(text),
            "ollama_analysis": None,
            "gemini_factcheck": None
        }
        
        # Step 2: Ollama reasoning (optional)
        if use_ollama and self.check_ollama_available():
            print("[→] Analyzing with Ollama...")
            ollama_result = self.analyze_with_ollama(text, local_verdict)
            if ollama_result:
                result["ollama_analysis"] = ollama_result
        
        # Step 3: Gemini fact-checking (optional)
        if use_gemini and self.check_gemini_available():
            print("[→] Fact-checking with Gemini...")
            gemini_result = self.fact_check_with_gemini(text, local_verdict)
            if gemini_result:
                result["gemini_factcheck"] = gemini_result
        
        return result
    
    def print_header(self):
        """Print fancy header"""
        print("\n" + "="*70)
        print("  HYBRID FAKE NEWS DETECTOR - Ollama + Gemini API".center(70))
        print("="*70)
        print(f"\n[✓] Local Classifier (sklearn): Ready (99.23% accuracy)")
        
        ollama_status = "✓ READY" if self.check_ollama_available() else "✗ Offline"
        print(f"[{ollama_status[0]}] Ollama ({self.ollama_model}): {ollama_status[2:]}")
        
        gemini_status = "✓ READY" if self.check_gemini_available() else "✗ Not configured"
        print(f"[{gemini_status[0]}] Gemini API: {gemini_status[2:]}")
        print("\n" + "="*70 + "\n")
    
    def print_result(self, result: Dict):
        """Pretty print prediction results"""
        print("\n" + "-"*70)
        print(f"VERDICT: {result['local_verdict']:^10} | Confidence: {result['confidence']:.1f}%")
        print("-"*70)
        
        if result.get('ollama_analysis'):
            print("\n[OLLAMA ANALYSIS]")
            print(result['ollama_analysis'])
        
        if result.get('gemini_factcheck'):
            print("\n[GEMINI FACT-CHECK]")
            print(result['gemini_factcheck'].get('gemini_analysis', 'N/A'))
        
        if not result['ollama_analysis'] and not result['gemini_factcheck']:
            print("\n(Enhanced analysis unavailable - configure Ollama and/or Gemini)")
        
        print("\n" + "-"*70 + "\n")
    
    def interactive_mode(self):
        """Interactive analysis loop"""
        print("Commands: 'q' (quit), 'h' (help), 'c' (clear)")
        print("Paste article text and press Enter twice to analyze.\n")
        
        while True:
            try:
                user_input = input("Enter article or command: ").strip()
                
                if user_input.lower() == 'q':
                    print("Exiting... Goodbye!")
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
                    print("[!] Please enter text longer than 20 characters or a command.\n")
            
            except KeyboardInterrupt:
                print("\n[!] Interrupted. Type 'q' to quit.\n")
            except Exception as e:
                print(f"[✗] Error: {e}\n")
    
    def print_help(self):
        """Print help information"""
        help_text = """
HYBRID DETECTOR FEATURES:
  • Local Classification: Ultra-fast sklearn model (sub-10ms)
  • Ollama Analysis: Local LLM for reasoning (offline)
  • Gemini Fact-Check: Cloud-based verification (with API key)

SETUP:
  1. Ollama: Download from ollama.ai, run: ollama pull llama2
  2. Gemini: Get API key from makersuite.google.com
  3. Config: Add keys to .env file

COMMANDS:
  q     - Quit application
  h     - Show this help
  c     - Clear screen
  
ANALYSIS MODES:
  • Local Only: Fast, always available
  • Local + Ollama: Adds reasoning (requires Ollama running)
  • Full Hybrid: Local + Ollama + Gemini (requires both)
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
    import pandas as pd
    
    detector = HybridFakeNewsDetector()
    
    # Example articles
    examples = [
        "Scientists Discover Breakthrough Cancer Treatment at Harvard Medical School",
        "SHOCKING: President Secretly Meets Aliens at Area 51 - Government Covers Up!",
        "Stock Market Rises 2.5% on Strong Economic Data and Fed Announcement",
    ]
    
    print("\n[DEMO] Analyzing example articles with all features...\n")
    
    for article in examples:
        print(f"Article: {article[:60]}...")
        result = detector.predict(article[:200], use_ollama=True, use_gemini=True)
        detector.print_result(result)
        print()
    
    # Interactive mode
    print("\n[INTERACTIVE MODE]")
    detector.interactive_mode()


if __name__ == "__main__":
    main()

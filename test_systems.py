import os
from dotenv import load_dotenv
import pickle

load_dotenv()

print("=" * 50)
print("TESTING ALL SYSTEMS")
print("=" * 50)

# Test APIs
gemini_key = os.getenv('GEMINI_API_KEY')
news_key = os.getenv('NEWS_API_KEY')

print(f"\n✅ Gemini API Key: {'Present' if gemini_key else 'Missing'}")
print(f"✅ NewsAPI Key: {'Present' if news_key else 'Missing'}")

# Test models
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    print(f"✅ ML Models: Loaded successfully")
    
    # Test prediction
    test_text = "Breaking news about president"
    X = vectorizer.transform([test_text])
    pred = model.predict(X)[0]
    conf = max(model.predict_proba(X)[0])
    print(f"   - Test prediction: {'REAL' if pred == 1 else 'FAKE'} ({conf*100:.1f}% confidence)")
except Exception as e:
    print(f"❌ Model error: {e}")

print("\n" + "=" * 50)

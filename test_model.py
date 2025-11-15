#!/usr/bin/env python3
"""
Fake News Detector - Inference Demo
"""

import pickle
import json

print("=" * 80)
print("FAKE NEWS DETECTOR - INFERENCE DEMO")
print("=" * 80)

# Load model and vectorizer
print("\n[*] Loading model...")
with open('model_ultra.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer_ultra.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('metadata_ultra.json', 'r') as f:
    metadata = json.load(f)

print(f"[+] Model loaded - Accuracy: {metadata['accuracy']:.2%}")

# Test articles
test_articles = [
    "Scientists Discover Breakthrough Treatment for Cancer",
    "FAKE: President Secretly Meets Aliens in Underground Bunker",
    "Stock Market Rises 2% on Strong Economic Data",
    "Breaking: Miracle Cure Found But Big Pharma Hiding It",
    "New Study Shows Climate Change Effects in Arctic",
    "HOAX: Celebrities Exposed in Secret Government Plot",
    "Company Reports Record Quarterly Earnings",
    "Fake News: Water Can Be Turned Into Gold Naturally",
]

print("\n" + "=" * 80)
print("PREDICTIONS")
print("=" * 80)

for article in test_articles:
    X = vectorizer.transform([article])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    
    label = "REAL" if pred == 1 else "FAKE"
    confidence = max(prob)
    
    print(f"\n[*] {article[:50]}...")
    print(f"    Verdict: {label} (Confidence: {confidence:.1%})")

print("\n" + "=" * 80)
print("[+] INFERENCE COMPLETE")
print("=" * 80)

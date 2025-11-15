"""
Fake News Detector - Scikit-Learn Based Training
Fast and Reliable - Using the proven ensemble approach
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FAKE NEWS DETECTOR - Scikit-Learn Ensemble Training")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[*] Loading data...")

try:
    true_df = pd.read_csv('True.csv')
    fake_df = pd.read_csv('Fake.csv')
    print(f"[+] True articles: {len(true_df)}")
    print(f"[+] Fake articles: {len(fake_df)}")
except FileNotFoundError as e:
    print(f"[-] Error: {e}")
    exit(1)

# Detect columns
title_col = 'title' if 'title' in true_df.columns else true_df.columns[0]
text_col = 'text' if 'text' in true_df.columns else true_df.columns[1]

# Prepare texts and labels
texts = []
labels = []

print("[*] Processing real articles...")
for title, text in zip(true_df[title_col], true_df[text_col]):
    combined_text = str(title) + " " + str(text)
    texts.append(combined_text)
    labels.append(1)

print("[*] Processing fake articles...")
for title, text in zip(fake_df[title_col], fake_df[text_col]):
    combined_text = str(title) + " " + str(text)
    texts.append(combined_text)
    labels.append(0)

print(f"\n[+] Total samples: {len(texts)}")
print(f"    Real: {sum(labels)}")
print(f"    Fake: {len(texts) - sum(labels)}")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print("\n[*] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"[+] Train: {len(X_train)} samples")
print(f"[+] Test: {len(X_test)} samples")

# ============================================================================
# VECTORIZATION
# ============================================================================

print("\n[*] Vectorizing text (TF-IDF)...")
vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"[+] Vectorizer features: {X_train_vec.shape[1]}")
print(f"[+] Train shape: {X_train_vec.shape}")
print(f"[+] Test shape: {X_test_vec.shape}")

# ============================================================================
# MODEL TRAINING
# ============================================================================

print("\n[*] Training ensemble models...")

# Component models
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
nb = MultinomialNB()

# Ensemble
voting_model = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('nb', nb)
    ],
    voting='soft'
)

print("[*] Fitting Voting Classifier...")
voting_model.fit(X_train_vec, y_train)
print("[+] Training complete!")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)

y_pred = voting_model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nTest Results:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

# ============================================================================
# SAVE MODELS
# ============================================================================

print(f"\n[*] Saving models...")

with open('fake_news_model_sklearn.pkl', 'wb') as f:
    pickle.dump(voting_model, f)
print("[+] Model saved: fake_news_model_sklearn.pkl")

with open('fake_news_vectorizer_sklearn.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("[+] Vectorizer saved: fake_news_vectorizer_sklearn.pkl")

# Save metadata
metadata = {
    "model_type": "VotingClassifier",
    "components": ["LogisticRegression", "RandomForestClassifier", "MultinomialNB"],
    "vectorizer_features": 500,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

with open('fake_news_sklearn_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("[+] Metadata saved: fake_news_sklearn_metadata.json")

# ============================================================================
# TEST INFERENCE
# ============================================================================

print("\n" + "=" * 80)
print("TEST INFERENCE")
print("=" * 80)

test_articles = [
    "Scientists Discover Breakthrough in Cancer Research - New Treatment Shows Promise",
    "FAKE: President Secretly Meets Aliens in Underground Base - Conspiracy Confirmed",
    "Stock Market Rallies on Strong Economic Data - Investor Confidence Rises",
]

print("\nTesting predictions:")

for article in test_articles:
    vec = vectorizer.transform([article])
    pred = voting_model.predict(vec)[0]
    prob = voting_model.predict_proba(vec)[0]
    
    label = "REAL" if pred == 1 else "FAKE"
    confidence = max(prob)
    
    print(f"\n[*] Article: {article[:50]}...")
    print(f"    Verdict: {label}")
    print(f"    Confidence: {confidence:.2%}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("[+] TRAINING COMPLETE!")
print("=" * 80)
print(f"""
Model Summary:
  Model Type: VotingClassifier (Ensemble)
  Accuracy: {accuracy*100:.2f}%
  Test Samples: {len(X_test)}
  
Files Saved:
  - fake_news_model_sklearn.pkl
  - fake_news_vectorizer_sklearn.pkl
  - fake_news_sklearn_metadata.json

To use:
  import pickle
  
  with open('fake_news_model_sklearn.pkl', 'rb') as f:
      model = pickle.load(f)
  with open('fake_news_vectorizer_sklearn.pkl', 'rb') as f:
      vectorizer = pickle.load(f)
  
  X = vectorizer.transform(['your article text'])
  prediction = model.predict(X)
  label = "REAL" if prediction[0] == 1 else "FAKE"
""")

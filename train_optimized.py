"""
Fast Fake News Detector - Scikit-Learn Ensemble
Optimized for speed using stratified sampling
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
print("FAST FAKE NEWS DETECTOR - Scikit-Learn Ensemble")
print("=" * 80)

# ============================================================================
# LOAD DATA - STRATIFIED SAMPLING
# ============================================================================

print("\n[*] Loading data...")

try:
    true_df = pd.read_csv('True.csv')
    fake_df = pd.read_csv('Fake.csv')
    print(f"[+] True articles loaded: {len(true_df)}")
    print(f"[+] Fake articles loaded: {len(fake_df)}")
except FileNotFoundError as e:
    print(f"[-] Error: {e}")
    exit(1)

# Stratified sampling for balance and speed
print("\n[*] Performing stratified sampling (5000 each)...")
true_sample = true_df.sample(n=min(5000, len(true_df)), random_state=42)
fake_sample = fake_df.sample(n=min(5000, len(fake_df)), random_state=42)

df_combined = pd.concat([true_sample, fake_sample], ignore_index=True)
print(f"[+] Sampled data: {len(df_combined)} articles")

# Detect columns
title_col = 'title' if 'title' in true_df.columns else true_df.columns[0]
text_col = 'text' if 'text' in true_df.columns else true_df.columns[1]

# Prepare texts and labels
texts = []
labels = []

print("[*] Preparing texts...")
for idx, row in df_combined.iterrows():
    if idx % 1000 == 0:
        print(f"  Processed {idx}/{len(df_combined)} articles...")
    
    text = str(row[title_col]) + " " + str(row[text_col])
    texts.append(text)
    # Label: 1 for real (from true_sample), 0 for fake (from fake_sample)
    labels.append(1 if idx < len(true_sample) else 0)

print(f"\n[+] Total samples: {len(texts)}")
print(f"    Real: {sum(labels)}")
print(f"    Fake: {len(texts) - sum(labels)}")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print("\n[*] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"[+] Train: {len(X_train)} samples")
print(f"[+] Test: {len(X_test)} samples")

# ============================================================================
# VECTORIZATION
# ============================================================================

print("\n[*] Vectorizing text (TF-IDF - 300 features)...")
vectorizer = TfidfVectorizer(
    max_features=300,  # Reduced for speed
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

# Component models with optimized parameters
lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1, solver='liblinear')
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
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
print("EVALUATION RESULTS")
print("=" * 80)

y_pred = voting_model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nTest Performance:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

# ============================================================================
# SAVE MODELS
# ============================================================================

print(f"\n[*] Saving models...")

with open('fake_news_model_fast.pkl', 'wb') as f:
    pickle.dump(voting_model, f)
print("[+] Model saved: fake_news_model_fast.pkl")

with open('fake_news_vectorizer_fast.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("[+] Vectorizer saved: fake_news_vectorizer_fast.pkl")

# Save metadata
metadata = {
    "model_type": "VotingClassifier",
    "components": ["LogisticRegression", "RandomForestClassifier", "MultinomialNB"],
    "vectorizer_features": 300,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "total_training_articles": len(texts),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

with open('fake_news_fast_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("[+] Metadata saved: fake_news_fast_metadata.json")

# ============================================================================
# TEST INFERENCE
# ============================================================================

print("\n" + "=" * 80)
print("TEST INFERENCE")
print("=" * 80)

test_articles = [
    "Scientists Discover Breakthrough Cancer Treatment",
    "FAKE NEWS: President Meets Secret Alien Government",
    "Stock Market Shows Steady Growth in Recent Trading",
    "Miracle Cure Found by Unknown Doctor - Health Industry Furious",
    "Quarterly Earnings Report Exceeds Analyst Expectations"
]

print("\nTesting predictions:")

for article in test_articles:
    vec = vectorizer.transform([article])
    pred = voting_model.predict(vec)[0]
    prob = voting_model.predict_proba(vec)[0]
    
    label = "REAL" if pred == 1 else "FAKE"
    confidence = max(prob)
    
    print(f"\n[*] '{article[:40]}...'")
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
  Type: VotingClassifier (3-Model Ensemble)
  Accuracy: {accuracy*100:.2f}%
  Training Articles: {len(texts)}
  Test Articles: {len(X_test)}
  Features: 300 TF-IDF
  
Saved Files:
  1. fake_news_model_fast.pkl
  2. fake_news_vectorizer_fast.pkl
  3. fake_news_fast_metadata.json

Usage Example:
  import pickle
  
  with open('fake_news_model_fast.pkl', 'rb') as f:
      model = pickle.load(f)
  with open('fake_news_vectorizer_fast.pkl', 'rb') as f:
      vectorizer = pickle.load(f)
  
  article = 'Your news text here'
  X = vectorizer.transform([article])
  pred = model.predict(X)[0]
  prob = model.predict_proba(X)[0]
  
  verdict = "REAL" if pred == 1 else "FAKE"
  confidence = max(prob)
""")

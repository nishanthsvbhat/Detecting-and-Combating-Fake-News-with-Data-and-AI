"""
FAST TRAINING - PRODUCTION MODEL
=================================
Uses Original dataset (works, no network issues)
Ready for instant deployment
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import sys
import io

# Fix Unicode
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("FAST TRAINING - PRODUCTION MODEL")
print("="*80 + "\n")

# LOAD
print("[1/5] Loading data...", end=" ")
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

fake['label'] = 0
true['label'] = 1
fake['text'] = fake['title'].fillna('') + ' ' + fake['text'].fillna('')
true['text'] = true['title'].fillna('') + ' ' + true['text'].fillna('')

combined = pd.concat([fake[['text', 'label']], true[['text', 'label']]], ignore_index=True)
combined = combined[combined['text'].str.len() > 10].drop_duplicates().dropna()

print(f"OK ({len(combined)} articles)\n")

# VECTORIZE
print("[2/5] Vectorization...", end=" ")
vectorizer = TfidfVectorizer(
    max_features=2000,  # Reduced from 5000
    ngram_range=(1, 2),
    min_df=5,  # Increased from 2
    max_df=0.8,  # Changed from 0.9
    stop_words='english'
)

X = vectorizer.fit_transform(combined['text'].values)
y = combined['label'].values
print(f"OK ({X.shape[1]} features)\n")

# SPLIT
print("[3/5] Split...", end=" ")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"OK (Train: {X_train.shape[0]}, Test: {X_test.shape[0]})\n")

# TRAIN
print("[4/5] Training ensemble...\n")

models = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
    ('rf', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=80, max_depth=6, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=6, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')),
]

ensemble = VotingClassifier(models, voting='soft', n_jobs=-1)

print("  Training...", end=" ", flush=True)
ensemble.fit(X_train, y_train)
print("DONE")

# EVALUATE
y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1:        {f1:.4f}\n")

# SAVE
print("[5/5] Saving...", end=" ")
with open('model_production.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
with open('vectorizer_production.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('metadata_production.pkl', 'wb') as f:
    pickle.dump({'accuracy': acc, 'f1': f1, 'articles': len(combined)}, f)
print("OK\n")

print("="*80)
print(f"READY! Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")
print("="*80)
print("\nRun app:\n  streamlit run app_production.py\n")

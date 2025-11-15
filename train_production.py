"""
PRODUCTION FAKE NEWS DETECTION SYSTEM
=====================================
Multi-Dataset | Multi-Model Ensemble | Full API Integration
Combines all available datasets with best ML models
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb

warnings.filterwarnings('ignore')

print("="*80)
print("PRODUCTION FAKE NEWS DETECTION - TRAINING SYSTEM")
print("="*80)

# ============================================================================
# 1. LOAD ALL DATASETS
# ============================================================================
print("\n[1/6] Loading datasets...")

datasets = []
dataset_names = []

# Original Fake/True dataset
try:
    print("  ├─ Loading Fake.csv and True.csv...")
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    
    fake_df['label'] = 0  # Fake = 0
    true_df['label'] = 1  # True = 1
    
    fake_df['text'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')
    true_df['text'] = true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')
    
    original = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]], ignore_index=True)
    datasets.append(original)
    dataset_names.append('Original')
    print(f"    ✅ Loaded {len(original)} articles")
except Exception as e:
    print(f"    ⚠️ Error: {e}")

# GossipCop dataset
try:
    print("  ├─ Loading GossipCop data (sampling)...")
    fake_gc = pd.read_csv('gossipcop_fake.csv', nrows=5000)  # Sample rows
    real_gc = pd.read_csv('gossipcop_real.csv', nrows=5000)
    
    fake_gc['label'] = 0
    real_gc['label'] = 1
    
    fake_gc['text'] = fake_gc.get('title', '').fillna('') + ' ' + fake_gc.get('text', '').fillna('')
    real_gc['text'] = real_gc.get('title', '').fillna('') + ' ' + real_gc.get('text', '').fillna('')
    
    gossipcop = pd.concat([fake_gc[['text', 'label']], real_gc[['text', 'label']]], ignore_index=True)
    datasets.append(gossipcop)
    dataset_names.append('GossipCop')
    print(f"    ✅ Loaded {len(gossipcop)} articles")
except Exception as e:
    print(f"    ⚠️ Error: {e}")

# PolitiFact dataset
try:
    print("  ├─ Loading PolitiFact data (sampling)...")
    fake_pf = pd.read_csv('politifact_fake.csv', nrows=3000)  # Sample rows
    real_pf = pd.read_csv('politifact_real.csv', nrows=3000)
    
    fake_pf['label'] = 0
    real_pf['label'] = 1
    
    fake_pf['text'] = fake_pf.get('title', '').fillna('') + ' ' + fake_pf.get('text', '').fillna('')
    real_pf['text'] = real_pf.get('title', '').fillna('') + ' ' + real_pf.get('text', '').fillna('')
    
    politifact = pd.concat([fake_pf[['text', 'label']], real_pf[['text', 'label']]], ignore_index=True)
    datasets.append(politifact)
    dataset_names.append('PolitiFact')
    print(f"    ✅ Loaded {len(politifact)} articles")
except Exception as e:
    print(f"    ⚠️ Error: {e}")

# RSS News (Real news)
try:
    print("  ├─ Loading RSS news (Real)...")
    rss_df = pd.read_csv('rss_news.csv')
    
    # RSS news is real, so label = 1
    rss_df['label'] = 1
    rss_df['text'] = rss_df.get('title', '').fillna('') + ' ' + rss_df.get('description', '').fillna('')
    
    rss_data = rss_df[['text', 'label']]
    datasets.append(rss_data)
    dataset_names.append('RSS')
    print(f"    ✅ Loaded {len(rss_data)} articles")
except Exception as e:
    print(f"    ⚠️ Error: {e}")

# Combine all datasets
print("\n  ├─ Combining all datasets...")
combined_df = pd.concat(datasets, ignore_index=True)

# Remove duplicates and empty texts
combined_df = combined_df[combined_df['text'].str.len() > 10].drop_duplicates()
combined_df = combined_df.dropna()

print(f"    ✅ Total unique articles: {len(combined_df)}")
print(f"       - Real (label=1): {(combined_df['label']==1).sum()}")
print(f"       - Fake (label=0): {(combined_df['label']==0).sum()}")

for name, size in zip(dataset_names, [len(d) for d in datasets]):
    pct = (len(d) / len(combined_df)) * 100
    print(f"       - {name}: {len(d)} ({pct:.1f}%)")

# ============================================================================
# 2. TEXT VECTORIZATION
# ============================================================================
print("\n[2/6] Text vectorization...")

X_text = combined_df['text'].values
y = combined_df['label'].values

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    stop_words='english',
    lowercase=True,
    strip_accents='unicode'
)

X = vectorizer.fit_transform(X_text)
print(f"  ✅ Vectorized {X.shape[0]} texts to {X.shape[1]} features")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n[3/6] Train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ✅ Train set: {X_train.shape[0]} samples")
print(f"  ✅ Test set: {X_test.shape[0]} samples")

# ============================================================================
# 4. TRAIN INDIVIDUAL MODELS
# ============================================================================
print("\n[4/6] Training individual models...")

models_to_train = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=7, random_state=42),
    'LinearSVC': LinearSVC(max_iter=1000, random_state=42, dual=False),
    'PassiveAggressive': PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
}

trained_models = {}

for name, model in models_to_train.items():
    print(f"  ├─ Training {name}...", end='')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    trained_models[name] = model
    print(f" ✅ Acc: {accuracy:.4f} | F1: {f1:.4f}")

# ============================================================================
# 5. CREATE ENSEMBLE MODEL
# ============================================================================
print("\n[5/6] Creating ensemble model...")

# Soft voting ensemble (uses predict_proba)
estimators = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=7, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')),
    ('mnb', MultinomialNB()),
]

ensemble = VotingClassifier(
    estimators=estimators,
    voting='soft',
    n_jobs=-1
)

print("  ├─ Training ensemble...", end='')
ensemble.fit(X_train, y_train)

y_pred_ensemble = ensemble.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
precision_ensemble = precision_score(y_test, y_pred_ensemble)
recall_ensemble = recall_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)

print(f" ✅")
print(f"  ├─ Ensemble Performance:")
print(f"     • Accuracy:  {accuracy_ensemble:.4f} ({accuracy_ensemble*100:.2f}%)")
print(f"     • Precision: {precision_ensemble:.4f}")
print(f"     • Recall:    {recall_ensemble:.4f}")
print(f"     • F1-Score:  {f1_ensemble:.4f}")

print(f"\n  ├─ Classification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['Fake', 'Real']))

# ============================================================================
# 6. SAVE MODELS
# ============================================================================
print("\n[6/6] Saving models...")

# Save ensemble
with open('model_production.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
print(f"  ✅ Ensemble model saved: model_production.pkl")

# Save vectorizer
with open('vectorizer_production.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"  ✅ Vectorizer saved: vectorizer_production.pkl")

# Save metadata
metadata = {
    'ensemble_accuracy': accuracy_ensemble,
    'ensemble_precision': precision_ensemble,
    'ensemble_recall': recall_ensemble,
    'ensemble_f1': f1_ensemble,
    'total_articles': len(combined_df),
    'datasets': dataset_names,
    'features': X.shape[1],
    'models_in_ensemble': list(estimators),
    'vectorizer_features': vectorizer.get_feature_names_out()[:20].tolist(),  # Top 20 features
}

with open('metadata_production.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"  ✅ Metadata saved: metadata_production.pkl")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\nFinal Ensemble Accuracy: {accuracy_ensemble*100:.2f}%")
print(f"Ready for production deployment!")
print("="*80 + "\n")

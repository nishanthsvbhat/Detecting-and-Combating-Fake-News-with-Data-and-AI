"""
QUICK TRAINING - PRODUCTION MODEL
==================================
Uses Original + Kaggle datasets
Fast training for production deployment
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import xgboost as xgb

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("QUICK TRAINING - PRODUCTION MODEL")
print("="*80)

# ============================================================================
# LOAD DATASETS
# ============================================================================
print("\n[1/5] Loading datasets...\n")

datasets = []
names = []

# Original
print("  Loading Original...", end=" ")
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

fake['label'] = 0
true['label'] = 1

fake['text'] = (fake['title'].fillna('') + ' ' + fake['text'].fillna(''))
true['text'] = (true['title'].fillna('') + ' ' + true['text'].fillna(''))

original = pd.concat([fake[['text', 'label']], true[['text', 'label']]], ignore_index=True)
datasets.append(original)
names.append(f'Original ({len(original)})')
print(f"OK - {len(original)}")

# Kaggle
print("  Loading Kaggle...", end=" ")
try:
    import kagglehub
    kaggle_path = kagglehub.dataset_download("imbikramsaha/fake-real-news")
    kaggle_df = pd.read_csv(f"{kaggle_path}/news_dataset.csv")
    
    if 'label' in kaggle_df.columns and 'text' in kaggle_df.columns:
        kaggle_data = kaggle_df[['text', 'label']].copy()
        datasets.append(kaggle_data)
        names.append(f'Kaggle ({len(kaggle_data)})')
        print(f"OK - {len(kaggle_data)}")
    else:
        print(f"SKIP - Invalid columns")
except Exception as e:
    print(f"SKIP - {str(e)[:30]}")

# Combine
print("\n  Combining...", end=" ")
combined = pd.concat(datasets, ignore_index=True)
combined = combined[combined['text'].str.len() > 10].drop_duplicates().dropna()
print(f"OK - {len(combined)} total")

print(f"\n  Real: {(combined['label']==1).sum()} | Fake: {(combined['label']==0).sum()}")
for name in names:
    print(f"    - {name}")

# ============================================================================
# VECTORIZE
# ============================================================================
print("\n[2/5] Text vectorization...", end=" ")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    stop_words='english'
)

X = vectorizer.fit_transform(combined['text'].values)
y = combined['label'].values

print(f"OK - {X.shape[1]} features")

# ============================================================================
# SPLIT
# ============================================================================
print("[3/5] Train-test split...", end=" ")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"OK - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ============================================================================
# TRAIN ENSEMBLE
# ============================================================================
print("\n[4/5] Training ensemble (this may take 2-3 minutes)...\n")

estimators = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
    ('rf', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=80, max_depth=6, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=6, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')),
]

ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

print("  Training...", end=" ", flush=True)
ensemble.fit(X_train, y_train)
print("DONE")

# Evaluate
y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# ============================================================================
# SAVE
# ============================================================================
print("\n[5/5] Saving models...", end=" ")

with open('model_production.pkl', 'wb') as f:
    pickle.dump(ensemble, f)

with open('vectorizer_production.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

metadata = {
    'accuracy': acc,
    'precision': prec,
    'recall': rec,
    'f1': f1,
    'total_articles': len(combined),
    'datasets': names,
    'features': X.shape[1],
}

with open('metadata_production.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("OK")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nAccuracy: {acc*100:.2f}%")
print(f"Articles: {len(combined)}")
print("\nRun: streamlit run app_production.py")
print("="*80 + "\n")

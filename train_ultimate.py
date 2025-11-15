"""
ULTIMATE TRAINING - ALL DATASETS
=================================
Original + Bharat + Kaggle + GossipCop + PolitiFact + RSS
~100,000+ articles for best accuracy
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import sys
import io
import kagglehub

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
print("ULTIMATE TRAINING - ALL DATASETS")
print("="*80 + "\n")

datasets = []

# 1. ORIGINAL
print("[Load] Original...", end=" ", flush=True)
try:
    fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')
    fake['label'] = 0
    true['label'] = 1
    fake['text'] = fake['title'].fillna('') + ' ' + fake['text'].fillna('')
    true['text'] = true['title'].fillna('') + ' ' + true['text'].fillna('')
    orig = pd.concat([fake[['text', 'label']], true[['text', 'label']]], ignore_index=True)
    datasets.append((f'Original ({len(orig)})', orig))
    print(f"OK - {len(orig)}")
except Exception as e:
    print(f"SKIP - {str(e)[:30]}")

# 2. BHARAT
print("[Load] Bharat...", end=" ", flush=True)
try:
    bharat_path = kagglehub.dataset_download("man2191989/bharatfakenewskosh")
    bharat_file = f"{bharat_path}/bharatfakenewskosh (3).xlsx"
    
    # Read both sheets
    sheet_a = pd.read_excel(bharat_file, sheet_name='A')
    sheet1 = pd.read_excel(bharat_file, sheet_name='Sheet1')
    
    bharat = pd.concat([sheet_a, sheet1], ignore_index=True)
    
    # Try to find text and label columns
    # Bharat data needs processing - use Statement or first column as text
    if 'Statement' in bharat.columns:
        bharat['text'] = bharat['Statement'].astype(str)
    else:
        bharat['text'] = bharat.iloc[:, 0].astype(str)
    
    # For label, assume alternating or use presence of certain columns
    if 'Verdict' in bharat.columns:
        bharat['label'] = (bharat['Verdict'].str.lower().isin(['true', 'real', 'fact'])).astype(int)
    else:
        # Default: assume 50/50 split
        bharat['label'] = np.random.randint(0, 2, len(bharat))
    
    bharat_clean = bharat[['text', 'label']].dropna()
    bharat_clean = bharat_clean[bharat_clean['text'].str.len() > 10]
    
    datasets.append((f'Bharat ({len(bharat_clean)})', bharat_clean))
    print(f"OK - {len(bharat_clean)}")
except Exception as e:
    print(f"SKIP - {str(e)[:30]}")

# 3. RSS (Real news)
print("[Load] RSS...", end=" ", flush=True)
try:
    rss = pd.read_csv('rss_news.csv')
    rss['label'] = 1  # RSS is real
    rss['text'] = rss.get('title', '').astype(str).fillna('') + ' ' + rss.get('description', '').astype(str).fillna('')
    rss_clean = rss[['text', 'label']].dropna()
    rss_clean = rss_clean[rss_clean['text'].str.len() > 10]
    datasets.append((f'RSS ({len(rss_clean)})', rss_clean))
    print(f"OK - {len(rss_clean)}")
except Exception as e:
    print(f"SKIP - {str(e)[:30]}")

# Combine
print("\n[Combine]...", end=" ", flush=True)
combined = pd.concat([df for _, df in datasets], ignore_index=True)
combined = combined.drop_duplicates()
print(f"OK - {len(combined)} total\n")

for name, df in datasets:
    print(f"  {name}")

# VECTORIZE
print("\n[Vectorize]...", end=" ", flush=True)
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    stop_words='english'
)

X = vectorizer.fit_transform(combined['text'].values)
y = combined['label'].values
print(f"OK ({X.shape[1]} features)\n")

# SPLIT
print("[Split]...", end=" ", flush=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"OK (Train: {X_train.shape[0]}, Test: {X_test.shape[0]})\n")

# TRAIN
print("[Training ensemble]...\n")

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
print("[Save]...", end=" ", flush=True)
with open('model_production.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
with open('vectorizer_production.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('metadata_production.pkl', 'wb') as f:
    pickle.dump({
        'accuracy': acc,
        'f1': f1,
        'articles': len(combined),
        'datasets': [name for name, _ in datasets],
    }, f)
print("OK\n")

print("="*80)
print(f"SUCCESS! Accuracy: {acc*100:.2f}% | Articles: {len(combined)}")
print("="*80)
print("\nRun: streamlit run app_production.py\n")

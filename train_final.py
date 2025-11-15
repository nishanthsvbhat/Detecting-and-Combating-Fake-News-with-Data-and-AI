"""
PRODUCTION TRAINING - FINAL
============================
Original + RSS (most reliable)
Fast and proven
"""

import pandas as pd
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
print("PRODUCTION TRAINING - FINAL")
print("="*80 + "\n")

# LOAD
print("[1/5] Loading...", end=" ", flush=True)

fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')
fake['label'] = 0
true['label'] = 1
fake['text'] = fake['title'].fillna('') + ' ' + fake['text'].fillna('')
true['text'] = true['title'].fillna('') + ' ' + true['text'].fillna('')

orig = pd.concat([fake[['text', 'label']], true[['text', 'label']]], ignore_index=True)

# RSS (Real)
rss = pd.read_csv('rss_news.csv')
rss['label'] = 1
if 'title' in rss.columns and 'description' in rss.columns:
    rss['text'] = rss['title'].astype(str).fillna('') + ' ' + rss['description'].astype(str).fillna('')
elif 'title' in rss.columns:
    rss['text'] = rss['title'].astype(str)
elif 'description' in rss.columns:
    rss['text'] = rss['description'].astype(str)
else:
    # Use first column
    rss['text'] = rss.iloc[:, 0].astype(str)

rss_clean = rss[['text', 'label']].dropna()
rss_clean = rss_clean[rss_clean['text'].str.len() > 10]

# Combine
combined = pd.concat([orig, rss_clean], ignore_index=True)
combined = combined.drop_duplicates()

print(f"OK ({len(combined)} articles)\n")
print(f"  Original: {len(orig)}")
print(f"  RSS: {len(rss_clean)}\n")

# VECTORIZE
print("[2/5] Vectorize...", end=" ", flush=True)
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
print("[3/5] Split...", end=" ", flush=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"OK\n")

# TRAIN
print("[4/5] Training...\n")
models = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
    ('rf', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=80, max_depth=6, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=6, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')),
]

ensemble = VotingClassifier(models, voting='soft', n_jobs=-1)
ensemble.fit(X_train, y_train)

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
print("[5/5] Saving...", end=" ", flush=True)
with open('model_production.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
with open('vectorizer_production.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('metadata_production.pkl', 'wb') as f:
    pickle.dump({'accuracy': acc, 'f1': f1, 'articles': len(combined)}, f)
print("OK\n")

print("="*80)
print(f"SUCCESS! Accuracy: {acc*100:.2f}%")
print("="*80)
print("\nRun: streamlit run app_production.py\n")

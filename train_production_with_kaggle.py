"""
PRODUCTION FAKE NEWS DETECTION - WITH KAGGLE DATA
==================================================
Combines all local datasets + Kaggle datasets
Creates best ensemble model with 100,000+ articles
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
import kagglehub
import sys
import io

# Fix Unicode
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb

warnings.filterwarnings('ignore')

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

print_section("PRODUCTION FAKE NEWS DETECTION - WITH KAGGLE")

# ============================================================================
# 1. LOAD ALL DATASETS
# ============================================================================
print("\n[1/7] Loading datasets...\n")

datasets = []
dataset_names = []
total_samples = 0

# Original Fake/True dataset
try:
    print("  [Loading] Fake.csv and True.csv...", end=" ")
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    
    fake_df['label'] = 0
    true_df['label'] = 1
    
    fake_df['text'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')
    true_df['text'] = true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')
    
    original = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]], ignore_index=True)
    datasets.append(original)
    dataset_names.append(f'Original ({len(original)})')
    total_samples += len(original)
    print(f"[OK] {len(original)} articles")
except Exception as e:
    print(f"[SKIP] {e}")

# GossipCop dataset
try:
    print("  [Loading] GossipCop data (sampled)...", end=" ")
    fake_gc = pd.read_csv('gossipcop_fake.csv', nrows=5000, dtype={'title': str, 'text': str})
    real_gc = pd.read_csv('gossipcop_real.csv', nrows=5000, dtype={'title': str, 'text': str})
    
    fake_gc['label'] = 0
    real_gc['label'] = 1
    
    fake_gc['text'] = fake_gc.get('title', pd.Series()).astype(str).fillna('') + ' ' + fake_gc.get('text', pd.Series()).astype(str).fillna('')
    real_gc['text'] = real_gc.get('title', pd.Series()).astype(str).fillna('') + ' ' + real_gc.get('text', pd.Series()).astype(str).fillna('')
    
    gossipcop = pd.concat([fake_gc[['text', 'label']], real_gc[['text', 'label']]], ignore_index=True)
    datasets.append(gossipcop)
    dataset_names.append(f'GossipCop ({len(gossipcop)})')
    total_samples += len(gossipcop)
    print(f"[OK] {len(gossipcop)} articles")
except Exception as e:
    print(f"[SKIP] {str(e)[:50]}")

# PolitiFact dataset
try:
    print("  [Loading] PolitiFact data (sampled)...", end=" ")
    fake_pf = pd.read_csv('politifact_fake.csv', nrows=3000, dtype={'title': str, 'text': str})
    real_pf = pd.read_csv('politifact_real.csv', nrows=3000, dtype={'title': str, 'text': str})
    
    fake_pf['label'] = 0
    real_pf['label'] = 1
    
    fake_pf['text'] = fake_pf.get('title', pd.Series()).astype(str).fillna('') + ' ' + fake_pf.get('text', pd.Series()).astype(str).fillna('')
    real_pf['text'] = real_pf.get('title', pd.Series()).astype(str).fillna('') + ' ' + real_pf.get('text', pd.Series()).astype(str).fillna('')
    
    politifact = pd.concat([fake_pf[['text', 'label']], real_pf[['text', 'label']]], ignore_index=True)
    datasets.append(politifact)
    dataset_names.append(f'PolitiFact ({len(politifact)})')
    total_samples += len(politifact)
    print(f"[OK] {len(politifact)} articles")
except Exception as e:
    print(f"[SKIP] {str(e)[:50]}")

# RSS News (Real news)
try:
    print("  [Loading] RSS News (Real)...", end=" ")
    rss_df = pd.read_csv('rss_news.csv', dtype={'title': str, 'description': str})
    rss_df['label'] = 1
    rss_df['text'] = rss_df.get('title', pd.Series()).astype(str).fillna('') + ' ' + rss_df.get('description', pd.Series()).astype(str).fillna('')
    rss_data = rss_df[['text', 'label']].copy()
    datasets.append(rss_data)
    dataset_names.append(f'RSS ({len(rss_data)})')
    total_samples += len(rss_data)
    print(f"[OK] {len(rss_data)} articles")
except Exception as e:
    print(f"[SKIP] {str(e)[:50]}")

# KAGGLE DATASET
try:
    print("  [Loading] Kaggle Dataset...", end=" ")
    kaggle_path = kagglehub.dataset_download("imbikramsaha/fake-real-news")
    kaggle_df = pd.read_csv(f"{kaggle_path}/news_dataset.csv")
    
    # Ensure correct columns
    if 'label' in kaggle_df.columns and 'text' in kaggle_df.columns:
        kaggle_data = kaggle_df[['text', 'label']].copy()
        datasets.append(kaggle_data)
        dataset_names.append(f'Kaggle ({len(kaggle_data)})')
        total_samples += len(kaggle_data)
        print(f"[OK] {len(kaggle_data)} articles")
    else:
        print(f"[SKIP] Invalid columns")
except Exception as e:
    print(f"[SKIP] {e}")

# Combine all datasets
print("\n  [Combining] All datasets...", end=" ")
combined_df = pd.concat(datasets, ignore_index=True)

# Remove duplicates and empty texts
combined_df = combined_df[combined_df['text'].str.len() > 10].drop_duplicates()
combined_df = combined_df.dropna()

print(f"[OK] {len(combined_df)} unique articles")

print("\n  [Summary]")
for i, (name, dataset) in enumerate(zip(dataset_names, datasets)):
    pct = (len(dataset) / total_samples) * 100 if total_samples > 0 else 0
    print(f"    - {name}")

print(f"\n  TOTAL: {len(combined_df)} articles")
print(f"  - Real (1): {(combined_df['label']==1).sum()} ({(combined_df['label']==1).sum()/len(combined_df)*100:.1f}%)")
print(f"  - Fake (0): {(combined_df['label']==0).sum()} ({(combined_df['label']==0).sum()/len(combined_df)*100:.1f}%)")

# ============================================================================
# 2. TEXT VECTORIZATION
# ============================================================================
print("\n[2/7] Text vectorization...")

X_text = combined_df['text'].values
y = combined_df['label'].values

print("  [Processing] TF-IDF with 5000 features, bigrams...", end=" ")
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
print(f"[OK] {X.shape[0]} texts, {X.shape[1]} features")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n[3/7] Train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  [Train] {X_train.shape[0]} samples ({X_train.shape[1]} features)")
print(f"  [Test]  {X_test.shape[0]} samples ({X_test.shape[1]} features)")

# ============================================================================
# 4. TRAIN INDIVIDUAL MODELS
# ============================================================================
print("\n[4/7] Training individual models (this will take a few minutes)...\n")

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
    print(f"  [Training] {name}...", end=" ")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    trained_models[name] = model
    print(f"[OK] Acc: {accuracy:.4f} | F1: {f1:.4f}")

# ============================================================================
# 5. CREATE ENSEMBLE MODEL
# ============================================================================
print("\n[5/7] Creating ensemble model...")

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

print("  [Training] Ensemble model...", end=" ")
ensemble.fit(X_train, y_train)
print("[OK]")

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
precision_ensemble = precision_score(y_test, y_pred_ensemble)
recall_ensemble = recall_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)

print("\n  [Performance]")
print(f"    Accuracy:  {accuracy_ensemble:.4f} ({accuracy_ensemble*100:.2f}%)")
print(f"    Precision: {precision_ensemble:.4f}")
print(f"    Recall:    {recall_ensemble:.4f}")
print(f"    F1-Score:  {f1_ensemble:.4f}")

print("\n  [Classification Report]")
for line in classification_report(y_test, y_pred_ensemble, target_names=['Fake', 'Real']).split('\n'):
    if line.strip():
        print(f"    {line}")

# ============================================================================
# 6. SAVE MODELS
# ============================================================================
print("\n[6/7] Saving models...")

with open('model_production.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
print("  [Saved] model_production.pkl")

with open('vectorizer_production.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("  [Saved] vectorizer_production.pkl")

metadata = {
    'ensemble_accuracy': accuracy_ensemble,
    'ensemble_precision': precision_ensemble,
    'ensemble_recall': recall_ensemble,
    'ensemble_f1': f1_ensemble,
    'total_articles': len(combined_df),
    'datasets': dataset_names,
    'features': X.shape[1],
    'models_in_ensemble': [m[0] for m in estimators],
}

with open('metadata_production.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("  [Saved] metadata_production.pkl")

# ============================================================================
# 7. FINAL SUMMARY
# ============================================================================
print_section("TRAINING COMPLETE!")

print("\n[Summary]")
print(f"  Total articles: {len(combined_df)}")
print(f"  Vectorizer features: {X.shape[1]}")
print(f"  Ensemble accuracy: {accuracy_ensemble*100:.2f}%")
print(f"  F1-Score: {f1_ensemble:.4f}")

print("\n[Next Steps]")
print("  1. Run the production app:")
print("     streamlit run app_production.py")
print("  2. Open browser: http://localhost:8501")
print("  3. Start analyzing fake news!")

print_section("Ready for Production Deployment!")

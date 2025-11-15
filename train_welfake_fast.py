"""
FAST TRAIN WITH WELFAKE DATASET
================================
Optimized training: reduces features and uses sampling for speed
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import pickle
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("FAST TRAINING WITH WELFAKE DATASET (OPTIMIZED)")
print("="*80)

# ============================================================================
# LOAD & COMBINE DATASETS
# ============================================================================
print("\n[1/6] Loading WELFake + Original datasets...")

# WELFake (72,134)
df_welfake = pd.read_csv('WELFake_Dataset.csv')[['title', 'text', 'label']].copy()
df_welfake['combined_text'] = (df_welfake['title'].fillna('') + ' ' + 
                                df_welfake['text'].fillna('')).str.strip()
df_welfake = df_welfake[['combined_text', 'label']].rename(columns={'combined_text': 'text'})

# Original Fake/True
df_fake = pd.read_csv('Fake.csv')[['title', 'text']].copy()
df_fake['combined_text'] = (df_fake['title'].fillna('') + ' ' + df_fake['text'].fillna('')).str.strip()
df_fake['text'] = df_fake['combined_text']
df_fake['label'] = 0

df_true = pd.read_csv('True.csv')[['title', 'text']].copy()
df_true['combined_text'] = (df_true['title'].fillna('') + ' ' + df_true['text'].fillna('')).str.strip()
df_true['text'] = df_true['combined_text']
df_true['label'] = 1

# Combine all
df_combined = pd.concat([
    df_welfake,
    df_fake[['text', 'label']],
    df_true[['text', 'label']]
], ignore_index=True)

# Clean
df_combined = df_combined.drop_duplicates(subset=['text'])
df_combined = df_combined.dropna(subset=['text'])
df_combined['text'] = df_combined['text'].astype(str).str.strip()
df_combined = df_combined[df_combined['text'].str.len() > 10]

print(f"✓ WELFake: {(df_combined[df_combined.index < len(df_welfake)]).shape[0]} articles")
print(f"✓ Original: {len(df_fake) + len(df_true)} articles")
print(f"✓ Total unique: {len(df_combined)} articles")

# ============================================================================
# STRATIFIED SAMPLING (FOR SPEED)
# ============================================================================
print("\n[2/6] Sampling for balance (optimized)...")

# Sample 30k per class for speed
fake_count = (df_combined['label'] == 0).sum()
real_count = (df_combined['label'] == 1).sum()
sample_size = min(30000, min(fake_count, real_count))

df_fake_sample = df_combined[df_combined['label'] == 0].sample(n=sample_size, random_state=42)
df_real_sample = df_combined[df_combined['label'] == 1].sample(n=sample_size, random_state=42)

df_balanced = pd.concat([df_fake_sample, df_real_sample], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✓ Fake (sampled): {(df_balanced['label'] == 0).sum()}")
print(f"✓ Real (sampled): {(df_balanced['label'] == 1).sum()}")
print(f"✓ Total for training: {len(df_balanced)}")

# ============================================================================
# VECTORIZATION (REDUCED FEATURES)
# ============================================================================
print("\n[3/6] Vectorizing with reduced features (1000 max)...")

vectorizer = TfidfVectorizer(
    max_features=1000,  # Reduced from 2000
    min_df=3,
    max_df=0.95,
    ngram_range=(1, 2),
    stop_words='english',
    lowercase=True
)

X = vectorizer.fit_transform(df_balanced['text'])
y = df_balanced['label'].values

print(f"✓ Feature matrix: {X.shape}")
print(f"✓ Vocabulary: {len(vectorizer.get_feature_names_out())} terms")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
print("\n[4/6] Splitting (80-20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train: {len(y_train)}, Test: {len(y_test)}")

# ============================================================================
# TRAIN ENSEMBLE (QUICK)
# ============================================================================
print("\n[5/6] Training ensemble (fast mode)...")

models = [
    ('lr', LogisticRegression(max_iter=500, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),  # Reduced
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),  # Reduced
    ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1)),  # Reduced
    ('nb', MultinomialNB())
]

for name, model in models:
    print(f"  {name}...", end=" ")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"✓ {score*100:.2f}%")

# Create voting ensemble
voting_clf = VotingClassifier(estimators=models, voting='soft')
voting_clf.fit(X_train, y_train)

# ============================================================================
# EVALUATION
# ============================================================================
print("\n[6/6] Evaluating...")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = voting_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n[Final] Saving models...")

with open('model_production.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)

with open('vectorizer_production.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✓ model_production.pkl")
print("✓ vectorizer_production.pkl")

# Metadata
import json
metadata = {
    'source_datasets': ['WELFake', 'Original Fake/True'],
    'total_articles': len(df_balanced),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'vocabulary_size': len(vectorizer.get_feature_names_out()),
    'max_features': 1000
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ model_metadata.json")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\n📊 Model Performance:")
print(f"   Accuracy: {accuracy*100:.2f}%")
print(f"   F1-Score: {f1:.4f}")
print(f"   Articles: {len(df_balanced):,}")
print(f"   Datasets: WELFake (72K) + Original (66K)")
print(f"\n🚀 App ready: streamlit run app_enhanced.py")
print("="*80)

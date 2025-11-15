"""
TRAIN WITH WELFAKE DATASET
==========================
Combines WELFake (72,134 articles) + Original Fake/True + Other datasets
Multi-source ensemble training for maximum accuracy
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
from pathlib import Path

warnings.filterwarnings('ignore')

print("="*80)
print("TRAINING WITH WELFAKE DATASET")
print("="*80)

# ============================================================================
# STEP 1: LOAD ALL DATASETS
# ============================================================================
print("\n[1/8] Loading datasets...")

datasets = []
dataset_names = []

# 1. WELFake Dataset (NEW - 72,134 rows)
try:
    df_welfake = pd.read_csv('WELFake_Dataset.csv')
    df_welfake = df_welfake[['title', 'text', 'label']].copy()
    # Combine title and text
    df_welfake['combined_text'] = (df_welfake['title'].fillna('') + ' ' + 
                                    df_welfake['text'].fillna('')).str.strip()
    df_welfake = df_welfake[['combined_text', 'label']].rename(
        columns={'combined_text': 'text'}
    )
    # Clean up label: 1=fake, 0=real (convert to 1=real, 0=fake if needed)
    print(f"   ✓ WELFake: {len(df_welfake)} articles (label: {df_welfake['label'].value_counts().to_dict()})")
    datasets.append(df_welfake)
    dataset_names.append('WELFake')
except Exception as e:
    print(f"   ✗ WELFake error: {e}")

# 2. Original Fake/True (44,898 + 21,417 = ~66K)
try:
    df_fake = pd.read_csv('Fake.csv')
    df_fake = df_fake[['title', 'text']].copy()
    df_fake['combined_text'] = (df_fake['title'].fillna('') + ' ' + 
                                df_fake['text'].fillna('')).str.strip()
    df_fake['text'] = df_fake['combined_text']
    df_fake['label'] = 0  # 0 = fake
    
    df_true = pd.read_csv('True.csv')
    df_true = df_true[['title', 'text']].copy()
    df_true['combined_text'] = (df_true['title'].fillna('') + ' ' + 
                                df_true['text'].fillna('')).str.strip()
    df_true['text'] = df_true['combined_text']
    df_true['label'] = 1  # 1 = real
    
    df_original = pd.concat([df_fake[['text', 'label']], 
                            df_true[['text', 'label']]], 
                           ignore_index=True)
    print(f"   ✓ Original Fake/True: {len(df_original)} articles")
    datasets.append(df_original)
    dataset_names.append('Original')
except Exception as e:
    print(f"   ✗ Original error: {e}")

# 3. GossipCop
try:
    df_gossip_fake = pd.read_csv('gossipcop_fake.csv')[['text']].copy()
    df_gossip_fake['label'] = 0
    df_gossip_real = pd.read_csv('gossipcop_real.csv')[['text']].copy()
    df_gossip_real['label'] = 1
    df_gossip = pd.concat([df_gossip_fake, df_gossip_real], ignore_index=True)
    print(f"   ✓ GossipCop: {len(df_gossip)} articles")
    datasets.append(df_gossip)
    dataset_names.append('GossipCop')
except Exception as e:
    print(f"   ✗ GossipCop error: {e}")

# 4. PolitiFact
try:
    df_politi_fake = pd.read_csv('politifact_fake.csv')[['text']].copy()
    df_politi_fake['label'] = 0
    df_politi_real = pd.read_csv('politifact_real.csv')[['text']].copy()
    df_politi_real['label'] = 1
    df_politi = pd.concat([df_politi_fake, df_politi_real], ignore_index=True)
    print(f"   ✓ PolitiFact: {len(df_politi)} articles")
    datasets.append(df_politi)
    dataset_names.append('PolitiFact')
except Exception as e:
    print(f"   ✗ PolitiFact error: {e}")

# 5. RSS News
try:
    df_rss = pd.read_csv('rss_news.csv')[['text', 'label']].copy()
    print(f"   ✓ RSS News: {len(df_rss)} articles")
    datasets.append(df_rss)
    dataset_names.append('RSS')
except Exception as e:
    print(f"   ✗ RSS error: {e}")

if not datasets:
    print("❌ No datasets loaded!")
    exit(1)

# Combine all datasets
print(f"\n   Combining {len(datasets)} datasets...")
df_combined = pd.concat(datasets, ignore_index=True)

# Remove duplicates
df_combined = df_combined.drop_duplicates(subset=['text'])
df_combined = df_combined.dropna(subset=['text'])
df_combined['text'] = df_combined['text'].astype(str).str.strip()
df_combined = df_combined[df_combined['text'].str.len() > 10]

print(f"   ✓ Combined: {len(df_combined)} unique articles")
print(f"   ✓ Fake: {(df_combined['label'] == 0).sum()}")
print(f"   ✓ Real: {(df_combined['label'] == 1).sum()}")

# ============================================================================
# STEP 2: STRATIFIED SAMPLING FOR BALANCE
# ============================================================================
print("\n[2/8] Balancing dataset...")

# Stratified sampling to balance fake/real
min_count = min((df_combined['label'] == 0).sum(), (df_combined['label'] == 1).sum())
target_size = min(50000, min_count)  # Cap at 50k per class for speed

df_fake_balanced = df_combined[df_combined['label'] == 0].sample(
    n=min(target_size, (df_combined['label'] == 0).sum()),
    random_state=42
)
df_real_balanced = df_combined[df_combined['label'] == 1].sample(
    n=min(target_size, (df_combined['label'] == 1).sum()),
    random_state=42
)

df_balanced = pd.concat([df_fake_balanced, df_real_balanced], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   ✓ Balanced dataset: {len(df_balanced)} articles")
print(f"   ✓ Fake: {(df_balanced['label'] == 0).sum()}")
print(f"   ✓ Real: {(df_balanced['label'] == 1).sum()}")

# ============================================================================
# STEP 3: TEXT PREPROCESSING & VECTORIZATION
# ============================================================================
print("\n[3/8] Vectorizing text with TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=2000,
    min_df=5,
    max_df=0.95,
    ngram_range=(1, 2),
    stop_words='english',
    lowercase=True,
    strip_accents='unicode'
)

X = vectorizer.fit_transform(df_balanced['text'])
y = df_balanced['label'].values

print(f"   ✓ Feature matrix shape: {X.shape}")
print(f"   ✓ Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n[4/8] Splitting data (80-20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Training set: {len(y_train)} articles")
print(f"   ✓ Test set: {len(y_test)} articles")

# ============================================================================
# STEP 5: TRAIN INDIVIDUAL MODELS
# ============================================================================
print("\n[5/8] Training 5 ensemble models...")

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'NaiveBayes': MultinomialNB()
}

trained_models = {}
for name, model in models.items():
    print(f"   Training {name}...", end=" ")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"✓ {score*100:.2f}%")
    trained_models[name] = model

# ============================================================================
# STEP 6: CREATE VOTING ENSEMBLE
# ============================================================================
print("\n[6/8] Creating voting ensemble...")

voting_clf = VotingClassifier(
    estimators=[
        ('lr', trained_models['LogisticRegression']),
        ('rf', trained_models['RandomForest']),
        ('gb', trained_models['GradientBoosting']),
        ('xgb', trained_models['XGBoost']),
        ('nb', trained_models['NaiveBayes'])
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
ensemble_score = voting_clf.score(X_test, y_test)
print(f"   ✓ Ensemble accuracy: {ensemble_score*100:.2f}%")

# ============================================================================
# STEP 7: DETAILED EVALUATION
# ============================================================================
print("\n[7/8] Detailed evaluation...")

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

y_pred = voting_clf.predict(X_test)

accuracy = (y_pred == y_test).mean()
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"   Accuracy:  {accuracy*100:.2f}%")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"\n   Confusion Matrix:")
print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")

# ============================================================================
# STEP 8: SAVE MODELS
# ============================================================================
print("\n[8/8] Saving models...")

# Save ensemble model
with open('model_production.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)
print(f"   ✓ Ensemble model saved: model_production.pkl")

# Save vectorizer
with open('vectorizer_production.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"   ✓ Vectorizer saved: vectorizer_production.pkl")

# Save individual models
with open('individual_models.pkl', 'wb') as f:
    pickle.dump(trained_models, f)
print(f"   ✓ Individual models saved: individual_models.pkl")

# Create metadata
metadata = {
    'datasets_used': dataset_names,
    'total_articles_trained': len(df_balanced),
    'vocabulary_size': len(vectorizer.get_feature_names_out()),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'test_set_size': len(y_test),
    'ngrams': (1, 2),
    'max_features': 2000,
    'individual_model_scores': {
        name: float(trained_models[name].score(X_test, y_test))
        for name in trained_models
    }
}

import json
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Metadata saved: model_metadata.json")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\n📊 Final Model Statistics:")
print(f"   Total training articles: {len(df_balanced):,}")
print(f"   Datasets combined: {', '.join(dataset_names)}")
print(f"   Ensemble accuracy: {accuracy*100:.2f}%")
print(f"   Models saved and ready for deployment")
print("\n🚀 Ready to run: streamlit run app_enhanced.py")
print("="*80)

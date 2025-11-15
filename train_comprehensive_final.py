"""
COMPREHENSIVE MODEL TRAINING SCRIPT
====================================
Trains on ALL available datasets:
1. Original Fake.csv & True.csv (44,898 articles)
2. GossipCop (gossipcop_fake.csv + gossipcop_real.csv)
3. PolitiFact (politifact_fake.csv + politifact_real.csv)
4. RSS News (NEW - rss_news.csv - labeled as REAL)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

print("=" * 70)
print("TRAINING FAKE NEWS DETECTION MODEL")
print("=" * 70)

# ============================================================================
# LOAD ALL DATA
# ============================================================================

all_data = []

# 1. ORIGINAL DATA
print("\n1️⃣  Loading Original Data (Fake.csv & True.csv)...")
try:
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    
    fake_df['label'] = 0  # 0 = Fake
    true_df['label'] = 1  # 1 = Real
    
    # Use text column
    fake_df['text'] = fake_df['text'].astype(str)
    true_df['text'] = true_df['text'].astype(str)
    
    all_data.append(fake_df[['text', 'label']])
    all_data.append(true_df[['text', 'label']])
    
    print(f"   ✅ Fake: {len(fake_df)} articles")
    print(f"   ✅ Real: {len(true_df)} articles")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. GOSSIPCOP DATA
print("\n2️⃣  Loading GossipCop Data...")
try:
    gossip_fake = pd.read_csv('gossipcop_fake.csv')
    gossip_real = pd.read_csv('gossipcop_real.csv')
    
    gossip_fake['label'] = 0
    gossip_real['label'] = 1
    
    # Find text column
    text_col = 'text' if 'text' in gossip_fake.columns else 'content' if 'content' in gossip_fake.columns else gossip_fake.columns[0]
    
    gossip_fake['text'] = gossip_fake[text_col].astype(str)
    gossip_real['text'] = gossip_real[text_col].astype(str)
    
    all_data.append(gossip_fake[['text', 'label']])
    all_data.append(gossip_real[['text', 'label']])
    
    print(f"   ✅ GossipCop Fake: {len(gossip_fake)} articles")
    print(f"   ✅ GossipCop Real: {len(gossip_real)} articles")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. POLITIFACT DATA
print("\n3️⃣  Loading PolitiFact Data...")
try:
    politi_fake = pd.read_csv('politifact_fake.csv')
    politi_real = pd.read_csv('politifact_real.csv')
    
    politi_fake['label'] = 0
    politi_real['label'] = 1
    
    text_col = 'text' if 'text' in politi_fake.columns else 'content' if 'content' in politi_fake.columns else politi_fake.columns[0]
    
    politi_fake['text'] = politi_fake[text_col].astype(str)
    politi_real['text'] = politi_real[text_col].astype(str)
    
    all_data.append(politi_fake[['text', 'label']])
    all_data.append(politi_real[['text', 'label']])
    
    print(f"   ✅ PolitiFact Fake: {len(politi_fake)} articles")
    print(f"   ✅ PolitiFact Real: {len(politi_real)} articles")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. RSS NEWS DATA (NEW)
print("\n4️⃣  Loading RSS News Data (NEW)...")
try:
    rss_df = pd.read_csv('rss_news.csv')
    rss_df['label'] = 1  # Label as REAL (reputable RSS sources)
    
    # Combine title and content for better text
    rss_df['text'] = (rss_df['title'].astype(str) + " " + rss_df['content'].astype(str))
    
    all_data.append(rss_df[['text', 'label']])
    
    print(f"   ✅ RSS News: {len(rss_df)} articles (labeled as REAL)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ============================================================================
# COMBINE ALL DATA
# ============================================================================

print("\n5️⃣  Combining All Data...")
df_combined = pd.concat(all_data, ignore_index=True)

# Remove empty texts
df_combined = df_combined[df_combined['text'].str.len() > 10]

print(f"   ✅ Total articles: {len(df_combined)}")
print(f"   ✅ Fake (0): {(df_combined['label'] == 0).sum()}")
print(f"   ✅ Real (1): {(df_combined['label'] == 1).sum()}")

# Shuffle
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

X = df_combined['text'].values
y = df_combined['label'].values

# ============================================================================
# VECTORIZATION
# ============================================================================

print("\n6️⃣  Vectorizing Text (TF-IDF)...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    lowercase=True,
    stop_words='english'
)

X_vectorized = vectorizer.fit_transform(X)
print(f"   ✅ Vectorized shape: {X_vectorized.shape}")

# ============================================================================
# TRAIN MODELS
# ============================================================================

print("\n7️⃣  Training Individual Models...")

# Model 1: PassiveAggressive
print("   • PassiveAggressive Classifier...")
pa_model = PassiveAggressiveClassifier(n_iter_no_change=5, random_state=42, max_iter=100)
pa_model.fit(X_vectorized, y)
print(f"     ✅ Accuracy: {pa_model.score(X_vectorized, y):.4f}")

# Model 2: RandomForest
print("   • Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_vectorized, y)
print(f"     ✅ Accuracy: {rf_model.score(X_vectorized, y):.4f}")

# Model 3: LinearSVC
print("   • Linear SVM...")
svm_model = LinearSVC(random_state=42, max_iter=2000, dual='auto')
svm_model.fit(X_vectorized, y)
print(f"     ✅ Accuracy: {svm_model.score(X_vectorized, y):.4f}")

# Model 4: Naive Bayes
print("   • Multinomial Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_vectorized, y)
print(f"     ✅ Accuracy: {nb_model.score(X_vectorized, y):.4f}")

# Model 5: XGBoost
print("   • XGBoost Classifier...")
xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                          random_state=42, n_jobs=-1)
xgb_model.fit(X_vectorized, y)
print(f"     ✅ Accuracy: {xgb_model.score(X_vectorized, y):.4f}")

# ============================================================================
# ENSEMBLE MODEL
# ============================================================================

print("\n8️⃣  Creating Ensemble Model (Voting)...")

ensemble = VotingClassifier(
    estimators=[
        ('pa', pa_model),
        ('rf', rf_model),
        ('svm', svm_model),
        ('nb', nb_model),
        ('xgb', xgb_model)
    ],
    voting='soft',
    n_jobs=-1
)

ensemble.fit(X_vectorized, y)
ensemble_accuracy = ensemble.score(X_vectorized, y)
print(f"   ✅ Ensemble Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n9️⃣  Saving Models...")

# Create model directory
model_dir = Path('model_artifacts_comprehensive')
model_dir.mkdir(exist_ok=True)

# Save ensemble and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
print("   ✅ Saved: model.pkl (ensemble)")

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("   ✅ Saved: vectorizer.pkl")

# Also save individual models in model_artifacts
with open(model_dir / 'ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
with open(model_dir / 'vectorizer_model.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open(model_dir / 'pa_model.pkl', 'wb') as f:
    pickle.dump(pa_model, f)
with open(model_dir / 'rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open(model_dir / 'svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open(model_dir / 'nb_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)
with open(model_dir / 'xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

print(f"   ✅ Saved all models in: {model_dir}/")

# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)

print(f"""
📊 TRAINING STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Articles Trained On: {len(df_combined):,}
├─ Fake News: {(y == 0).sum():,}
├─ Real News: {(y == 1).sum():,}
└─ Datasets: 4 (Original + GossipCop + PolitiFact + RSS)

Vectorizer Configuration:
├─ Max Features: 5000
├─ N-gram: (1, 2)
└─ Shape: {X_vectorized.shape}

Model Performance (Training Accuracy):
├─ PassiveAggressive: {pa_model.score(X_vectorized, y):.4f}
├─ Random Forest: {rf_model.score(X_vectorized, y):.4f}
├─ Linear SVM: {svm_model.score(X_vectorized, y):.4f}
├─ Naive Bayes: {nb_model.score(X_vectorized, y):.4f}
├─ XGBoost: {xgb_model.score(X_vectorized, y):.4f}
└─ 🏆 Ensemble (Voting): {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)

✅ Models saved to:
   • model.pkl (main ensemble)
   • vectorizer.pkl
   • {model_dir}/ (all models)
""")

print("=" * 70)
print("🚀 Ready to use with app_best.py!")
print("=" * 70)

"""
ULTRA-FAST WELFAKE TRAINING
============================
Minimal features, smaller sample for immediate deployment
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
import pickle
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("ULTRA-FAST WELFAKE TRAINING")
print("="*70)

# Load WELFake (just use this one for speed)
print("\n[1/5] Loading WELFake dataset...")
df = pd.read_csv('WELFake_Dataset.csv')[['title', 'text', 'label']].copy()
df['combined_text'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).str.strip()
df = df[['combined_text', 'label']].rename(columns={'combined_text': 'text'})
df = df.drop_duplicates(subset=['text']).dropna(subset=['text'])
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len() > 10]

print(f"✓ Total: {len(df)}")
print(f"✓ Fake: {(df['label'] == 0).sum()}")
print(f"✓ Real: {(df['label'] == 1).sum()}")

# Sample for speed
print("\n[2/5] Sampling (10k per class)...")
df_fake = df[df['label'] == 0].sample(n=10000, random_state=42)
df_real = df[df['label'] == 1].sample(n=10000, random_state=42)
df_train = pd.concat([df_fake, df_real], ignore_index=True).sample(frac=1, random_state=42)

print(f"✓ Training set: {len(df_train)}")

# Vectorize (minimal)
print("\n[3/5] Vectorizing (500 features)...")
vectorizer = TfidfVectorizer(
    max_features=500,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2),
    stop_words='english'
)

X = vectorizer.fit_transform(df_train['text'])
y = df_train['label'].values

print(f"✓ Shape: {X.shape}")

# Train-test split
print("\n[4/5] Training ensemble...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = [
    ('lr', LogisticRegression(max_iter=300, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)),
    ('nb', MultinomialNB())
]

for name, model in models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"  {name}: {score*100:.2f}%")

voting_clf = VotingClassifier(estimators=models, voting='soft')
voting_clf.fit(X_train, y_train)

accuracy = voting_clf.score(X_test, y_test)

print(f"\n✓ Ensemble accuracy: {accuracy*100:.2f}%")

# Save
print("\n[5/5] Saving models...")
with open('model_production.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)
with open('vectorizer_production.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

import json
with open('model_metadata.json', 'w') as f:
    json.dump({
        'source': 'WELFake Dataset',
        'articles': len(df_train),
        'accuracy': float(accuracy),
        'features': 500
    }, f)

print("✓ Models saved!")
print("\n✅ DONE! Ready for deployment.")
print(f"Accuracy: {accuracy*100:.2f}%")
